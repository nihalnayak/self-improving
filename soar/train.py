"""Main SOAR outer loop.

Read top-to-bottom: 6 numbered blocks per outer step.

Block 1: TEACHER PROPOSES — generate G candidate problems with the teacher adapter.
Block 2: BASELINE EVAL   — measure current student's accuracy on Q_eval.
Block 3: PER-CANDIDATE   — for each candidate: snapshot student, inner-train, eval, restore.
Block 4: TEACHER UPDATE  — one GRPO step on the teacher using (candidate, improvement) rewards.
Block 5: LOGGING         — write wandb metrics.
Block 6: CHECKPOINT      — occasionally save both adapters.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from statistics import mean

import torch
import wandb

from soar.config import SoarConfig, smoke_config
from soar.data import load_mmlu_pro_split, format_mcq_prompt, MCQ
from soar.grpo import grpo_loss, tokenize_rollouts
from soar.models import chat_prompt, load_base_and_adapters, load_vllm_engine
from soar.rollout import sync_adapter_to_disk, vllm_generate, cleanup_old_adapters
from soar.student import inner_train_student, restore_student_lora, snapshot_student_lora
from soar.teacher import build_teacher_prompt, parse_teacher_output
from soar.verify import score_answer


# ------------- helpers used by the main loop -------------

def _device(model) -> str:
    return next(model.parameters()).device.type


def _teacher_params(peft_model):
    # Filter by name only — PEFT toggles requires_grad when set_adapter() switches heads,
    # but the tensor objects stay the same, so the optimizer refs remain valid.
    return [p for n, p in peft_model.named_parameters() if ".teacher." in n]


def _student_params(peft_model):
    return [p for n, p in peft_model.named_parameters() if ".student." in n]


def eval_student_on(engine, lora_path: str, questions: list[MCQ], tok, cfg: SoarConfig) -> float:
    """Greedy accuracy on a fixed set of MCQs using the given student LoRA path."""
    prompts = [chat_prompt(tok, format_mcq_prompt(q)) for q in questions]
    completions = vllm_generate(
        engine, prompts, lora_path=lora_path, lora_name="student",
        max_new_tokens=cfg.student_max_new_tokens, temperature=0.0,
    )
    scores = [score_answer(c, q.answer) for c, q in zip(completions, questions)]
    return float(mean(scores)) if scores else 0.0


# ------------- main -------------

def main(cfg: SoarConfig):
    # ---- 0. Setup ----
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir) / "config.json", "w") as f:
        json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}, f, indent=2)

    print(f"[setup] loading model: {cfg.base_model}")
    model, tok = load_base_and_adapters(cfg)
    print("[setup] loading vLLM engine")
    engine = load_vllm_engine(cfg)
    print("[setup] loading data")
    q_eval, q_train = load_mmlu_pro_split(cfg.subject, cfg.n_eval_questions, cfg.seed)
    print(f"[setup] |Q_eval|={len(q_eval)}  |Q_train_pool|={len(q_train)}")

    teacher_opt = torch.optim.AdamW(_teacher_params(model), lr=cfg.teacher_lr)
    student_opt = torch.optim.AdamW(_student_params(model), lr=cfg.student_lr)

    wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode, config=cfg.__dict__)
    rng = random.Random(cfg.seed)
    dev = _device(model)

    # ---- outer loop ----
    for outer_step in range(cfg.n_outer_steps):
        t_start = time.time()

        # 1. TEACHER PROPOSES G candidate problems
        teacher_user = build_teacher_prompt(q_train, k=cfg.n_train_exemplars, rng=rng)
        teacher_prompt_str = chat_prompt(tok, teacher_user)
        teacher_lora_path = sync_adapter_to_disk(model, "teacher", cfg.out_dir, f"outer{outer_step}")
        raw_candidates = vllm_generate(
            engine, [teacher_prompt_str] * cfg.group_size,
            lora_path=teacher_lora_path, lora_name="teacher",
            max_new_tokens=cfg.teacher_max_new_tokens, temperature=cfg.teacher_temperature,
        )
        problems = [parse_teacher_output(t) for t in raw_candidates]
        n_valid = sum(p is not None for p in problems)
        print(f"[step {outer_step}] teacher proposed {cfg.group_size}, parsed {n_valid}")

        # 2. BASELINE STUDENT ACCURACY on Q_eval (before any inner training)
        student_lora_path = sync_adapter_to_disk(model, "student", cfg.out_dir, f"outer{outer_step}_base")
        baseline_acc = eval_student_on(engine, student_lora_path, q_eval, tok, cfg)
        print(f"[step {outer_step}] baseline student acc = {baseline_acc:.3f}")

        # 3. PER-CANDIDATE: snapshot student LoRA, inner-train, eval, restore
        snapshot = snapshot_student_lora(model)
        teacher_rewards = []
        for i, prob in enumerate(problems):
            if prob is None:
                teacher_rewards.append(0.0)
                continue
            inner_stats = inner_train_student(
                model, tok, engine, cfg, student_opt, prob, tag=f"o{outer_step}_c{i}",
            )
            trained_path = sync_adapter_to_disk(model, "student", cfg.out_dir, f"o{outer_step}_c{i}_trained")
            new_acc = eval_student_on(engine, trained_path, q_eval, tok, cfg)
            teacher_rewards.append(new_acc - baseline_acc)
            restore_student_lora(model, snapshot)
            print(f"[step {outer_step}] cand {i}: acc {new_acc:.3f} (Δ{new_acc - baseline_acc:+.3f}) inner={inner_stats.get('mean_reward', 0):.2f}")

        # 4. TEACHER GRPO UPDATE
        if n_valid > 0:
            teacher_batch_prompts = [teacher_prompt_str] * cfg.group_size
            batch = tokenize_rollouts(
                tok, teacher_batch_prompts, raw_candidates, device=dev,
                max_prompt_len=cfg.max_prompt_len,
                max_total_len=cfg.max_prompt_len + cfg.teacher_max_new_tokens,
            )
            batch.rewards = torch.tensor(teacher_rewards, dtype=torch.float32, device=dev)
            model.set_adapter("teacher")
            model.train()
            out = grpo_loss(model, batch, group_size=cfg.group_size, kl_beta=cfg.kl_beta)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(_teacher_params(model), 1.0)
            teacher_opt.step()
            teacher_opt.zero_grad(set_to_none=True)
            teacher_metrics = {f"teacher/{k}": (v.item() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
        else:
            teacher_metrics = {"teacher/skipped": 1.0}

        # 5. LOGGING
        dt = time.time() - t_start
        log = {
            "outer_step": outer_step,
            "baseline_acc": baseline_acc,
            "mean_reward": float(mean(teacher_rewards)) if teacher_rewards else 0.0,
            "max_reward": float(max(teacher_rewards)) if teacher_rewards else 0.0,
            "valid_problems": n_valid,
            "step_sec": dt,
            **teacher_metrics,
        }
        wandb.log(log, step=outer_step)
        print(f"[step {outer_step}] mean_r={log['mean_reward']:+.3f} max_r={log['max_reward']:+.3f} dt={dt:.1f}s")

        # 6. CHECKPOINT (and keep adapter dir bounded)
        if outer_step % cfg.ckpt_every == 0:
            ckpt_dir = Path(cfg.out_dir) / f"ckpt_outer{outer_step}"
            for name in ("teacher", "student"):
                model.set_adapter(name)
                model.save_pretrained(str(ckpt_dir), selected_adapters=[name])
        cleanup_old_adapters(cfg.out_dir, keep_last=8)

        # [Promotion mechanism goes here — stub:]
        # if max(teacher_rewards) > cfg.promotion_threshold:
        #     i = int(torch.tensor(teacher_rewards).argmax())
        #     inner_train_student(model, tok, engine, cfg, student_opt, problems[i], tag=f"promo{outer_step}")

    wandb.finish()
    print("[done]")


# ------------- CLI -------------

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny config end-to-end sanity run")
    ap.add_argument("--out", type=str, default=None, help="override out_dir")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = smoke_config() if args.smoke else SoarConfig()
    if args.out:
        cfg.out_dir = args.out
    main(cfg)
