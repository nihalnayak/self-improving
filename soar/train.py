"""SOAR outer loop (Algorithm 1, SOAR paper page 21).

Per outer step:
    1. TEACHER GENERATES g*n parsed problems (rejection sampling, §B.2 Prop. 1).
    2. BASELINE EVAL          — baseline student accuracy on freshly subsampled Q_R.
    3. INNER TRAINING         — for each (X_k, j in 1..r): restore baseline -> inner-train
                                 on X_k -> eval on Q_R -> R_{k,j} = Δacc.
    4. PROMOTION CHECK        — rolling-m MA of R̄_t = mean_k R_k; if MA > τ, promote
                                 baseline student to the median-reward run of argmax_k R_k.
    5. TEACHER RLOO UPDATE    — dataset-level LOO advantages over g rewards, broadcast
                                 across each X_k's n constituent rollouts.
    6. LOGGING + CHECKPOINT.
"""

import argparse
import json
import random
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from typing import List, Optional, Tuple

import torch
import wandb

from soar.config import SoarConfig, smoke_config
from soar.data import MathProblem, load_fail128_split
from soar.models import chat_prompt, load_base_and_adapters, load_vllm_engine, student_chat_prompt
from soar.rloo import rloo_advantages, rloo_loss, tokenize_rollouts
from soar.rollout import cleanup_old_adapters, sync_adapter_to_disk, vllm_generate
from soar.student import (
    inner_train_student_on_dataset, restore_student_lora, snapshot_student_lora,
)
from soar.teacher import build_teacher_prompt, parse_teacher_output
from soar.verify import success_math_answer


# ------------- helpers -------------

def _device(model) -> str:
    return next(model.parameters()).device.type


def _teacher_params(peft_model):
    return [p for n, p in peft_model.named_parameters() if ".teacher." in n]


def _student_params(peft_model):
    return [p for n, p in peft_model.named_parameters() if ".student." in n]


def eval_student_on(engine, lora_path: str, questions: List[MathProblem], tok,
                    cfg: SoarConfig) -> float:
    """Greedy pass@1 success rate on `questions` using the given student LoRA path."""
    prompts = [student_chat_prompt(tok, q.problem) for q in questions]
    completions = vllm_generate(
        engine, prompts, lora_path=lora_path, lora_name="student",
        max_new_tokens=cfg.student_max_new_tokens, temperature=0.0,
    )
    scores = [1.0 if success_math_answer(c, q.answer) else 0.0
              for c, q in zip(completions, questions)]
    return float(mean(scores)) if scores else 0.0


def generate_teacher_batch(
    engine, teacher_prompt_str: str, teacher_lora_path: str, cfg: SoarConfig,
    needed: int,
) -> Tuple[List[str], List[MathProblem]]:
    """Rejection-sample teacher completions until we have `needed` parsed problems, or until
    we exhaust the per-step budget. Returns (accepted_raw_completions, accepted_problems).
    """
    budget = cfg.teacher_rejection_cap * needed
    accepted_raw: List[str] = []
    accepted_problems: List[MathProblem] = []
    total_generated = 0
    while len(accepted_problems) < needed and total_generated < budget:
        batch = min(needed - len(accepted_problems), needed)
        raw = vllm_generate(
            engine, [teacher_prompt_str] * batch,
            lora_path=teacher_lora_path, lora_name="teacher",
            max_new_tokens=cfg.teacher_max_new_tokens, temperature=cfg.teacher_temperature,
        )
        total_generated += len(raw)
        for r in raw:
            p = parse_teacher_output(r)
            if p is not None:
                accepted_raw.append(r)
                accepted_problems.append(p)
                if len(accepted_problems) >= needed:
                    break
    return accepted_raw, accepted_problems


def append_promotion_questions(out_dir: str, promotion_idx: int, X_kstar: List[MathProblem]):
    """Append the promoted dataset to {out_dir}/promotion_questions.jsonl (one problem per line
    with an added `promotion_idx` field)."""
    path = Path(out_dir) / "promotion_questions.jsonl"
    with open(path, "a") as f:
        for p in X_kstar:
            d = asdict(p)
            d["promotion_idx"] = promotion_idx
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


# ------------- main -------------

def main(cfg: SoarConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir) / "config.json", "w") as f:
        json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
                  f, indent=2)

    print(f"[setup] loading model: {cfg.base_model}")
    model, tok = load_base_and_adapters(cfg)
    print("[setup] loading vLLM engine")
    engine = load_vllm_engine(cfg)
    print(f"[setup] loading fail@128 split from {cfg.fail128_dir}")
    train_set, test_set = load_fail128_split(cfg.fail128_dir)
    print(f"[setup] |train|={len(train_set)}  |test|={len(test_set)}")
    assert len(train_set) >= cfg.n_reward_questions, "not enough train problems for Q_R"

    teacher_opt = torch.optim.AdamW(_teacher_params(model), lr=cfg.teacher_lr)
    student_opt = torch.optim.AdamW(_student_params(model), lr=cfg.student_lr)

    wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode, config=cfg.__dict__)
    dev = _device(model)

    # The persistent baseline student snapshot. Starts as the freshly-initialised student LoRA.
    baseline_snapshot = snapshot_student_lora(model)
    promotions_window: deque[float] = deque(maxlen=cfg.promotion_window)
    n_promotions = 0

    g, n_per = cfg.outer_group_size, cfg.inner_dataset_size
    r_parallel = cfg.r_parallel_students
    needed_teacher = g * n_per

    for outer_step in range(cfg.n_outer_steps):
        t_start = time.time()

        # ----- 1. TEACHER GENERATES g*n parsed problems -----
        teacher_user = build_teacher_prompt()
        teacher_prompt_str = chat_prompt(tok, teacher_user)
        teacher_lora_path = sync_adapter_to_disk(model, "teacher", cfg.out_dir, f"outer{outer_step}")
        accepted_raw, accepted_problems = generate_teacher_batch(
            engine, teacher_prompt_str, teacher_lora_path, cfg, needed=needed_teacher,
        )
        if len(accepted_problems) < needed_teacher:
            print(f"[step {outer_step}] teacher parse yield too low "
                  f"({len(accepted_problems)}/{needed_teacher}) — skipping outer step")
            wandb.log({"outer_step": outer_step, "skipped": 1}, step=outer_step)
            cleanup_old_adapters(cfg.out_dir, keep_last=8)
            continue
        X = [accepted_problems[k * n_per:(k + 1) * n_per] for k in range(g)]
        print(f"[step {outer_step}] teacher produced g*n={needed_teacher} parsed problems")

        # ----- 2. BASELINE EVAL on Q_R -----
        Q_R = rng.sample(train_set, cfg.n_reward_questions)
        # Push baseline weights to disk and eval
        restore_student_lora(model, baseline_snapshot)
        baseline_lora_path = sync_adapter_to_disk(
            model, "student", cfg.out_dir, f"outer{outer_step}_baseline",
        )
        base_acc = eval_student_on(engine, baseline_lora_path, Q_R, tok, cfg)
        print(f"[step {outer_step}] baseline acc on Q_R = {base_acc:.3f}")

        # ----- 3. INNER TRAINING: g datasets × r parallel runs -----
        R_matrix: List[List[float]] = [[0.0] * r_parallel for _ in range(g)]
        # Keep snapshots of each (k, j) trained student in case we promote.
        trained_snapshots: List[List[Optional[dict]]] = [[None] * r_parallel for _ in range(g)]
        for k in range(g):
            for j in range(r_parallel):
                restore_student_lora(model, baseline_snapshot)
                tag = f"o{outer_step}_k{k}_j{j}"
                inner_train_student_on_dataset(
                    model, tok, engine, cfg, student_opt, X[k], tag=tag, rng=rng,
                )
                trained_path = sync_adapter_to_disk(model, "student", cfg.out_dir, f"{tag}_trained")
                acc = eval_student_on(engine, trained_path, Q_R, tok, cfg)
                R_matrix[k][j] = acc - base_acc
                trained_snapshots[k][j] = snapshot_student_lora(model)
                print(f"[step {outer_step}] X_{k} run{j}: acc={acc:.3f} Δ={acc - base_acc:+.3f}")

        R_k = [float(mean(row)) for row in R_matrix]
        R_bar_t = float(mean(R_k))
        print(f"[step {outer_step}] R_k = {['%.3f' % r for r in R_k]}  R̄_t = {R_bar_t:+.3f}")

        # ----- 4. PROMOTION CHECK -----
        promoted_this_step = 0
        promotions_window.append(R_bar_t)
        ma = float(mean(promotions_window)) if len(promotions_window) == cfg.promotion_window else -float("inf")
        if ma > cfg.promotion_threshold and n_promotions < cfg.max_promotions_to_track:
            k_star = int(max(range(g), key=lambda k: R_k[k]))
            # Pick the median-reward run across the r parallel runs on X_{k*}.
            rewards_kstar = R_matrix[k_star]
            # argmedian: index whose value equals the sorted-median
            sorted_j = sorted(range(r_parallel), key=lambda j: rewards_kstar[j])
            j_star = sorted_j[r_parallel // 2]
            baseline_snapshot = {name: t.detach().clone()
                                 for name, t in trained_snapshots[k_star][j_star].items()}
            n_promotions += 1
            append_promotion_questions(cfg.out_dir, n_promotions, X[k_star])
            promotions_window.clear()
            promoted_this_step = 1
            print(f"[step {outer_step}] PROMOTED (k*={k_star}, j*={j_star}, MA={ma:+.3f})  "
                  f"total promotions = {n_promotions}")

        # ----- 5. TEACHER RLOO UPDATE -----
        # Dataset-level LOO advantages over g rewards, broadcast to each X_k's n rollouts.
        R_k_tensor = torch.tensor(R_k, dtype=torch.float32, device=dev)
        if g >= 2 and (R_k_tensor.max() - R_k_tensor.min()).item() > 1e-6:
            dataset_adv = rloo_advantages(R_k_tensor, group_size=g)        # [g]
            per_rollout_adv = dataset_adv.repeat_interleave(n_per)         # [g*n]
            teacher_batch_prompts = [teacher_prompt_str] * needed_teacher
            batch = tokenize_rollouts(
                tok, teacher_batch_prompts, accepted_raw, device=dev,
                max_prompt_len=cfg.max_prompt_len,
                max_total_len=cfg.max_prompt_len + cfg.teacher_max_new_tokens,
            )
            model.set_adapter("teacher")
            model.train()
            out = rloo_loss(model, batch, advantages=per_rollout_adv, kl_beta=cfg.kl_beta)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(_teacher_params(model), 1.0)
            teacher_opt.step()
            teacher_opt.zero_grad(set_to_none=True)
            teacher_metrics = {f"teacher/{k}": (v.item() if isinstance(v, torch.Tensor) else v)
                               for k, v in out.items()}
        else:
            teacher_metrics = {"teacher/skipped": 1.0}

        # ----- 6. LOGGING + CHECKPOINT -----
        dt = time.time() - t_start
        log = {
            "outer_step": outer_step,
            "baseline_acc": base_acc,
            "teacher/dataset_reward_mean": R_bar_t,
            "teacher/dataset_reward_max": float(max(R_k)),
            "teacher/dataset_reward_min": float(min(R_k)),
            "promotion/count_total": n_promotions,
            "promotion/triggered_this_step": promoted_this_step,
            "promotion/moving_avg": ma if ma != -float("inf") else 0.0,
            "step_sec": dt,
            **teacher_metrics,
        }
        wandb.log(log, step=outer_step)
        print(f"[step {outer_step}] R̄_t={R_bar_t:+.3f} promo={promoted_this_step} dt={dt:.1f}s")

        if outer_step % cfg.ckpt_every == 0:
            ckpt_dir = Path(cfg.out_dir) / f"ckpt_outer{outer_step}"
            for name in ("teacher", "student"):
                model.set_adapter(name)
                model.save_pretrained(str(ckpt_dir), selected_adapters=[name])
            # Also persist the baseline student snapshot (not the live student weights).
            restore_student_lora(model, baseline_snapshot)
            model.set_adapter("student")
            model.save_pretrained(str(ckpt_dir / "baseline_student"),
                                  selected_adapters=["student"])
        cleanup_old_adapters(cfg.out_dir, keep_last=8)

    wandb.finish()
    print(f"[done] total promotions = {n_promotions}")


# ------------- CLI -------------

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end sanity run")
    ap.add_argument("--out", type=str, default=None, help="override out_dir")
    ap.add_argument("--fail128-dir", type=str, default=None, help="override fail128_dir")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = smoke_config() if args.smoke else SoarConfig()
    if args.out:
        cfg.out_dir = args.out
    if args.fail128_dir:
        cfg.fail128_dir = args.fail128_dir
    main(cfg)
