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

from soar import dist_utils
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


def _log(*args, **kwargs):
    """Rank-0-only print; silent on other ranks."""
    if dist_utils.is_main():
        print(*args, **kwargs)


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
) -> Tuple[List[str], List[MathProblem], int]:
    """Rejection-sample teacher completions until we have `needed` parsed problems, or until
    we exhaust the per-step budget. Returns (accepted_raw_completions, accepted_problems,
    total_generated).
    """
    budget = cfg.teacher_rejection_cap * needed
    accepted_raw: List[str] = []
    accepted_problems: List[MathProblem] = []
    total_generated = 0
    rounds = 0
    while len(accepted_problems) < needed and total_generated < budget:
        batch = min(needed - len(accepted_problems), needed)

        # minor modication
        _teacher_prompt_str = teacher_prompt_str + "<question>"

        raw = vllm_generate(
            engine, [_teacher_prompt_str] * batch,
            lora_path=teacher_lora_path, lora_name="teacher",
            max_new_tokens=cfg.teacher_max_new_tokens, temperature=cfg.teacher_temperature,
            top_p=0.7
        )
        total_generated += len(raw)
        rounds += 1
        for r in raw:
            p = parse_teacher_output("<question>" + r)
            if p is not None:
                accepted_raw.append(r)
                accepted_problems.append(p)
                if len(accepted_problems) >= needed:
                    break
        print(f"  [teacher-gen] round {rounds}: generated {total_generated} "
              f"accepted {len(accepted_problems)}/{needed} "
              f"(yield {len(accepted_problems) / max(total_generated, 1):.2%})")
    return accepted_raw, accepted_problems, total_generated


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
    dist_utils.init()

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    if dist_utils.is_main():
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(cfg.out_dir) / "config.json", "w") as f:
            json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
                      f, indent=2)
    dist_utils.barrier()

    _log(f"[setup] loading model: {cfg.base_model}  "
         f"(world_size={dist_utils.world_size()}, rank={dist_utils.rank()})")
    model, tok = load_base_and_adapters(cfg)
    # vLLM lives on rank 0 only — other ranks sit idle during rollouts (per plan).
    if dist_utils.is_main():
        _log("[setup] loading vLLM engine")
        engine = load_vllm_engine(cfg)
    else:
        engine = None
    # Hold non-main ranks here until rank 0 finishes vLLM init; otherwise they
    # race into the first FSDP collective below and silently block.
    dist_utils.barrier()
    _log(f"[setup] loading fail@128 split from {cfg.fail128_dir}")
    train_set, test_set = load_fail128_split(cfg.fail128_dir)
    _log(f"[setup] |train|={len(train_set)}  |test|={len(test_set)}")
    assert len(train_set) >= cfg.n_reward_questions, "not enough train problems for Q_R"

    teacher_opt = torch.optim.AdamW(_teacher_params(model), lr=cfg.teacher_lr)

    if dist_utils.is_main():
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
        _log(f"\n========== OUTER STEP {outer_step}/{cfg.n_outer_steps} ==========")

        # ----- 1. TEACHER GENERATES g*n parsed problems -----
        t_phase = time.time()
        _log(f"[step {outer_step}] (1) teacher-gen: need {needed_teacher} parsed problems "
             f"(g={g}, n={n_per}, budget={cfg.teacher_rejection_cap * needed_teacher})")
        teacher_user = build_teacher_prompt()
        teacher_prompt_str = chat_prompt(tok, teacher_user)
        teacher_lora_path = sync_adapter_to_disk(model, "teacher", cfg.out_dir, f"outer{outer_step}")
        if dist_utils.is_main():
            accepted_raw, accepted_problems, total_generated = generate_teacher_batch(
                engine, teacher_prompt_str, teacher_lora_path, cfg, needed=needed_teacher,
            )
            payload = (accepted_raw, accepted_problems, total_generated)
        else:
            payload = None
        accepted_raw, accepted_problems, total_generated = dist_utils.broadcast_object(payload, 0)
        teacher_gen_sec = time.time() - t_phase
        teacher_yield = len(accepted_problems) / max(total_generated, 1)
        if len(accepted_problems) < needed_teacher:
            _log(f"[step {outer_step}] teacher parse yield too low "
                 f"({len(accepted_problems)}/{needed_teacher}, generated={total_generated}, "
                 f"yield={teacher_yield:.2%}) — skipping outer step")
            if dist_utils.is_main():
                wandb.log({
                    "outer_step": outer_step, "skipped": 1,
                    "teacher/yield": teacher_yield,
                    "teacher/total_generated": total_generated,
                    "timing/teacher_gen_sec": teacher_gen_sec,
                }, step=outer_step)
            cleanup_old_adapters(cfg.out_dir, keep_last=8)
            continue
        X = [accepted_problems[k * n_per:(k + 1) * n_per] for k in range(g)]
        _log(f"[step {outer_step}] teacher produced {needed_teacher} parsed problems "
             f"(generated={total_generated}, yield={teacher_yield:.2%}, {teacher_gen_sec:.1f}s)")
        _log(f"  sample problem: {accepted_problems[0].problem[:160]!r}")
        _log(f"  sample answer:  {accepted_problems[0].answer!r}")

        # ----- 2. BASELINE EVAL on Q_R -----
        t_phase = time.time()
        # Only rank 0 draws from rng (its state would diverge otherwise); broadcast.
        if dist_utils.is_main():
            Q_R = rng.sample(train_set, cfg.n_reward_questions)
        else:
            Q_R = None
        Q_R = dist_utils.broadcast_object(Q_R, 0)
        _log(f"[step {outer_step}] (2) baseline-eval on |Q_R|={len(Q_R)}")
        # Push baseline weights to disk and eval
        restore_student_lora(model, baseline_snapshot)
        baseline_lora_path = sync_adapter_to_disk(
            model, "student", cfg.out_dir, f"outer{outer_step}_baseline",
        )
        if dist_utils.is_main():
            base_acc = eval_student_on(engine, baseline_lora_path, Q_R, tok, cfg)
        else:
            base_acc = None
        base_acc = dist_utils.broadcast_object(base_acc, 0)
        baseline_eval_sec = time.time() - t_phase
        _log(f"[step {outer_step}] baseline acc on Q_R = {base_acc:.3f} ({baseline_eval_sec:.1f}s)")

        # ----- 3. INNER TRAINING: g datasets × r parallel runs -----
        t_phase = time.time()
        total_inner = g * r_parallel
        _log(f"[step {outer_step}] (3) inner-training: {g}×{r_parallel}={total_inner} runs")
        R_matrix: List[List[float]] = [[0.0] * r_parallel for _ in range(g)]
        # Keep snapshots of each (k, j) trained student in case we promote.
        trained_snapshots: List[List[Optional[dict]]] = [[None] * r_parallel for _ in range(g)]
        run_idx = 0
        for k in range(g):
            for j in range(r_parallel):
                run_idx += 1
                t_run = time.time()
                restore_student_lora(model, baseline_snapshot)
                # Fresh Adam state per inner run: SOAR Alg. 1 treats each (k, j) as an
                # independent fine-tune starting from the baseline. Carrying m/v moments
                # or the step counter across runs would make R_{k,j} non-exchangeable.
                student_opt = torch.optim.AdamW(_student_params(model), lr=cfg.student_lr)
                tag = f"o{outer_step}_k{k}_j{j}"
                inner_train_student_on_dataset(
                    model, tok, engine, cfg, student_opt, X[k], tag=tag, rng=rng,
                )
                trained_path = sync_adapter_to_disk(model, "student", cfg.out_dir, f"{tag}_trained")
                if dist_utils.is_main():
                    acc = eval_student_on(engine, trained_path, Q_R, tok, cfg)
                else:
                    acc = None
                acc = dist_utils.broadcast_object(acc, 0)
                R_matrix[k][j] = acc - base_acc
                trained_snapshots[k][j] = snapshot_student_lora(model)
                run_sec = time.time() - t_run
                _log(f"[step {outer_step}] ({run_idx}/{total_inner}) X_{k} run{j}: "
                     f"acc={acc:.3f} Δ={acc - base_acc:+.3f} ({run_sec:.1f}s)")

        inner_train_sec = time.time() - t_phase
        R_k = [float(mean(row)) for row in R_matrix]
        R_bar_t = float(mean(R_k))
        R_flat = [r for row in R_matrix for r in row]
        R_std = float(torch.tensor(R_flat).std().item()) if len(R_flat) > 1 else 0.0
        _log(f"[step {outer_step}] R_k = {['%.3f' % r for r in R_k]}  "
             f"R̄_t = {R_bar_t:+.3f}  R_std = {R_std:.3f}  ({inner_train_sec:.1f}s)")

        # ----- 4. PROMOTION CHECK -----
        promoted_this_step = 0
        promotions_window.append(R_bar_t)
        ma = float(mean(promotions_window)) if len(promotions_window) == cfg.promotion_window else -float("inf")
        _log(f"[step {outer_step}] (4) promotion-check: "
             f"window={len(promotions_window)}/{cfg.promotion_window} "
             f"MA={ma if ma != -float('inf') else 'n/a'} τ={cfg.promotion_threshold} "
             f"total_promotions={n_promotions}/{cfg.max_promotions_to_track}")
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
            if dist_utils.is_main():
                append_promotion_questions(cfg.out_dir, n_promotions, X[k_star])
            # Persist the promoted student LoRA to disk (needed for PS baseline eval).
            # The live student weights are the LAST (k=g-1, j=r-1) trained student from the
            # inner-training loop above — not necessarily the promoted (k*, j*). Restore first.
            promo_dir = Path(cfg.out_dir) / "promotions" / f"promo_{n_promotions:03d}"
            prev_active = model.active_adapter
            restore_student_lora(model, baseline_snapshot)
            model.set_adapter("student")
            with dist_utils.summon_full_params(model, writeback=False, rank0_only=True):
                if dist_utils.is_main():
                    promo_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(promo_dir), selected_adapters=["student"])
            model.set_adapter(prev_active)
            dist_utils.barrier()
            if dist_utils.is_main():
                with open(Path(cfg.out_dir) / "promotions" / "index.jsonl", "a") as f:
                    f.write(json.dumps({
                        "promotion_idx": n_promotions, "outer_step": outer_step,
                        "k_star": k_star, "j_star": j_star,
                        "R_bar_t": R_bar_t, "moving_avg": ma,
                    }) + "\n")
            promotions_window.clear()
            promoted_this_step = 1
            _log(f"[step {outer_step}] PROMOTED (k*={k_star}, j*={j_star}, MA={ma:+.3f})  "
                 f"total promotions = {n_promotions}")

        # ----- 5. TEACHER RLOO UPDATE -----
        # Dataset-level LOO advantages over g rewards, broadcast to each X_k's n rollouts.
        t_phase = time.time()
        R_k_tensor = torch.tensor(R_k, dtype=torch.float32, device=dev)
        R_spread = (R_k_tensor.max() - R_k_tensor.min()).item() if g >= 1 else 0.0
        if g >= 2 and R_spread > 1e-6:
            _log(f"[step {outer_step}] (5) teacher-RLOO: R_spread={R_spread:.4f}")
            dataset_adv = rloo_advantages(R_k_tensor, group_size=g)        # [g]
            per_rollout_adv = dataset_adv.repeat_interleave(n_per)         # [g*n]
            teacher_batch_prompts = [teacher_prompt_str] * needed_teacher
            batch = tokenize_rollouts(
                tok, teacher_batch_prompts, accepted_raw, device=dev,
                max_prompt_len=cfg.max_prompt_len,
                max_total_len=cfg.max_prompt_len + cfg.teacher_max_new_tokens,
            )
            # Shard the batch across ranks (see student.py for the same pattern).
            total = batch.input_ids.shape[0]
            local_idx = dist_utils.shard_indices(total)
            local_idx_t = torch.tensor(local_idx, dtype=torch.long, device=dev)
            local_batch = type(batch)(
                input_ids=batch.input_ids.index_select(0, local_idx_t),
                attention_mask=batch.attention_mask.index_select(0, local_idx_t),
                completion_mask=batch.completion_mask.index_select(0, local_idx_t),
                rewards=batch.rewards,
            )
            local_adv = per_rollout_adv.index_select(0, local_idx_t)
            model.set_adapter("teacher")
            model.train()
            out = rloo_loss(model, local_batch, advantages=local_adv, kl_beta=cfg.kl_beta)
            out["loss"].backward()
            if dist_utils.is_distributed():
                grad_norm = model.clip_grad_norm_(1.0)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(_teacher_params(model), 1.0)
            teacher_opt.step()
            teacher_opt.zero_grad(set_to_none=True)
            teacher_metrics = {f"teacher/{k}": (v.item() if isinstance(v, torch.Tensor) else v)
                               for k, v in out.items()}
            teacher_metrics["teacher/grad_norm"] = float(grad_norm)
            teacher_metrics["teacher/adv_min"] = float(dataset_adv.min().item())
            teacher_metrics["teacher/adv_max"] = float(dataset_adv.max().item())
            loss_val = teacher_metrics.get("teacher/loss", float("nan"))
            _log(f"  loss={loss_val:.4f} grad_norm={float(grad_norm):.3f} "
                 f"adv_range=[{float(dataset_adv.min()):.3f},{float(dataset_adv.max()):.3f}]")
        else:
            _log(f"[step {outer_step}] (5) teacher-RLOO: SKIPPED "
                 f"(g={g}, R_spread={R_spread:.2e} ≤ 1e-6)")
            teacher_metrics = {"teacher/skipped": 1.0}
        teacher_update_sec = time.time() - t_phase

        # ----- 6. LOGGING + CHECKPOINT -----
        dt = time.time() - t_start
        gpu_mem_gb = (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0
        log = {
            "outer_step": outer_step,
            "baseline_acc": base_acc,
            "teacher/dataset_reward_mean": R_bar_t,
            "teacher/dataset_reward_max": float(max(R_k)),
            "teacher/dataset_reward_min": float(min(R_k)),
            "teacher/dataset_reward_std": R_std,
            "teacher/dataset_reward_spread": R_spread,
            "teacher/yield": teacher_yield,
            "teacher/total_generated": total_generated,
            "promotion/count_total": n_promotions,
            "promotion/triggered_this_step": promoted_this_step,
            "promotion/moving_avg": ma if ma != -float("inf") else 0.0,
            "promotion/window_fill": len(promotions_window) / cfg.promotion_window,
            "timing/teacher_gen_sec": teacher_gen_sec,
            "timing/baseline_eval_sec": baseline_eval_sec,
            "timing/inner_train_sec": inner_train_sec,
            "timing/teacher_update_sec": teacher_update_sec,
            "timing/step_sec": dt,
            "sys/gpu_max_mem_gb": gpu_mem_gb,
            **{f"R_k/dataset_{k}": R_k[k] for k in range(g)},
            **teacher_metrics,
        }
        if dist_utils.is_main():
            wandb.log(log, step=outer_step)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _log(f"[step {outer_step}] (6) summary: R̄_t={R_bar_t:+.3f} promo={promoted_this_step} "
             f"dt={dt:.1f}s  [gen={teacher_gen_sec:.0f}s eval={baseline_eval_sec:.0f}s "
             f"inner={inner_train_sec:.0f}s update={teacher_update_sec:.0f}s]  "
             f"peak_mem={gpu_mem_gb:.1f}GB")

        if outer_step % cfg.ckpt_every == 0:
            ckpt_dir = Path(cfg.out_dir) / f"ckpt_outer{outer_step}"
            _log(f"[step {outer_step}] checkpointing to {ckpt_dir}")
            for name in ("teacher", "student"):
                model.set_adapter(name)
                with dist_utils.summon_full_params(model, writeback=False, rank0_only=True):
                    if dist_utils.is_main():
                        model.save_pretrained(str(ckpt_dir), selected_adapters=[name])
            # Also persist the baseline student snapshot (not the live student weights).
            restore_student_lora(model, baseline_snapshot)
            model.set_adapter("student")
            with dist_utils.summon_full_params(model, writeback=False, rank0_only=True):
                if dist_utils.is_main():
                    model.save_pretrained(str(ckpt_dir / "baseline_student"),
                                          selected_adapters=["student"])
            dist_utils.barrier()
        cleanup_old_adapters(cfg.out_dir, keep_last=8)

    if dist_utils.is_main():
        wandb.finish()
    _log(f"[done] total promotions = {n_promotions}")
    dist_utils.shutdown()


# ------------- CLI -------------

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end sanity run")

    # model
    ap.add_argument("--base-model", type=str, default=None)
    ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--lora-alpha", type=int, default=None)

    # data
    ap.add_argument("--fail128-dir", type=str, default=None)
    ap.add_argument("--n-reward-questions", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)

    # outer (teacher)
    ap.add_argument("--n-outer-steps", type=int, default=None)
    ap.add_argument("--outer-group-size", type=int, default=None)
    ap.add_argument("--inner-dataset-size", type=int, default=None)
    ap.add_argument("--r-parallel-students", type=int, default=None)
    ap.add_argument("--teacher-lr", type=float, default=None)
    ap.add_argument("--teacher-max-new-tokens", type=int, default=None)
    ap.add_argument("--teacher-temperature", type=float, default=None)
    ap.add_argument("--teacher-rejection-cap", type=int, default=None)

    # inner (student)
    ap.add_argument("--inner-steps", type=int, default=None)
    ap.add_argument("--student-rollouts", type=int, default=None)
    ap.add_argument("--student-lr", type=float, default=None)
    ap.add_argument("--student-max-new-tokens", type=int, default=None)
    ap.add_argument("--student-temperature", type=float, default=None)

    # rl
    ap.add_argument("--kl-beta", type=float, default=None)
    ap.add_argument("--max-prompt-len", type=int, default=None)

    # promotion
    ap.add_argument("--promotion-threshold", type=float, default=None)
    ap.add_argument("--promotion-window", type=int, default=None)
    ap.add_argument("--max-promotions-to-track", type=int, default=None)

    # vllm
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--vllm-max-model-len", type=int, default=None)
    ap.add_argument("--vllm-max-loras", type=int, default=None)

    # logging
    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--wandb-mode", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = smoke_config() if args.smoke else SoarConfig()
    for name, value in vars(args).items():
        if name == "smoke" or value is None:
            continue
        setattr(cfg, name, value)
    main(cfg)
