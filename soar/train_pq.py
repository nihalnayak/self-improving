"""PQ baseline (SOAR paper §4.3): Promotion Questions, mixed-corpus RLOO.

Train a fresh student via RLOO on {fail@128 train} ∪ 𝒟_best (problems from
promoted datasets accumulated during a SOAR run). Report pass@k on the fail@128
test split, averaged over the 200-step window that follows a reward-slope-based
early-stopping point (Appendix B.6).

Paper §4.3: *"We train a fresh base student with standard RLOO on a combination
of PQ (𝒟_best) and the fail@128 train set. This isolates the value of the
synthetic questions."*

Hyperparameters follow paper Table 3 (Student + Evaluation-specific) and B.6/B.7:
AdamW, cosine LR schedule with 20 warmup steps, LR=1e-5, KL=0.001, batch=8
problems/step, group=32 rollouts/problem, max_tokens=1024, T=1.0, max 1500 steps.
Checkpoint selection: 25-step CMA, 15% slope threshold, 200-step averaging.
Test metric: pass@k with 32 samples/problem at T=1.0, k ∈ {1,4,8,16,32}.

CLI: python -m soar.train_pq --soar-run runs/soar-v1 [--out runs/pq-v1] [--smoke]
"""

import argparse
import dataclasses
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import torch
import wandb
from transformers import get_cosine_schedule_with_warmup

from soar.config import SoarConfig, smoke_config
from soar.data import MathProblem, _read_jsonl, load_fail128_split
from soar.models import load_base_and_adapters, load_vllm_engine, student_chat_prompt
from soar.rollout import cleanup_old_adapters, sync_adapter_to_disk, vllm_generate
from soar.student import inner_train_student_on_dataset
from soar.verify import pass_at_k, success_math_answer


# ------------- helpers -------------

def _student_params(peft_model):
    return [p for n, p in peft_model.named_parameters() if ".student." in n]


def _load_promotion_questions(soar_run: Path) -> List[MathProblem]:
    """Read `{soar_run}/promotion_questions.jsonl`. The extra `promotion_idx` field
    added by `soar.train.append_promotion_questions` is ignored by `from_dict`."""
    path = soar_run / "promotion_questions.jsonl"
    if not path.exists():
        print(f"[pq] WARN: {path} not found — training on fail@128 train only")
        return []
    return _read_jsonl(path)


def _greedy_pass1(engine, questions: List[MathProblem], lora_path: Path, tok,
                  max_new_tokens: int) -> float:
    prompts = [student_chat_prompt(tok, q.problem) for q in questions]
    completions = vllm_generate(
        engine, prompts, lora_path=str(lora_path), lora_name="student",
        max_new_tokens=max_new_tokens, temperature=0.0,
    )
    scores = [1.0 if success_math_answer(c, q.answer) else 0.0
              for c, q in zip(completions, questions)]
    return float(mean(scores)) if scores else 0.0


def _sample_per_question(engine, questions: List[MathProblem], lora_path: Path, tok,
                         *, n_samples: int, temperature: float,
                         max_new_tokens: int) -> List[List[str]]:
    flat: List[str] = []
    for q in questions:
        flat.extend([student_chat_prompt(tok, q.problem)] * n_samples)
    comps = vllm_generate(
        engine, flat, lora_path=str(lora_path), lora_name="student",
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    return [comps[i * n_samples:(i + 1) * n_samples] for i in range(len(questions))]


def _cma(xs: List[float], window: int) -> List[float]:
    """Centered moving average with window `window`. Edge points use the available
    half-window (shrunk CMA). Returns a list the same length as `xs`."""
    if window <= 1 or len(xs) <= 1:
        return list(xs)
    half = window // 2
    out: List[float] = []
    n = len(xs)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def _select_early_stop(reward_history: List[float], cma_window: int,
                       slope_threshold: float) -> int:
    """SOAR Appendix B.6: smooth reward curve (CMA), compute discrete slopes
    normalized by the range of observed rewards, pick the earliest step whose
    normalized slope falls below `slope_threshold * max(normalized_slopes)`.

    Returns a step index into `reward_history`. If no slope is detected (e.g.
    too-short run), returns max(0, len(reward_history) - 1).
    """
    if len(reward_history) < max(2, cma_window):
        return max(0, len(reward_history) - 1)
    smooth = _cma(reward_history, cma_window)
    r_range = max(reward_history) - min(reward_history)
    if r_range <= 1e-9:
        return max(0, len(reward_history) - 1)
    slopes = [(smooth[i + 1] - smooth[i]) / r_range for i in range(len(smooth) - 1)]
    if not slopes:
        return max(0, len(reward_history) - 1)
    max_slope = max(slopes)
    if max_slope <= 0:
        return max(0, len(reward_history) - 1)
    threshold = slope_threshold * max_slope
    # Scan AFTER the slope peak so we don't trip on the warmup region where slope
    # hasn't ramped yet.
    peak_idx = max(range(len(slopes)), key=lambda i: slopes[i])
    for i in range(peak_idx, len(slopes)):
        if slopes[i] < threshold:
            return i
    return len(reward_history) - 1


# ------------- main -------------

def main(cfg: SoarConfig, soar_run: str, out_dir: str) -> None:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    cfg.out_dir = out_dir
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(cfg.out_dir) / "ckpts"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir) / "config.json", "w") as f:
        json.dump({k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in cfg.__dict__.items()}, f, indent=2)

    print(f"[pq] loading fail@128 from {cfg.fail128_dir}")
    train_set, test_set = load_fail128_split(cfg.fail128_dir)
    print(f"[pq] |train|={len(train_set)}  |test|={len(test_set)}")

    d_best = _load_promotion_questions(Path(soar_run))
    assert train_set or d_best, "[pq] no training data at all"
    combined = list(train_set) + list(d_best)
    rng.shuffle(combined)
    print(f"[pq] |D_best|={len(d_best)}  |combined|={len(combined)}")

    print(f"[pq] loading model: {cfg.base_model}")
    model, tok = load_base_and_adapters(cfg)
    model.set_adapter("student")
    print("[pq] loading vLLM engine")
    engine = load_vllm_engine(cfg)

    student_opt = torch.optim.AdamW(_student_params(model), lr=cfg.pq_lr)
    lr_sched = get_cosine_schedule_with_warmup(
        student_opt,
        num_warmup_steps=cfg.pq_lr_warmup_steps,
        num_training_steps=cfg.pq_max_train_steps,
    )

    # Shallow-copy cfg with PQ-specific overrides for the inner RLOO call.
    cfg_pq = dataclasses.replace(
        cfg,
        student_group_size=cfg.pq_group_size,
        student_max_new_tokens=cfg.pq_max_new_tokens,
        student_temperature=cfg.pq_temperature,
        kl_beta=cfg.pq_kl_beta,
    )

    wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode,
               config={**cfg.__dict__, "baseline": "pq", "soar_run": soar_run})

    reward_history: List[float] = []
    diag_history: List[Dict[str, float]] = []
    ckpt_steps: List[int] = []

    t_start = time.time()
    for step in range(cfg.pq_max_train_steps):
        tag = f"pq_s{step}"
        stats = inner_train_student_on_dataset(
            model, tok, engine, cfg_pq, student_opt, combined, tag=tag, rng=rng,
            batch_problems=cfg.pq_batch_problems, inner_steps=1,
        )
        lr_sched.step()

        r = float(stats.get("mean_reward", 0.0))
        reward_history.append(r)
        log = {"pq/step": step, "pq/mean_reward": r,
               "pq/lr": lr_sched.get_last_lr()[0]}
        for key in ("loss", "pg_loss", "kl", "adv_abs_mean"):
            if key in stats:
                log[f"pq/{key}"] = float(stats[key])
        if stats.get("skipped"):
            log["pq/skipped"] = 1.0
        wandb.log(log, step=step)

        # Persistent checkpoint for post-hoc selection.
        if step % cfg.pq_test_ckpt_every == 0 or step == cfg.pq_max_train_steps - 1:
            ck_dir = ckpt_root / f"step_{step:05d}"
            ck_dir.mkdir(parents=True, exist_ok=True)
            prev_active = model.active_adapter
            model.set_adapter("student")
            model.save_pretrained(str(ck_dir), selected_adapters=["student"])
            model.set_adapter(prev_active)
            ckpt_steps.append(step)

        # Periodic greedy diagnostic (not reported).
        if step % cfg.pq_diag_every == 0:
            qr = rng.sample(train_set, min(cfg.n_reward_questions, len(train_set)))
            diag_lora = sync_adapter_to_disk(model, "student", cfg.out_dir, f"diag_s{step}")
            acc = _greedy_pass1(engine, qr, Path(diag_lora), tok, cfg.student_max_new_tokens)
            diag_history.append({"step": step, "train_greedy_pass1": acc})
            wandb.log({"pq/train_greedy_pass1": acc}, step=step)
            print(f"[pq] step {step:5d}  mean_r={r:+.3f}  "
                  f"train_pass@1(greedy)={acc:.3f}  dt={time.time() - t_start:.1f}s")
        cleanup_old_adapters(cfg.out_dir, keep_last=8)

    print(f"[pq] training done in {time.time() - t_start:.1f}s")

    # ----- checkpoint selection + test pass@k (B.6) -----
    early_stop = _select_early_stop(
        reward_history, cma_window=cfg.pq_slope_cma_window,
        slope_threshold=cfg.pq_slope_threshold,
    )
    window_lo = early_stop
    window_hi = min(early_stop + cfg.pq_test_window, cfg.pq_max_train_steps)
    window_ckpts = [s for s in ckpt_steps if window_lo <= s < window_hi]
    if not window_ckpts:
        # Fall back to the last available checkpoint.
        window_ckpts = ckpt_steps[-1:]
    print(f"[pq] early_stop step = {early_stop}; eval window = [{window_lo}, {window_hi}); "
          f"{len(window_ckpts)} checkpoints to evaluate")

    golds = [q.answer for q in test_set]
    per_ckpt_pk: List[Dict[str, float]] = []
    for s in window_ckpts:
        lp = ckpt_root / f"step_{s:05d}"
        comps = _sample_per_question(
            engine, test_set, lp, tok,
            n_samples=cfg.pq_pass_at_k_samples,
            temperature=cfg.pq_temperature,
            max_new_tokens=cfg.pq_max_new_tokens,
        )
        pk = pass_at_k(comps, golds, ks=tuple(cfg.pq_pass_at_k_ks))
        per_ckpt_pk.append({"step": s, **{str(k): v for k, v in pk.items()}})
        print(f"[pq] ckpt step {s}: " + ", ".join(f"pass@{k}={pk[k]:.4f}"
                                                   for k in cfg.pq_pass_at_k_ks))

    # Average over the window per k.
    avg_pk: Dict[str, float] = {}
    for k in cfg.pq_pass_at_k_ks:
        avg_pk[str(k)] = float(mean(ck[str(k)] for ck in per_ckpt_pk))
    for k_str, v in avg_pk.items():
        wandb.log({f"pq/test_pass_at_{k_str}": v}, step=cfg.pq_max_train_steps)
    print("[pq] window-averaged test pass@k = "
          + ", ".join(f"{k}:{v:.4f}" for k, v in avg_pk.items()))

    report = {
        "soar_run": soar_run,
        "fail128_dir": cfg.fail128_dir,
        "n_training_steps": cfg.pq_max_train_steps,
        "n_combined_problems": len(combined),
        "n_d_best": len(d_best),
        "early_stop_step": early_stop,
        "eval_window": [window_lo, window_hi],
        "window_ckpt_steps": window_ckpts,
        "per_ckpt_pass_at_k": per_ckpt_pk,
        "test_pass_at_k": avg_pk,
        "train_reward_history": reward_history,
        "train_greedy_diag": diag_history,
        "eval": {
            "n_samples": cfg.pq_pass_at_k_samples,
            "temperature": cfg.pq_temperature,
            "max_new_tokens": cfg.pq_max_new_tokens,
            "ks": list(cfg.pq_pass_at_k_ks),
        },
    }
    with open(Path(cfg.out_dir) / "pq_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[pq] wrote {Path(cfg.out_dir) / 'pq_report.json'}")
    wandb.finish()


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soar-run", type=str, required=True,
                    help="path to a completed SOAR run (used to read promotion_questions.jsonl)")
    ap.add_argument("--out", type=str, default=None, help="override out_dir")
    ap.add_argument("--fail128-dir", type=str, default=None, help="override fail128_dir")
    ap.add_argument("--smoke", action="store_true", help="tiny config for a sanity run")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = smoke_config() if args.smoke else SoarConfig()
    if args.fail128_dir:
        cfg.fail128_dir = args.fail128_dir
    out_dir = args.out or (str(Path(args.soar_run) / "pq_eval"))
    main(cfg, args.soar_run, out_dir)
