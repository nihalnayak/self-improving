"""Decoupled SGS, step 1: train a Conjecturer that writes calibrated hints, on sorting.

The Conjecturer sees a hard sorting task, its gold answer and a target per-attempt
success rate t in {1/5, 2/5, 3/5, 4/5}, and writes a natural-language hint. The
frozen base model (the Solver, same init; it plays SGS's frozen Guide) attempts
the task `--n-solve` times with that hint, and

    R = 1 - |t - actual_rate|        (0 if the hint is missing / unparsable)

trains the Conjecturer with GRPO (PPO clip, token-mean, low_var_kl to the frozen
Solver, which shares the init weights).

Pipeline (one Ray runner, 2 models on the shared GPUs):
    1. hard tasks   pass@5 of the frozen Solver on --dataset (no hint); keep 0/5
    2. split        seeded shuffle, 50/50 -> train (Conjecturer) / heldout
                    (validation, and the later Solver-training set)
    3. targets      within each split, order easiest -> hardest and cut into 4
                    equal buckets: easiest -> 4/5, ..., hardest -> 1/5
    4. GRPO         --iters on-policy steps: --batch-size train tasks x
                    --group-size hints, each hint scored with --n-solve attempts
    5. validation   every --eval-every steps (and step 0): 1 hint per heldout task
                    at its bucket target; mean |t - actual| overall and per target
    6. final data   for every heldout task and each of the 4 targets, sample
                    --final-candidates hints and keep the one whose actual rate is
                    closest to t (first sampled wins ties) -> final_dataset.jsonl

DESIGN NOTE (difficulty ordering): the pass@5 filter leaves every hard task at
0/5, so solve rate cannot order them. Difficulty is a proxy: list length, with
ties broken by the seeded shuffle. Revisit if a better signal is available
(e.g. pass@k at larger k, or hinted solve rate).

Outputs in --out-dir:
    hard_eval.jsonl             per-task pass@5 of the base Solver (no hint)
    splits/{train,heldout}.jsonl  {list, sorted, length, target}
    metrics.jsonl, eval.jsonl, samples/iter_XXXX.json, eval_samples/iter_XXXX.jsonl
    final_dataset.jsonl         {list, sorted, length, target, hint, actual_rate, abs_err}
    ckpt/iter_XXXX/conj         Conjecturer checkpoint (verl FSDP + hf_model)

The Solver prompt for a hinted task is SOLVER_PROMPT + "\\n\\nHint: {hint}"
(misc/sort_hint_calibrate.solver_text).

Usage (2 GPUs; see files/scripts/decoupled_sgs/train_conjecture_sorting_qwen3_4b.sbatch):
    python -m decoupled_sgs.train_conjecture --out-dir runs/conj --n-gpus 2
"""

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import ray

from sgs.common import SOLVER_PROMPT, load_targets, parse_sorted, ray_runtime_env, user_msg
from sgs.sgs_grpo import grpo_advantages
from sgs.sgs_simple import Policy, build_verl_config, phase, response_stats

TARGETS = (0.8, 0.6, 0.4, 0.2)   # bucket 0 (easiest) -> 4/5, ..., bucket 3 (hardest) -> 1/5


# =============================================================================
# §1 Args
# =============================================================================

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Decoupled SGS: GRPO-trained hint Conjecturer")
    ap.add_argument("--dataset", default="data/sorting/sorting_1000.jsonl",
                    help="jsonl with a `list` field per row; hard tasks are filtered from it")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B", help="init for Conjecturer and Solver")
    ap.add_argument("--n-gpus", type=int, default=2)
    ap.add_argument("--iters", type=int, default=100, help="GRPO optimizer steps")
    ap.add_argument("--batch-size", type=int, default=16, help="train tasks per step")
    ap.add_argument("--group-size", type=int, default=8, help="hints per task (GRPO group)")
    ap.add_argument("--n-solve", type=int, default=5,
                    help="Solver attempts per hint (also the hard-task pass@n)")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--micro-batch-size-per-gpu", type=int, default=4)
    ap.add_argument("--kl-coef", type=float, default=0.001,
                    help="low_var_kl coefficient vs the frozen Solver; 0 = no KL")
    ap.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip epsilon")
    ap.add_argument("--thinking", action="store_true",
                    help="enable_thinking=True in the chat template (both models)")
    ap.add_argument("--prompt-length", type=int, default=6144,
                    help="must fit the task + a full-length hint in the Solver prompt")
    ap.add_argument("--response-length", type=int, default=4096,
                    help="max response tokens for both models (one value keeps the "
                         "frozen Solver's KL ref log-probs on the Conjecturer's shapes)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    ap.add_argument("--no-offload", action="store_true")
    ap.add_argument("--model-dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--eval-every", type=int, default=10, help="0 = off")
    ap.add_argument("--save-every", type=int, default=0, help="0 = only at the end")
    ap.add_argument("--final-candidates", type=int, default=4,
                    help="hints sampled per (heldout task, target) for final_dataset.jsonl")
    ap.add_argument("--final-limit", type=int, default=0,
                    help="if >0, build final_dataset.jsonl from only the first N heldout tasks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="sgs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    return ap.parse_args(argv)


def build_configs(args):
    """{tag: verl config}. The Conjecturer gets verl's GRPO loss; one optimizer step
    per training step (mini-batch = batch_size x group_size sequences)."""
    from omegaconf import OmegaConf

    seq_args = argparse.Namespace(**{**vars(args),
                                     "mini_batch_size": args.batch_size * args.group_size})
    cfgs = {tag: build_verl_config(seq_args, tag, args.response_length)
            for tag in ("conj", "solver")}
    a = "actor_rollout_ref.actor"
    for key, val in {"policy_loss.loss_mode": "vanilla",
                     "loss_agg_mode": "token-mean",
                     "clip_ratio": args.clip_ratio,
                     "clip_ratio_low": args.clip_ratio,
                     "clip_ratio_high": args.clip_ratio,
                     "use_kl_loss": args.kl_coef > 0,
                     "kl_loss_coef": args.kl_coef,
                     "kl_loss_type": "low_var_kl"}.items():
        OmegaConf.update(cfgs["conj"], f"{a}.{key}", val)
    return cfgs


# =============================================================================
# §2 Prompt + reward
# =============================================================================

# Adapted from misc/sort_hint_calibrate.HINT_WRITER_TEMPLATE (same wording and
# clue list) without the history / feedback part. Every target is < n/n, so the
# "do not write out the full sorted list" rule always applies.
CONJ_PROMPT = (
    "A language model is asked to sort this list of {L} integers in "
    "ascending order:\n\n"
    "List: {xs}\n\n"
    "The correct sorted answer is:\n\n"
    "Sorted: {gold}\n\n"
    "The model sees the list plus a natural-language hint. We sample it "
    "{n} times. We want it to get the exact sorted list right in exactly "
    "{t} of {n} attempts: not more, not less.\n\n"
    "Write a hint. The more the hint reveals about the answer, the more often "
    "the model will solve it. Useful clues include: the "
    "smallest and largest values, how many numbers fall in each range "
    "(e.g. below 1000, 1000-1999, ...), some of the sorted values at the "
    "start, middle or end, which values are duplicated, or a good "
    "bucketing strategy. Do not write out the full sorted list. Write the "
    "hint as plain natural-language sentences addressed to the model.\n\n"
    "Return only the hint inside <hint>...</hint>."
)

_HINT_RE = re.compile(r"<hint>(.*?)</hint>", re.DOTALL)


def conj_messages(xs, target, n):
    return user_msg(CONJ_PROMPT.format(L=len(xs), xs=xs, gold=sorted(xs), n=n,
                                       t=round(target * n)))


def solver_hint_messages(xs, hint):
    return user_msg(SOLVER_PROMPT.format(list_str=str(xs)) + f"\n\nHint: {hint}")


def parse_hint(text):
    """Last non-empty <hint>...</hint> after any </think>, or None."""
    found = [h.strip() for h in _HINT_RE.findall(text.split("</think>")[-1]) if h.strip()]
    return found[-1] if found else None


def hint_rewards(targets, rates):
    """R = 1 - |t - rate|; 0 where the hint did not parse (rate is NaN)."""
    r = 1.0 - np.abs(np.asarray(targets) - np.asarray(rates))
    return np.nan_to_num(r, nan=0.0)


# =============================================================================
# §3 Data: hard filter, split, difficulty buckets
# =============================================================================

def assign_targets(tasks):
    """tasks (seed-shuffled) -> rows {list, sorted, length, target}, easiest -> hardest.
    Difficulty = list length (see DESIGN NOTE); the stable sort keeps the shuffle
    order within a length. 4 equal buckets: easiest -> 4/5, hardest -> 1/5."""
    order = sorted(tasks, key=len)
    n = len(order)
    return [{"list": xs, "sorted": sorted(xs), "length": len(xs),
             "target": TARGETS[min(i * len(TARGETS) // n, len(TARGETS) - 1)]}
            for i, xs in enumerate(order)]


def find_hard(args, solver, out_dir):
    """Base-Solver pass@n (no hint) on --dataset; returns the 0/n tasks."""
    tasks = load_targets(args.dataset)
    n = args.n_solve
    timing = {}
    with phase(0, "hard_eval", timing, f"{len(tasks)} tasks x {n} attempts"):
        _, texts = solver.generate([user_msg(SOLVER_PROMPT.format(list_str=str(xs)))
                                    for xs in tasks], n=n, step=0)
    correct = np.array([parse_sorted(texts[j]) == sorted(tasks[j // n])
                        for j in range(len(texts))]).reshape(len(tasks), n).sum(1)
    with open(out_dir / "hard_eval.jsonl", "w") as f:
        for xs, c in zip(tasks, correct):
            f.write(json.dumps({"list": xs, "length": len(xs), "n_correct": int(c)}) + "\n")
    hard = [xs for xs, c in zip(tasks, correct) if c == 0]
    lens = sorted({len(xs) for xs in tasks})
    print("[hard] " + " ".join(
        f"len_{L}={sum(len(x) == L for x in hard)}/{sum(len(x) == L for x in tasks)}"
        for L in lens) + f" total={len(hard)}/{len(tasks)}", flush=True)
    return hard


# =============================================================================
# §4 Hints -> actual solve rates
# =============================================================================

def write_hints(t, args, conj, rows, targets, g, timing, name="conj_gen"):
    """g hints per (row, target). Returns (rollouts, texts, hints), row-major, g per prompt."""
    with phase(t, name, timing, f"{len(rows)} prompts x {g} hints"):
        rollouts, texts = conj.generate(
            [conj_messages(r["list"], tg, args.n_solve) for r, tg in zip(rows, targets)], n=g, step=t)
    return rollouts, texts, [parse_hint(x) for x in texts]


def solve_rates(t, args, solver, tasks, hints, timing, name="solver_gen"):
    """Per-hint Solver success rate over --n-solve attempts; NaN for missing hints."""
    rates = np.full(len(hints), np.nan)
    idx = [i for i, h in enumerate(hints) if h is not None]
    if not idx:
        return rates
    n = args.n_solve
    with phase(t, name, timing, f"{len(idx)} hints x {n} attempts"):
        _, texts = solver.generate([solver_hint_messages(tasks[i], hints[i]) for i in idx],
                                   n=n, step=t)
    v = np.array([parse_sorted(texts[j]) == sorted(tasks[idx[j // n]])
                  for j in range(len(texts))]).reshape(len(idx), n)
    rates[idx] = v.mean(axis=1)
    return rates


def calibration_metrics(prefix, targets, rates):
    """Parse rate, reward and |t - rate| overall and per target. Unparsed hints
    count as rate 0 in abs_err / actual_rate (their reward is 0)."""
    targets, rates = np.asarray(targets), np.asarray(rates)
    ok = ~np.isnan(rates)
    err = np.abs(targets - np.nan_to_num(rates, nan=0.0))
    m = {f"{prefix}/parse_rate": float(ok.mean()),
         f"{prefix}/reward": float(hint_rewards(targets, rates).mean()),
         f"{prefix}/abs_err": float(err.mean())}
    for tg in TARGETS:
        s = targets == tg
        if s.any():
            m[f"{prefix}/abs_err/t={tg}"] = float(err[s].mean())
            m[f"{prefix}/actual_rate/t={tg}"] = float(np.nan_to_num(rates[s], nan=0.0).mean())
    return m


# =============================================================================
# §5 GRPO step, validation, final dataset
# =============================================================================

def train_step(t, args, rows, conj, solver):
    """One on-policy GRPO step of the Conjecturer on `rows`. Returns (metrics, samples)."""
    m, timing = {}, {}
    g = args.group_size
    rollouts, texts, hints = write_hints(t, args, conj, rows, [r["target"] for r in rows], g, timing)
    tasks = [rows[j // g]["list"] for j in range(len(hints))]
    targets = np.array([rows[j // g]["target"] for j in range(len(hints))])
    rates = solve_rates(t, args, solver, tasks, hints, timing)
    rewards = hint_rewards(targets, rates)
    m.update(calibration_metrics("train", targets, rates))

    adv = grpo_advantages(rewards, g)
    m["conj/frac_zero_adv_groups"] = float((np.abs(adv.reshape(-1, g)).sum(1) == 0).mean())
    m["conj/n_opt_steps"] = 0
    if np.any(adv):
        ref = None
        if args.kl_coef > 0:
            with phase(t, "conj_ref", timing, f"{len(adv)} rollouts (frozen Solver = KL ref)"):
                ref = solver.log_probs(rollouts)
        steps = math.ceil(len(adv) / conj.mini_batch_size)
        with phase(t, "conj_update", timing, f"{len(adv)} rows -> {steps} optimizer steps"):
            m.update({f"conj/{k}": v for k, v in
                      conj.update(rollouts, np.arange(len(adv)), adv, ref_log_probs=ref).items()})
        m["conj/n_opt_steps"] = steps
    m["conj/response_len_mean"], m["conj/response_clip_frac"] = response_stats(rollouts)
    m.update({f"timing_s/{k}": v for k, v in timing.items()})
    samples = [{"list": tasks[j], "target": float(targets[j]), "completion": texts[j],
                "hint": hints[j], "actual_rate": None if np.isnan(rates[j]) else float(rates[j]),
                "reward": float(rewards[j])}
               for j in range(0, len(texts), max(1, len(texts) // 16))]
    return m, samples


def evaluate(t, args, rows, conj, solver):
    """1 hint per heldout row at its bucket target. Returns (metrics, rows)."""
    timing = {}
    _, texts, hints = write_hints(t, args, conj, rows, [r["target"] for r in rows], 1, timing,
                                  name="val_conj_gen")
    rates = solve_rates(t, args, solver, [r["list"] for r in rows], hints, timing,
                        name="val_solver_gen")
    m = calibration_metrics("val", [r["target"] for r in rows], rates)
    out = [{**r, "completion": x, "hint": h,
            "actual_rate": None if np.isnan(a) else float(a)}
           for r, x, h, a in zip(rows, texts, hints, rates)]
    return m, out


def build_final_dataset(t, args, rows, conj, solver, path):
    """Every heldout row x every target: keep the closest of --final-candidates hints."""
    timing = {}
    c = args.final_candidates
    pairs = [(r, tg) for r in rows for tg in TARGETS]
    _, _, hints = write_hints(t, args, conj, [r for r, _ in pairs], [tg for _, tg in pairs], c,
                              timing, name="final_conj_gen")
    rates = solve_rates(t, args, solver, [pairs[j // c][0]["list"] for j in range(len(hints))],
                        hints, timing, name="final_solver_gen")
    targets = np.array([pairs[j // c][1] for j in range(len(hints))])
    err = np.where(np.isnan(rates), np.inf, np.abs(targets - np.nan_to_num(rates)))
    errs = []
    with open(path, "w") as f:
        for p, (r, tg) in enumerate(pairs):
            j = p * c + int(np.argmin(err[p * c:(p + 1) * c]))   # first sampled wins ties
            if hints[j] is None:
                continue   # no candidate parsed
            errs.append(float(err[j]))
            f.write(json.dumps({"list": r["list"], "sorted": r["sorted"], "length": r["length"],
                                "target": tg, "hint": hints[j], "actual_rate": float(rates[j]),
                                "abs_err": float(err[j])}) + "\n")
    print(f"[final] {len(errs)}/{len(pairs)} (task, target) rows -> {path}; "
          f"mean abs_err={np.mean(errs) if errs else float('nan'):.3f} "
          f"exact={np.mean(np.array(errs) == 0) if errs else float('nan'):.3f}", flush=True)


# =============================================================================
# §6 Runner + main
# =============================================================================

def write_jsonl(path, rows):
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


@ray.remote(num_cpus=1)
class ConjRunner:
    """Owns the Conjecturer (trained) and Solver (frozen) and the training loop."""

    def run(self, args):
        from tqdm import tqdm
        from transformers import AutoTokenizer
        from verl.single_controller.ray import RayResourcePool
        from verl.utils.profiler.performance import simple_timer

        rng = random.Random(args.seed)
        out_dir = Path(args.out_dir)
        for sub in ("samples", "eval_samples", "splits"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(
            {**vars(args), "difficulty_proxy": "list length (pass@5 is 0 for all hard tasks)"},
            indent=2))

        tok = AutoTokenizer.from_pretrained(args.model)
        pool = RayResourcePool(process_on_nodes=[args.n_gpus], use_gpu=True,
                               max_colocate_count=2, name_prefix="dsgs")
        policies = {}
        for tag, cfg in build_configs(args).items():
            print(f"[init] {tag}", flush=True)
            policies[tag] = Policy(tag, cfg, pool, tok)
        conj, solver = policies["conj"], policies["solver"]

        # --- data: hard filter -> 50/50 split -> difficulty buckets ---------------
        hard = find_hard(args, solver, out_dir)
        rng.shuffle(hard)
        half = len(hard) // 2
        train_rows, heldout_rows = assign_targets(hard[:half]), assign_targets(hard[half:])
        write_jsonl(out_dir / "splits" / "train.jsonl", train_rows)
        write_jsonl(out_dir / "splits" / "heldout.jsonl", heldout_rows)
        for name, rows in (("train", train_rows), ("heldout", heldout_rows)):
            print(f"[split] {name}: {len(rows)} tasks; " + " ".join(
                f"t={tg}:{sum(r['target'] == tg for r in rows)}"
                f"(len {min((r['length'] for r in rows if r['target'] == tg), default='-')}-"
                f"{max((r['length'] for r in rows if r['target'] == tg), default='-')})"
                for tg in TARGETS), flush=True)
        B = args.batch_size
        steps_per_epoch = len(train_rows) // B
        if steps_per_epoch == 0:
            raise SystemExit(f"{len(train_rows)} train tasks < --batch-size {B}")

        wandb = None
        if not args.no_wandb:
            import wandb
            wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args),
                       dir=str(out_dir))
            print(f"[wandb] {wandb.run.url}", flush=True)

        def run_eval(t):
            ev, rows = evaluate(t, args, heldout_rows, conj, solver)
            write_jsonl(out_dir / "eval_samples" / f"iter_{t:04d}.jsonl", rows)
            print(f"[eval {t}] " + " ".join(f"{k}={v:.4g}" for k, v in ev.items()), flush=True)
            with open(out_dir / "eval.jsonl", "a") as f:
                f.write(json.dumps({"iter": t, **ev}) + "\n")
            return ev

        if args.eval_every:
            ev = run_eval(0)
            if wandb is not None:
                wandb.log(ev, step=0, commit=True)

        perm = list(range(len(train_rows)))
        pbar = tqdm(total=args.iters, desc="Training Progress")
        for t in range(1, args.iters + 1):
            t0 = time.time()
            s = (t - 1) % steps_per_epoch
            if s == 0:
                rng.shuffle(perm)
            batch = [train_rows[i] for i in perm[s * B:(s + 1) * B]]
            metrics, samples = train_step(t, args, batch, conj, solver)
            metrics["epoch"] = (t - 1) // steps_per_epoch
            timing = {}
            if args.eval_every and t % args.eval_every == 0:
                with simple_timer("eval", timing):
                    metrics.update(run_eval(t))
            if (args.save_every and t % args.save_every == 0) or t == args.iters:
                with simple_timer("save", timing):
                    conj.save(out_dir / "ckpt" / f"iter_{t:04d}" / "conj", t)
            metrics.update({f"timing_s/{k}": v for k, v in timing.items()})
            metrics["iter"], metrics["time/iter_sec"] = t, time.time() - t0
            print(f"[iter {t}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                            for k, v in sorted(metrics.items())), flush=True)
            with open(out_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(metrics) + "\n")
            with open(out_dir / "samples" / f"iter_{t:04d}.json", "w") as f:
                json.dump(samples, f, indent=1)
            if wandb is not None:
                wandb.log(metrics, step=t, commit=True)
            pbar.set_postfix(reward=f"{metrics['train/reward']:.3f}",
                             abs_err=f"{metrics['train/abs_err']:.3f}",
                             iter_sec=f"{metrics['time/iter_sec']:.0f}")
            pbar.update(1)
        pbar.close()

        final_rows = heldout_rows[:args.final_limit] if args.final_limit else heldout_rows
        build_final_dataset(args.iters + 1, args, final_rows, conj, solver,
                            out_dir / "final_dataset.jsonl")

        if wandb is not None:
            wandb.finish()


def main():
    args = parse_args()
    ray.init(runtime_env=ray_runtime_env())
    ray.get(ConjRunner.remote().run.remote(args))


if __name__ == "__main__":
    # Re-import under the package name so Ray pickles our classes by reference.
    from decoupled_sgs import train_conjecture
    train_conjecture.main()
