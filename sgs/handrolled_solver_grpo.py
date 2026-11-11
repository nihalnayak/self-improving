"""Solver-only GRPO with the hand-rolled SGS trainer, on sorting, built on verl.

`sgs.sgs_grpo` with the Conjecturer, Guide and synthetic problems removed, run
exactly like verl's trainer in `sgs.solver_simple` (train batch = mini-batch):

    each iteration = one optimizer step, always on-policy (no stale steps):
        B <- next --mini-batch-size targets (epoch-wise shuffle, last partial batch dropped)
        k rollouts per target from the current weights
        GRPO advantages; θ_old log-probs recomputed by the FSDP actor (ratio = 1)
        one step of PPO clip, token-mean, + β·KL(π_θ || ρ) low_var_kl (`solver_grpo_update`)

--iters counts optimizer steps (verl global_steps): epochs x (|D| // mini_batch_size).

The KL reference ρ is a separate frozen copy of the init model (the `ref` Policy,
loaded only when --kl-coef > 0), standing in for verl's built-in ref worker.
Remaining differences from solver_simple: the sampler RNG (python `random`, not
torch's generator) and the ref living in its own Policy.

Args are sgs_grpo's; --batch-size must be 0 or --mini-batch-size; the Conjecturer /
Guide / --all-targets flags are ignored.

Usage (4 GPUs; see files/scripts/sgs/handrolled_solver_grpo_sorting_qwen3_4b.sbatch):
    python -m sgs.handrolled_solver_grpo --dataset files/data/sorting/sorting_1000.jsonl \
        --out-dir runs/handrolled_solver_grpo --n-gpus 4 --mini-batch-size 32 --iters 310 \
        --eval-every 31 --save-every 0
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import ray

from sgs.common import load_targets, parse_sorted, ray_runtime_env, solver_messages
from sgs.sgs_grpo import build_configs, parse_args, solver_grpo_update, target_metrics
from sgs.sgs_simple import Policy, evaluate, phase, response_stats


def solver_iteration(t, args, targets, ids, solved, solver, ref):
    """One on-policy GRPO step on targets[ids], in place on `solved`. Returns (metrics, samples)."""
    m, timing = {}, {}
    unsolved = [i for i in ids if not solved[i]]
    problems = [targets[i] for i in ids]
    k = args.k
    with phase(t, "solver_gen", timing,
               f"{len(problems)} problems x {k} = {len(problems) * k} rollouts"):
        rollouts, texts = solver.generate([solver_messages(p) for p in problems], n=k, step=t)
    v = np.array([float(parse_sorted(texts[j]) == sorted(problems[j // k]))
                  for j in range(len(texts))])
    rates = v.reshape(len(problems), k).mean(axis=1)
    for pos, i in enumerate(ids):
        solved[i] = solved[i] or bool(rates[pos] > 0)

    if args.reset_optimizer:
        solver.reset_optimizer()
    m.update(solver_grpo_update(t, args, solver, ref, rollouts, v, timing))
    assert m["solver/n_opt_steps"] in (0, 1), m["solver/n_opt_steps"]  # 0 = all-zero advantages
    m["solver/n_train_rollouts"] = len(v)
    m["solver/n_train_correct"] = int(v.sum())

    m.update({f"timing_s/{key}": val for key, val in timing.items()})
    m["solver/response_len_mean"], m["solver/response_clip_frac"] = response_stats(rollouts)
    m.update(target_metrics(ids, unsolved, rates, solved))
    examples = [{"problem": problems[j // k], "completion": texts[j], "correct": bool(v[j])}
                for j in range(0, len(texts), max(1, len(texts) // 8))]
    return m, {"solver_examples": examples}


@ray.remote(num_cpus=1)
class SolverRunner:
    """sgs_grpo.GRPORunner trimmed to the Solver (+ frozen ref for the KL)."""

    def run(self, args):
        from tqdm import tqdm
        from transformers import AutoTokenizer
        from verl.single_controller.ray import RayResourcePool
        from verl.utils.profiler.performance import simple_timer

        rng = random.Random(args.seed)
        out_dir = Path(args.out_dir)
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_samples").mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

        targets = load_targets(args.dataset)
        solved = [False] * len(targets)
        print(f"[data] {len(targets)} targets from {args.dataset}", flush=True)
        val_targets = load_targets(args.val_dataset) if args.val_dataset else None
        if val_targets:
            print(f"[data] {len(val_targets)} val targets from {args.val_dataset}", flush=True)

        tok = AutoTokenizer.from_pretrained(args.model)
        cfgs = build_configs(args)
        tags = {"solver": cfgs["solver"]}
        if args.kl_coef > 0:
            tags["ref"] = cfgs["guide"]   # frozen init model, never updated
        pool = RayResourcePool(process_on_nodes=[args.n_gpus], use_gpu=True,
                               max_colocate_count=len(tags), name_prefix="sgs")
        policies = {}
        for tag, cfg in tags.items():
            print(f"[init] {tag}", flush=True)
            policies[tag] = Policy(tag, cfg, pool, tok)
        solver, ref = policies["solver"], policies.get("ref")

        wandb = None
        if not args.no_wandb:
            import wandb
            wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args),
                       dir=str(out_dir))
            print(f"[wandb] {wandb.run.url}", flush=True)

        def run_eval(t):
            ev, rows = evaluate(t, targets, solver)
            if val_targets:
                val_ev, val_rows = evaluate(t, val_targets, solver, prefix="val")
                ev.update(val_ev)
                rows += val_rows
            with open(out_dir / "eval_samples" / f"iter_{t:04d}.jsonl", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            print(f"[eval {t}] " + " ".join(f"{k}={v:.4g}" for k, v in ev.items()), flush=True)
            with open(out_dir / "eval_pass5.jsonl", "a") as f:
                f.write(json.dumps({"iter": t, **ev}) + "\n")
            return ev

        if args.eval_every:
            ev = run_eval(0)
            if wandb is not None:
                wandb.log(ev, step=0, commit=True)

        # verl's train dataloader: reshuffle each epoch, drop the last partial batch.
        B = args.mini_batch_size
        steps_per_epoch = len(targets) // B
        perm = list(range(len(targets)))
        pbar = tqdm(total=args.iters, desc="Training Progress")
        for t in range(1, args.iters + 1):
            t0 = time.time()
            s = (t - 1) % steps_per_epoch
            if s == 0:
                rng.shuffle(perm)
            ids = perm[s * B:(s + 1) * B]
            metrics, samples = solver_iteration(t, args, targets, ids, solved, solver, ref)
            metrics["epoch"] = (t - 1) // steps_per_epoch
            timing = {}
            if args.eval_every and t % args.eval_every == 0:
                with simple_timer("eval", timing):
                    metrics.update(run_eval(t))
            if (args.save_every and t % args.save_every == 0) or t == args.iters:
                with simple_timer("save", timing):
                    solver.save(out_dir / "ckpt" / f"iter_{t:04d}" / "solver", t)
                    (out_dir / "solved.json").write_text(json.dumps(solved))
            metrics.update({f"timing_s/{key}": val for key, val in timing.items()})
            metrics["iter"], metrics["time/iter_sec"] = t, time.time() - t0
            print(f"[iter {t}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                            for k, v in sorted(metrics.items())), flush=True)
            with open(out_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(metrics) + "\n")
            with open(out_dir / "samples" / f"iter_{t:04d}.json", "w") as f:
                json.dump(samples, f, indent=1)
            if wandb is not None:
                wandb.log(metrics, step=t, commit=True)
            pbar.set_postfix(solve_rate=f"{metrics['cumulative_solve_rate']:.3f}",
                             iter_sec=f"{metrics['time/iter_sec']:.0f}")
            pbar.update(1)
        pbar.close()

        if wandb is not None:
            wandb.finish()


def main():
    args = parse_args()
    if args.batch_size not in (0, args.mini_batch_size):
        raise SystemExit(f"--batch-size {args.batch_size} != --mini-batch-size {args.mini_batch_size}: "
                         "every step must be on-policy (one optimizer step per sampled batch)")
    ray.init(runtime_env=ray_runtime_env())
    ray.get(SolverRunner.remote().run.remote(args))


if __name__ == "__main__":
    # Re-import under the package name so Ray pickles our classes by reference
    # (`sgs.handrolled_solver_grpo.X`) instead of by value from `__main__`.
    from sgs import handrolled_solver_grpo
    handrolled_solver_grpo.main()
