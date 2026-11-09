"""SGS with a GRPO Solver (off-policy within an iteration), on sorting, built on verl.

Same three models and iteration as `sgs.sgs_simple` (paper Algorithm 1). Only the
Solver update changes:

    sgs_simple   Solver <- REINFORCE½: A = v on problems with s <= 0.5, gpg loss
    sgs_grpo     Solver <- GRPO on all of B ∪ B_synth:
                   A_i = (v_i - mean_k v) / (std_k v + 1e-6)     per k-rollout group
                   PPO clipped ratio vs the vLLM sampling log-probs (θ_old = θ at
                   sampling time), token-mean, + β·KL(π_θ || ρ) (low_var_kl)

The KL reference is the frozen Guide ρ: it is the same init model as the Solver
and is never trained, so no fourth model is loaded.

Loop structure is the authors' (full_pipeline.run_iteration: STEP A generate,
STEP B one epoch): all Solver rollouts are sampled once per iteration, then one
pass of mini-batch optimizer steps. Mini-batch 1 is on-policy; later ones are
stale by their step count, which the PPO ratio + clip corrects.
See `sgs.sgs_onpolicy_grpo` for the on-policy variant.

The Conjecturer update is unchanged from sgs_simple (REINFORCE, A = R_synth, gpg).

--all-targets: one conjecture per target in B (solved or not) instead of only
the unsolved ones, so the Solver trains on |B| + ~|B| problems.

--mini-batch-size counts prompts (as in sgs.solver_simple): each optimizer step
sees mini_batch_size * k sequences, for every model.

File map:
    §1 Args                 sgs_simple's flags + GRPO / all-targets flags
    §2 Config               sgs_simple's verl config, Solver switched to GRPO loss
    §3 GRPO helpers         group advantages, Guide ref log-probs, one Solver update
    §4 SGS-GRPO iteration   steps 1-7
    §5 Runner + main        Ray actor that owns the three Policies and the loop

Usage (4 GPUs; see files/scripts/sgs/sgs_grpo_sorting_qwen3_4b.sbatch):
    python -m sgs.sgs_grpo --dataset files/data/sorting/sorting_1000.jsonl \
        --out-dir runs/sgs_grpo --n-gpus 4
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import ray

from sgs.common import load_targets, parse_sorted, ray_runtime_env, solver_messages
from sgs.sgs_simple import (Policy, build_verl_config, conj_messages, evaluate, guide_messages,
                            guide_score, minmax, parse_conjecture, parse_guide, phase,
                            response_stats, solve_rate_rewards)


# =============================================================================
# §1 Args
# =============================================================================

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SGS with a GRPO Solver on sorting with verl")
    ap.add_argument("--dataset", required=True, help="jsonl with a `list` field per row")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B",
                    help="init for Solver, Conjecturer and Guide (separate copies)")
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=0,
                    help="targets per iteration; 0 = full dataset (paper §4.1)")
    ap.add_argument("--all-targets", action="store_true",
                    help="conjecture for every target in B, not only the unsolved ones")
    ap.add_argument("--k", type=int, default=8, help="Solver rollouts per problem (GRPO group size)")
    ap.add_argument("--lr", type=float, default=1e-6, help="Solver and Conjecturer lr")
    ap.add_argument("--mini-batch-size", type=int, default=32,
                    help="prompts per optimizer step (x k sequences), as in solver_simple")
    ap.add_argument("--micro-batch-size-per-gpu", type=int, default=4)
    ap.add_argument("--kl-coef", type=float, default=0.001,
                    help="low_var_kl loss coefficient vs the frozen Guide; 0 = no KL")
    ap.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip epsilon")
    ap.add_argument("--too-easy-frac", type=float, default=0.3,
                    help="top fraction of synth solve rates that gets R_solve=0")
    ap.add_argument("--lo", type=int, default=0, help="integer range shown to Conjecturer")
    ap.add_argument("--hi", type=int, default=10000)
    ap.add_argument("--thinking", action="store_true",
                    help="enable_thinking=True in the chat template (all three models)")
    ap.add_argument("--prompt-length", type=int, default=2048)
    ap.add_argument("--solver-response-length", type=int, default=1024)
    ap.add_argument("--conj-response-length", type=int, default=1024)
    ap.add_argument("--guide-response-length", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    ap.add_argument("--no-offload", action="store_true",
                    help="keep FSDP params and optimizer on GPU (no CPU offload)")
    ap.add_argument("--model-dtype", choices=["fp32", "bf16"], default="fp32",
                    help="dtype of FSDP master weights, grads and optimizer state")
    ap.add_argument("--reset-optimizer", action="store_true",
                    help="fresh Adam state (m, v, step) before each update, i.e. every "
                         "iteration (authors' code: reset_trainer=True)")
    ap.add_argument("--save-every", type=int, default=5, help="0 = only at the end")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="per-length pass@5 on targets every N iters (and at iter 0); 0 = off")
    ap.add_argument("--val-dataset", default=None,
                    help="jsonl of held-out targets; pass@5 on it at each eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="sgs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    return ap.parse_args(argv)


# =============================================================================
# §2 Config
# =============================================================================

def build_configs(args):
    """{tag: verl config}. sgs_simple's config with mini-batch = prompts x k sequences;
    the Solver switches from gpg to verl's GRPO loss (PPO clip, token-mean, KL)."""
    from omegaconf import OmegaConf

    seq_args = argparse.Namespace(**{**vars(args), "mini_batch_size": args.mini_batch_size * args.k})
    cfgs = {tag: build_verl_config(seq_args, tag, resp_len)
            for tag, resp_len in (("solver", args.solver_response_length),
                                  ("conj", args.conj_response_length),
                                  ("guide", args.guide_response_length))}
    a = "actor_rollout_ref.actor"
    for key, val in {"policy_loss.loss_mode": "vanilla",
                     "loss_agg_mode": "token-mean",
                     "clip_ratio": args.clip_ratio,
                     "clip_ratio_low": args.clip_ratio,
                     "clip_ratio_high": args.clip_ratio,
                     "use_kl_loss": args.kl_coef > 0,
                     "kl_loss_coef": args.kl_coef,
                     "kl_loss_type": "low_var_kl"}.items():
        OmegaConf.update(cfgs["solver"], f"{a}.{key}", val)
    return cfgs


# =============================================================================
# §3 GRPO helpers
# =============================================================================

def grpo_advantages(v, k):
    """Per-rollout GRPO advantage (v - group mean) / (group std + 1e-6), groups of k
    consecutive rollouts (verl compute_grpo_outcome_advantage: unbiased std)."""
    g = np.asarray(v, dtype=float).reshape(-1, k)
    std = g.std(axis=1, ddof=1, keepdims=True) if k > 1 else np.zeros((len(g), 1))
    return ((g - g.mean(axis=1, keepdims=True)) / (std + 1e-6)).reshape(-1)


def ref_log_probs(guide, rollouts):
    """Per-token log-probs of `rollouts` under the frozen Guide (= init Solver),
    shaped like `responses` (verl ray_trainer._compute_ref_log_prob)."""
    from verl import DataProto
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

    keep = ["prompts", "responses", "response_mask", "input_ids", "attention_mask", "position_ids"]
    data = DataProto.from_dict(tensors={k: rollouts.batch[k] for k in keep},
                               meta_info={"temperature": 1.0, "multi_turn": False})
    td = left_right_2_no_padding(data.to_tensordict())
    tu.assign_non_tensor(td, calculate_entropy=False, compute_loss=False)
    out = guide.wg.compute_log_prob(td)
    return no_padding_2_padding(tu.get(out, "log_probs"), td).float()


def solver_grpo_update(t, args, solver, guide, rollouts, v, timing):
    """GRPO update of the Solver on all `rollouts` (groups of k). Returns metrics."""
    m = {}
    adv = grpo_advantages(v, args.k)
    m["solver/frac_zero_adv_groups"] = float((np.abs(adv.reshape(-1, args.k)).sum(1) == 0).mean())
    m["solver/n_opt_steps"] = 0
    if not np.any(adv):
        return m
    ref = None
    if args.kl_coef > 0:
        with phase(t, "solver_ref", timing, f"{len(v)} rollouts (Guide = KL reference)"):
            ref = ref_log_probs(guide, rollouts)
    steps = math.ceil(len(v) / solver.mini_batch_size)
    with phase(t, "solver_update", timing, f"{len(v)} rows -> {steps} optimizer steps "
               f"(mini-batch {solver.mini_batch_size})"):
        m.update({f"solver/{key}": val for key, val in
                  solver.update(rollouts, np.arange(len(v)), adv, ref_log_probs=ref).items()})
    m["solver/n_opt_steps"] = steps
    return m


# =============================================================================
# §4 SGS-GRPO iteration (paper Algorithm 1, GRPO Solver)
# =============================================================================

def conjecture(t, args, targets, todo, conj, timing):
    """Step 2: one conjecture per target in `todo`. Returns (rollouts, records)."""
    conj_rollouts, conj_texts = None, []
    if todo:
        with phase(t, "conj_gen", timing, f"{len(todo)} prompts"):
            conj_rollouts, conj_texts = conj.generate(
                [conj_messages(targets[i], args.lo, args.hi) for i in todo], n=1, step=t)
    conjs = []   # one record per prompted target, aligned with conj_rollouts rows
    for i, text in zip(todo, conj_texts):
        statement, xs = parse_conjecture(text)
        valid = xs is not None and len(xs) >= 2
        conjs.append({"target_idx": i, "target": targets[i], "completion": text,
                      "statement": statement, "problem": xs if valid else None})
    return conj_rollouts, conjs


def reward_and_update_conj(t, args, conjs, synth, synth_rates, conj_rollouts, conj, guide,
                           rng, timing):
    """Steps 4, 5, 7 (unchanged from sgs_simple). Returns metrics."""
    m = {}
    for c, s in zip(synth, synth_rates):
        c["solve_rate"] = float(s)

    # --- step 4: Guide scores each (target, conjecture) pair -----------------
    if synth:
        with phase(t, "guide_gen", timing, f"{len(synth)} pairs"):
            _, guide_texts = guide.generate(
                [guide_messages(c["target"], c["statement"]) for c in synth], n=1, step=t)
        for c, text in zip(synth, guide_texts):
            c["guide_completion"] = text
            c["guide_scores"] = parse_guide(text)
            c["r_guide"] = guide_score(c["guide_scores"])

    # --- step 5: R_synth = minmax(R_solve * R_guide) -------------------------
    r_solve = solve_rate_rewards(synth_rates, args.too_easy_frac, rng) if synth else np.zeros(0)
    for c, rs in zip(synth, r_solve):
        c["r_solve"] = float(rs)
        c["r_synth_raw"] = rs * c["r_guide"]
    raw = np.array([c.get("r_synth_raw", 0.0) for c in conjs])  # unparsable -> 0
    r_synth = minmax(raw)
    for c, r in zip(conjs, r_synth):
        c["r_synth"] = float(r)

    # --- step 7: Conjecturer REINFORCE (A = normalized R_synth) --------------
    m["conj/n_opt_steps"] = 0
    if r_synth.sum() > 0:
        steps = math.ceil(len(conjs) / conj.mini_batch_size)
        with phase(t, "conj_update", timing, f"{len(conjs)} rows -> {steps} optimizer steps "
                   f"(mini-batch {conj.mini_batch_size})"):
            if args.reset_optimizer:
                conj.reset_optimizer()
            m.update({f"conj/{key}": val for key, val in
                      conj.update(conj_rollouts, np.arange(len(conjs)), r_synth).items()})
        m["conj/n_opt_steps"] = steps

    if conj_rollouts is not None:
        m["conj/response_len_mean"], m["conj/response_clip_frac"] = response_stats(conj_rollouts)
    m.update({
        "conj/n_generated": len(conjs),
        "conj/parse_rate": len(synth) / max(1, len(conjs)),
        "conj/mean_len": float(np.mean([len(c["problem"]) for c in synth])) if synth else 0.0,
        "synth/mean_solve_rate": float(np.mean(synth_rates)) if synth else 0.0,
        "synth/frac_s0": float((synth_rates == 0).mean()) if synth else 0.0,
        "synth/frac_s1": float((synth_rates == 1).mean()) if synth else 0.0,
        "synth/mean_r_solve": float(r_solve.mean()) if synth else 0.0,
        "synth/mean_r_guide": float(np.mean([c["r_guide"] for c in synth])) if synth else 0.0,
        "synth/mean_r_synth_raw": float(raw.mean()) if len(raw) else 0.0,
        "guide/parse_rate": (float(np.mean([c["guide_scores"] is not None for c in synth]))
                             if synth else 0.0),
    })
    return m


def batch_ids(args, targets, solved, rng):
    """Step 1: batch B and the targets that get a conjecture."""
    ids = list(range(len(targets)))
    if args.batch_size:
        ids = rng.sample(ids, min(args.batch_size, len(ids)))
    unsolved = [i for i in ids if not solved[i]]
    return ids, unsolved, (ids if args.all_targets else unsolved)


def target_metrics(ids, unsolved, rates, solved):
    n_t = len(ids)
    return {
        "cumulative_solve_rate": float(np.mean(solved)),
        "targets/batch_size": n_t,
        "targets/n_unsolved_before": len(unsolved),
        "targets/mean_solve_rate": float(rates[:n_t].mean()) if n_t else 0.0,
        "targets/frac_any_correct": float((rates[:n_t] > 0).mean()) if n_t else 0.0,
    }


def sgs_grpo_iteration(t, args, targets, solved, solver, conj, guide, rng):
    """Run one SGS-GRPO iteration in place on `solved`. Returns (metrics, sample records)."""
    m, timing = {}, {}

    # --- steps 1-2: batch B, conjectures -------------------------------------
    ids, unsolved, todo = batch_ids(args, targets, solved, rng)
    conj_rollouts, conjs = conjecture(t, args, targets, todo, conj, timing)
    synth = [c for c in conjs if c["problem"] is not None]

    # --- step 3: k Solver rollouts on B ∪ B_synth, all from θ_old ------------
    problems = [targets[i] for i in ids] + [c["problem"] for c in synth]
    k = args.k
    with phase(t, "solver_gen", timing,
               f"{len(problems)} problems x {k} = {len(problems) * k} rollouts"):
        sol_rollouts, sol_texts = solver.generate([solver_messages(p) for p in problems], n=k, step=t)
    v = np.array([float(parse_sorted(sol_texts[j]) == sorted(problems[j // k]))
                  for j in range(len(sol_texts))])
    rates = v.reshape(len(problems), k).mean(axis=1)
    for pos, i in enumerate(ids):
        solved[i] = solved[i] or bool(rates[pos] > 0)
    synth_rates = rates[len(ids):]

    # --- step 6: Solver GRPO on all of B ∪ B_synth (one pass, off-policy) ----
    if args.reset_optimizer:
        solver.reset_optimizer()
    m.update(solver_grpo_update(t, args, solver, guide, sol_rollouts, v, timing))
    m["solver/n_train_rollouts"] = len(v)
    m["solver/n_train_correct"] = int(v.sum())

    # --- steps 4, 5, 7: Guide, rewards, Conjecturer --------------------------
    m.update(reward_and_update_conj(t, args, conjs, synth, synth_rates, conj_rollouts, conj,
                                    guide, rng, timing))

    # --- metrics -------------------------------------------------------------
    m.update({f"timing_s/{key}": val for key, val in timing.items()})
    m["solver/response_len_mean"], m["solver/response_clip_frac"] = response_stats(sol_rollouts)
    m.update(target_metrics(ids, unsolved, rates, solved))
    solver_examples = [{"problem": problems[j // k], "completion": sol_texts[j], "correct": bool(v[j])}
                       for j in range(0, len(sol_texts), max(1, len(sol_texts) // 8))]
    return m, {"conjectures": conjs, "solver_examples": solver_examples}


# =============================================================================
# §5 Runner + main
# =============================================================================

@ray.remote(num_cpus=1)
class GRPORunner:
    """Owns the three models and the training loop (verl's managers expect to
    be driven from inside a Ray actor). `iteration` is the per-iteration function,
    so sgs_onpolicy_grpo reuses this runner."""

    def run(self, args, iteration):
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
        # All three models share every GPU (sequential phases + vLLM sleep + FSDP offload).
        pool = RayResourcePool(process_on_nodes=[args.n_gpus], use_gpu=True,
                               max_colocate_count=3, name_prefix="sgs")
        policies = {}
        for tag, cfg in build_configs(args).items():
            print(f"[init] {tag}", flush=True)
            policies[tag] = Policy(tag, cfg, pool, tok)
        solver, conj, guide = policies["solver"], policies["conj"], policies["guide"]

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

        pbar = tqdm(total=args.iters, desc="Training Progress")
        for t in range(1, args.iters + 1):
            t0 = time.time()
            metrics, samples = iteration(t, args, targets, solved, solver, conj, guide, rng)
            timing = {}
            if args.eval_every and t % args.eval_every == 0:
                with simple_timer("eval", timing):
                    metrics.update(run_eval(t))
            if (args.save_every and t % args.save_every == 0) or t == args.iters:
                with simple_timer("save", timing):
                    for tag in ("solver", "conj"):
                        policies[tag].save(out_dir / "ckpt" / f"iter_{t:04d}" / tag, t)
                    (out_dir / "solved.json").write_text(json.dumps(solved))
            metrics.update({f"timing_s/{key}": val for key, val in timing.items()})
            metrics["iter"], metrics["time/iter_sec"] = t, time.time() - t0
            print(f"[iter {t}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                            for k, v in sorted(metrics.items())), flush=True)
            with open(out_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(metrics) + "\n")
            with open(out_dir / "synthetic.jsonl", "a") as f:
                for c in samples["conjectures"]:
                    if c["problem"] is not None:
                        f.write(json.dumps({"iter": t, "unsolved": c["target"],
                                            "synthetic": c["problem"]}) + "\n")
            with open(out_dir / "samples" / f"iter_{t:04d}.json", "w") as f:
                json.dump(samples, f, indent=1)
            if wandb is not None:
                wandb.log(metrics, step=t, commit=True)
            pbar.set_postfix(solve_rate=f"{metrics['cumulative_solve_rate']:.3f}",
                             n_train=metrics["solver/n_train_rollouts"],
                             iter_sec=f"{metrics['time/iter_sec']:.0f}")
            pbar.update(1)
        pbar.close()

        if wandb is not None:
            wandb.finish()


def main():
    args = parse_args()
    ray.init(runtime_env=ray_runtime_env())
    ray.get(GRPORunner.remote().run.remote(args, sgs_grpo_iteration))


if __name__ == "__main__":
    # Re-import under the package name so Ray pickles our classes by reference
    # (`sgs.sgs_grpo.X`) instead of by value from `__main__`.
    from sgs import sgs_grpo
    sgs_grpo.main()
