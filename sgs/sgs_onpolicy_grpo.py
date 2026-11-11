"""SGS with an on-policy GRPO Solver, on sorting, built on verl.

Same as `sgs.sgs_grpo` (GRPO Solver with KL to the frozen Guide, unchanged
REINFORCE Conjecturer, optional --all-targets) except for the Solver phase,
which follows `sgs.solver_simple` (verl RayPPOTrainer with train batch =
mini-batch): every optimizer step uses rollouts sampled from the current weights.

    for each full chunk of `mini_batch_size` problems of B ∪ B_synth (shuffled;
    the last partial chunk is dropped, as verl's drop_last):
        sample k rollouts per problem from the current π_θ
        GRPO advantages + Guide ref log-probs -> exactly one optimizer step

Because the Solver changes across chunks, the synthetic solve rates s(x~) for
R_solve are measured before the Solver phase with θ_old, as in sgs_grpo (k fresh
rollouts per synthetic problem, `synth_eval` phase). Target `solved` tracking
uses the training rollouts (dropped problems don't count this iteration).

Usage (4 GPUs; see files/scripts/sgs/sgs_onpolicy_grpo_sorting_qwen3_4b.sbatch):
    python -m sgs.sgs_onpolicy_grpo --dataset files/data/sorting/sorting_1000.jsonl \
        --out-dir runs/sgs_onpolicy_grpo --n-gpus 4
"""

import numpy as np
import ray

from sgs.common import parse_sorted, ray_runtime_env, solver_messages
from sgs.sgs_grpo import (GRPORunner, batch_ids, conjecture, parse_args, reward_and_update_conj,
                          solver_grpo_update, target_metrics)
from sgs.sgs_simple import phase, response_stats


def verify(texts, problems, k):
    """Binary verifier per rollout (row j is sample j % k of problems[j // k])."""
    return np.array([float(parse_sorted(texts[j]) == sorted(problems[j // k]))
                     for j in range(len(texts))])


def sgs_onpolicy_grpo_iteration(t, args, targets, solved, solver, conj, guide, rng):
    """Run one on-policy SGS-GRPO iteration in place on `solved`. Returns (metrics, samples)."""
    m, timing = {}, {}

    # --- steps 1-2: batch B, conjectures -------------------------------------
    ids, unsolved, todo = batch_ids(args, targets, solved, rng)
    conj_rollouts, conjs = conjecture(t, args, targets, todo, conj, timing)
    synth = [c for c in conjs if c["problem"] is not None]

    # --- pre-training solve rates s(x~) on B_synth with θ_old ----------------
    synth_problems = [c["problem"] for c in synth]
    synth_rates = np.zeros(0)
    if synth:
        with phase(t, "synth_eval", timing,
                   f"{len(synth)} problems x {args.k} = {len(synth) * args.k} rollouts"):
            _, texts = solver.generate([solver_messages(p) for p in synth_problems],
                                       n=args.k, step=t)
        synth_rates = verify(texts, synth_problems, args.k).reshape(len(synth), args.k).mean(axis=1)

    # --- steps 3 + 6: on-policy Solver GRPO, one optimizer step per chunk ----
    problems = [targets[i] for i in ids] + [c["problem"] for c in synth]
    k, M = args.k, args.mini_batch_size
    order = list(range(len(problems)))
    rng.shuffle(order)
    chunks = [order[s:s + M] for s in range(0, len(order) - M + 1, M)]   # drop_last
    if args.reset_optimizer:
        solver.reset_optimizer()
    rates = np.zeros(len(problems))
    trained = np.zeros(len(problems), dtype=bool)
    chunk_metrics, lens, clips, examples = [], [], [], []
    n_correct = 0
    for c_idx, chunk in enumerate(chunks):
        chunk_problems = [problems[p] for p in chunk]
        with phase(t, "solver_gen", timing, f"chunk {c_idx + 1}/{len(chunks)}: "
                   f"{len(chunk)} problems x {k} = {len(chunk) * k} rollouts"):
            rollouts, texts = solver.generate([solver_messages(p) for p in chunk_problems],
                                              n=k, step=t)
        v = verify(texts, chunk_problems, k)
        rates[chunk] = v.reshape(len(chunk), k).mean(axis=1)
        trained[chunk] = True
        n_correct += int(v.sum())
        chunk_metrics.append(solver_grpo_update(t, args, solver, guide, rollouts, v, timing))
        ln, cl = response_stats(rollouts)
        lens.append(ln)
        clips.append(cl)
        examples.append({"problem": chunk_problems[0], "completion": texts[0], "correct": bool(v[0])})
    for pos, i in enumerate(ids):
        if trained[pos]:
            solved[i] = solved[i] or bool(rates[pos] > 0)
    keys = {key for cm in chunk_metrics for key in cm}
    m.update({key: float(np.mean([cm[key] for cm in chunk_metrics if key in cm])) for key in keys})
    m["solver/n_opt_steps"] = int(sum(cm["solver/n_opt_steps"] for cm in chunk_metrics))
    m["solver/n_chunks"] = len(chunks)
    m["solver/n_dropped_problems"] = int(len(problems) - trained.sum())
    m["solver/n_train_rollouts"] = int(trained.sum()) * k
    m["solver/n_train_correct"] = n_correct
    m["solver/response_len_mean"], m["solver/response_clip_frac"] = (
        float(np.mean(lens)) if lens else 0.0, float(np.mean(clips)) if clips else 0.0)

    synth_trained = trained[len(ids):]
    m["synth/mean_solve_rate_train"] = (float(rates[len(ids):][synth_trained].mean())
                                        if synth_trained.any() else 0.0)

    # --- steps 4, 5, 7: Guide, rewards, Conjecturer --------------------------
    m.update(reward_and_update_conj(t, args, conjs, synth, synth_rates, conj_rollouts, conj,
                                    guide, rng, timing))

    # --- metrics -------------------------------------------------------------
    m.update({f"timing_s/{key}": val for key, val in timing.items()})
    # Targets dropped this iteration have no rollouts: report over trained ones only.
    tids = [i for pos, i in enumerate(ids) if trained[pos]]
    tset = set(tids)
    m.update(target_metrics(tids, [i for i in unsolved if i in tset],
                            rates[:len(ids)][trained[:len(ids)]], solved))
    return m, {"conjectures": conjs, "solver_examples": examples[::max(1, len(examples) // 8)]}


def main():
    args = parse_args()
    ray.init(runtime_env=ray_runtime_env())
    ray.get(GRPORunner.remote().run.remote(args, sgs_onpolicy_grpo_iteration))


if __name__ == "__main__":
    # Re-import under the package name so Ray pickles our classes by reference
    # (`sgs.sgs_onpolicy_grpo.X`) instead of by value from `__main__`.
    from sgs import sgs_onpolicy_grpo
    sgs_onpolicy_grpo.main()
