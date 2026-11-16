"""On-policy SGS-GRPO with majority-vote rewards on synthetic problems, on sorting.

Same as `sgs.sgs_onpolicy_grpo` except synthetic problems x~ from the Conjecturer
are scored without ground truth. For each x~, the k Solver rollouts vote:

    mode, count = most common parsed answer (unparsable answers never vote)
    agree = count / k
    agree >= --mv-ratio  ->  v_j = 1[answer_j == mode]      (pseudo-label = mode)
    otherwise            ->  v_j = 0 for all k              (zero GRPO advantage)

This replaces ground truth for synthetic problems in both places it was used:
the Solver GRPO reward, and s(x~) for the Conjecturer's R_solve (synth_eval).
Targets keep the ground-truth check. `synth/mv_label_acc` and `synth/gt_solve_rate`
compare with ground truth for logging only.

At --mv-ratio 1.0, passing groups are unanimous (zero advantage) and s(x~) is 0 or 1,
so R_solve is always 0 and the Conjecturer never updates.

Usage (4 GPUs; see files/scripts/sgs/sgs_onpolicy_grpo_majority_vote_sorting_qwen3_4b.sbatch):
    python -m sgs.sgs_onpolicy_grpo_majority_vote --mv-ratio 0.5 \
        --dataset files/data/sorting/sorting_1000.jsonl --out-dir runs/sgs_mv --n-gpus 4
"""

import argparse
from collections import Counter

import numpy as np
import ray

from sgs.common import parse_sorted, ray_runtime_env, solver_messages
from sgs.sgs_grpo import (GRPORunner, batch_ids, conjecture, parse_args, reward_and_update_conj,
                          solver_grpo_update, target_metrics)
from sgs.sgs_onpolicy_grpo import verify
from sgs.sgs_simple import phase, response_stats


def majority_vote(texts, k, ratio):
    """Majority-vote reward per rollout (row j is sample j % k of problem j // k).
    Returns (v, agree per problem, label per problem; None if below `ratio`)."""
    v, agree, labels = np.zeros(len(texts)), [], []
    for g in range(len(texts) // k):
        ans = [parse_sorted(t) for t in texts[g * k:(g + 1) * k]]
        votes = Counter(tuple(a) for a in ans if a is not None)
        mode, count = votes.most_common(1)[0] if votes else (None, 0)
        agree.append(count / k)
        if votes and count / k >= ratio:
            v[g * k:(g + 1) * k] = [float(a is not None and tuple(a) == mode) for a in ans]
            labels.append(list(mode))
        else:
            labels.append(None)
    return v, np.array(agree), labels


def sgs_onpolicy_grpo_mv_iteration(t, args, targets, solved, solver, conj, guide, rng):
    """Run one on-policy SGS-GRPO-MV iteration in place on `solved`. Returns (metrics, samples)."""
    m, timing = {}, {}

    # --- steps 1-2: batch B, conjectures -------------------------------------
    ids, unsolved, todo = batch_ids(args, targets, solved, rng)
    conj_rollouts, conjs = conjecture(t, args, targets, todo, conj, timing)
    synth = [c for c in conjs if c["problem"] is not None]

    # --- pre-training MV solve rates s(x~) on B_synth with θ_old -------------
    synth_problems = [c["problem"] for c in synth]
    synth_rates = np.zeros(0)
    if synth:
        with phase(t, "synth_eval", timing,
                   f"{len(synth)} problems x {args.k} = {len(synth) * args.k} rollouts"):
            _, texts = solver.generate([solver_messages(p) for p in synth_problems],
                                       n=args.k, step=t)
        v_mv, agree, labels = majority_vote(texts, args.k, args.mv_ratio)
        synth_rates = v_mv.reshape(len(synth), args.k).mean(axis=1)
        passed = [lab is not None for lab in labels]
        for c, a, lab in zip(synth, agree, labels):
            c["mv_agree"], c["mv_label"] = float(a), lab
        m["synth/mv_pass_frac"] = float(np.mean(passed))
        m["synth/mv_agree_mean"] = float(agree.mean())
        m["synth/mv_label_acc"] = (float(np.mean([lab == sorted(p) for lab, p in
                                                  zip(labels, synth_problems) if lab is not None]))
                                   if any(passed) else 0.0)
        m["synth/gt_solve_rate"] = float(verify(texts, synth_problems, args.k).mean())

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
        is_synth = np.repeat([p >= len(ids) for p in chunk], k)
        v = np.where(is_synth, majority_vote(texts, k, args.mv_ratio)[0],
                     verify(texts, chunk_problems, k))
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
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--mv-ratio", type=float, required=True,
                    help="min share of the k rollouts that must agree for a synthetic problem "
                         "to get any reward")
    mv, rest = ap.parse_known_args()
    args = parse_args(rest)
    args.mv_ratio = mv.mv_ratio
    ray.init(runtime_env=ray_runtime_env())
    ray.get(GRPORunner.remote().run.remote(args, sgs_onpolicy_grpo_mv_iteration))


if __name__ == "__main__":
    # Re-import under the package name so Ray pickles our classes by reference
    # (`sgs.sgs_onpolicy_grpo_majority_vote.X`) instead of by value from `__main__`.
    from sgs import sgs_onpolicy_grpo_majority_vote
    sgs_onpolicy_grpo_majority_vote.main()
