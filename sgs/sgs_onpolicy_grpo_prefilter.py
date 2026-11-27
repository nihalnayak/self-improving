"""On-policy SGS-GRPO that conjectures only for targets the base Solver fails.

Same as `sgs.sgs_onpolicy_grpo` except for `solved` at the start. There every
target starts unsolved, so iteration 1 conjectures for all of them. Here, before
iteration 1, the base Solver samples k rollouts per target; a target with any
correct rollout starts solved, so conjectures (B_synth) are generated only for
targets the base model fails. From then on `solved` updates from the training
rollouts as before. The Solver still trains on all targets B ∪ B_synth.

The pre-filter is generation only, so it does not count toward --max-solver-steps;
the step-0 / iter-0 eval runs before it, on the same base model.

Usage (4 GPUs; see files/scripts/sgs_steps_match/sgs_onpolicy_grpo_prefilter_sorting_qwen3_4b.sbatch):
    python -m sgs.sgs_onpolicy_grpo_prefilter --dataset data/sorting/sorting_1000.jsonl \
        --out-dir runs/sgs_prefilter --n-gpus 4 --max-solver-steps 100 --eval-every-steps 10
"""

import ray

from sgs.common import ray_runtime_env
from sgs.sgs_grpo import GRPORunner, parse_args
from sgs.sgs_onpolicy_grpo import sgs_onpolicy_grpo_iteration, verify
from sgs.sgs_simple import phase
from sgs.tasks import TASKS


def sgs_onpolicy_grpo_prefilter_iteration(t, args, targets, solved, solver, conj, guide, rng,
                                          solver_step=None):
    """sgs_onpolicy_grpo_iteration, preceded at t == 1 by the base-Solver pre-filter."""
    m = {}
    if t == 1:
        timing = {}
        task = TASKS[args.task]
        with phase(t, "prefilter", timing, f"{len(targets)} targets x {args.k} rollouts"):
            _, texts = solver.generate([task.solver_messages(x) for x in targets],
                                       n=args.k, step=t)
        passed = verify(texts, targets, args.k, task).reshape(len(targets), args.k).any(axis=1)
        for i, p in enumerate(passed):
            solved[i] = bool(p)
        m["prefilter/n_solved"] = int(passed.sum())
        m["prefilter/solve_frac"] = float(passed.mean())
        m["timing_s/prefilter"] = timing["prefilter"]
        print(f"[prefilter] base Solver solves {int(passed.sum())}/{len(targets)} targets "
              f"(pass@{args.k}); conjecturing for the rest", flush=True)
    out, samples = sgs_onpolicy_grpo_iteration(t, args, targets, solved, solver, conj, guide,
                                               rng, solver_step=solver_step)
    return {**m, **out}, samples


def main():
    args = parse_args()
    assert not args.all_targets, "--all-targets conjectures for every target; pre-filter is moot"
    ray.init(runtime_env=ray_runtime_env())
    ray.get(GRPORunner.remote().run.remote(args, sgs_onpolicy_grpo_prefilter_iteration))


if __name__ == "__main__":
    # Re-import under the package name so Ray pickles our classes by reference.
    from sgs import sgs_onpolicy_grpo_prefilter
    sgs_onpolicy_grpo_prefilter.main()
