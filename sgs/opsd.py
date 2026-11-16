"""On-policy self-distillation (OPSD / SDPO-style) on sorting with verl.

The student (Qwen3-4B, trained) samples k responses per prompt on the plain sort
prompt, as in GRPO. A frozen copy of the initial model (verl's distillation
teacher server) scores each response with the gold sorted list in its prompt,
using OPSD's reference-solution block (misc/sort_opsd_kl.teacher_text). The student
is trained only on forward KL(teacher || student) over the teacher's top-k
tokens, backpropagated directly (verl `forward_kl_topk`). There is no verifier-reward
loss and no KL-to-ref. Everything else (data, val pass@5, optimizer, rollout) is
sgs.solver_simple.

    OPSD:  github.com/siyan-zhao/OPSD   (frozen teacher, gold solution, forward KL)
    SDPO:  arXiv 2601.20802             (distillation-only RL via self-teacher)

verl only lacks one thing: its teacher always scores the student's own prompt.
OPSDAgentLoopWorker swaps in the hinted teacher prompt, then drops the extra
leading rows. Teacher row i is the distribution over token i+1 (same as the student
logits), so after dropping len(teacher_prompt) - len(prompt) rows, the response rows
line up exactly. Only prompt rows (masked out of the loss) differ.

Per eval (before training + once per epoch, on train and val targets), besides
verl's eval_samples/{step}.jsonl generations, this writes:
    eval_metrics.jsonl                 {step, epoch, eval/pass@5/..., val/pass@5/..., val-core/...}
    eval_samples/{step}_targets.jsonl  {step, split, length, uid, n_correct, n} per target

Usage (4 GPUs: 2 trainer + 2 teacher; see files/scripts/sgs/opsd_sorting_qwen3_4b.sbatch):
    python -m sgs.opsd --dataset files/data/sorting/sorting_1000_L8-128.jsonl \
        --val-dataset files/data/sorting/sorting_val_100.jsonl --out-dir runs/opsd \
        --n-gpus 2 --teacher-gpus 2
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import ray
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

from misc.sort_opsd_kl import teacher_text
from sgs import solver_simple
from sgs.common import load_targets, ray_runtime_env, solver_messages, user_msg


class OPSDAgentLoopWorker(AgentLoopWorker):
    """verl's agent loop worker, but the teacher scores the response under the hinted prompt."""

    def _tokenize(self, messages):
        """Prompt ids via verl's own Continuous Token builder, as the agent loop builds the student's."""
        if not hasattr(self, "_ct_builder"):
            self._ct_builder = create_continuous_token_builder(
                self.tokenizer, hf_model_type=self.hf_model_type,
                chat_template_kwargs=self.config.data.get("apply_chat_template_kwargs", {}))
        return self._ct_builder.build_initial_tokens(list(messages))

    async def _compute_teacher_logprobs(self, output, prompt_ids, response_ids, validate,
                                        sample_kwargs=None):
        if validate or not self.distillation_enabled:
            return
        # Our tokenization must reproduce verl's student prompt, or the row offset is wrong.
        assert self._tokenize(sample_kwargs["raw_prompt"]) == list(prompt_ids)
        t_ids = self._tokenize(sample_kwargs["extra_info"]["teacher_prompt"])
        n = len(t_ids) - len(prompt_ids)
        assert n > 0
        await super()._compute_teacher_logprobs(output, t_ids, response_ids, validate, sample_kwargs)
        for k in ("teacher_ids", "teacher_logprobs"):
            output.extra_fields[k] = output.extra_fields[k][n:]


class OPSDAgentLoopManager(AgentLoopManager):
    agent_loop_workers_class = ray.remote(OPSDAgentLoopWorker)


_val_metrics_update = RayPPOTrainer._val_metrics_update  # already patched by solver_simple


def _val_metrics_update_and_save(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
    """solver_simple's val metrics, also written to <out>/eval_metrics.jsonl + per-target pass counts."""
    m = _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns)
    out_dir = Path(self.config.trainer.validation_data_dir).parent
    step = self.global_steps
    with open(out_dir / "eval_metrics.jsonl", "a") as f:
        f.write(json.dumps({"step": step, "epoch": step / len(self.train_dataloader),
                            **{k: float(v) for k, v in m.items()}}) + "\n")
    acc = reward_extra_infos_dict.get("acc") or reward_extra_infos_dict["reward"]
    n_correct, n, src = defaultdict(int), defaultdict(int), {}
    for ds, uid, a in zip(data_sources, sample_uids, acc):
        n_correct[uid] += int(a > 0.5)
        n[uid] += 1
        src[uid] = ds
    samples_dir = out_dir / "eval_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    with open(samples_dir / f"{step}_targets.jsonl", "w") as f:
        for uid, ds in src.items():
            f.write(json.dumps({"step": step, "split": "val" if ds.startswith("val_") else "train",
                                "length": int(ds.split("_")[-1]), "uid": str(uid),
                                "n_correct": n_correct[uid], "n": n[uid]}) + "\n")
    return m


# Patched in the TaskRunner actor too: Ray imports this module there to load OPSDTaskRunner.
RayPPOTrainer._val_metrics_update = _val_metrics_update_and_save


@ray.remote(num_cpus=1)
class OPSDTaskRunner(solver_simple.SolverTaskRunner.__ray_metadata__.modified_class):
    """solver_simple's TaskRunner, imported from this module so the patch above applies."""


def write_parquet(targets, path, source_prefix=""):
    """solver_simple.write_parquet rows + the teacher's hinted prompt in extra_info."""
    from datasets import Dataset
    rows = [{"prompt": solver_messages(xs),
             "data_source": f"{source_prefix}sort_len_{len(xs)}",
             "reward_model": {"style": "rule", "ground_truth": sorted(xs)},
             "extra_info": {"index": i, "teacher_prompt": user_msg(teacher_text(xs, sorted(xs)))}}
            for i, xs in enumerate(targets)]
    Dataset.from_list(rows).to_parquet(str(path))


def main():
    from verl.trainer.main_ppo import run_ppo

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--teacher-gpus", type=int, default=2, help="GPUs for the frozen teacher (TP=1 each)")
    ap.add_argument("--topk", type=int, default=100, help="teacher top-k logprobs for forward KL")
    opsd_args, rest = ap.parse_known_args()
    args, hydra_overrides = solver_simple.parse_args(rest)
    out_dir = Path(args.out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    targets = load_targets(args.dataset)
    if args.eval_every is None:
        # verl's train dataloader drops the last partial batch
        args.eval_every = max(1, len(targets) // args.train_batch_size)
    (out_dir / "args.json").write_text(json.dumps(
        {**vars(args), **vars(opsd_args), "hydra_overrides": hydra_overrides}, indent=2))
    data_path = (out_dir / "data" / "train.parquet").resolve()
    write_parquet(targets, data_path)
    print(f"[data] {len(targets)} targets from {args.dataset} -> {data_path}", flush=True)
    val_paths = [data_path]
    if args.val_dataset:
        val_targets = load_targets(args.val_dataset)
        val_paths.append((out_dir / "data" / "val.parquet").resolve())
        write_parquet(val_targets, val_paths[-1], source_prefix="val_")
        print(f"[data] {len(val_targets)} val targets from {args.val_dataset} -> {val_paths[-1]}",
              flush=True)

    d, t = "distillation", "distillation.teacher_models.teacher_model"
    opsd_overrides = [
        f"{d}.enabled=true",
        f"{d}.n_gpus_per_node={opsd_args.teacher_gpus}",
        f"{d}.nnodes=1",
        f"{t}.model_path={args.model}",
        f"{t}.inference.tensor_model_parallel_size=1",
        f"{d}.distillation_loss.loss_mode=forward_kl_topk",
        f"{d}.distillation_loss.topk={opsd_args.topk}",
        f"{d}.distillation_loss.use_task_rewards=false",
        f"{d}.distillation_loss.use_policy_gradient=false",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={__name__}.OPSDAgentLoopManager",
    ]
    cfg = solver_simple.build_config(args, data_path, val_paths, opsd_overrides + hydra_overrides)
    cfg.actor_rollout_ref.actor.use_kl_loss = False  # no ref model, as in OPSD / SDPO
    ray.init(runtime_env=ray_runtime_env())
    run_ppo(cfg, task_runner_class=OPSDTaskRunner)


if __name__ == "__main__":
    # Re-import under the package name so Ray pickles our classes by reference
    # (`sgs.opsd.X`) instead of by value from `__main__`.
    from sgs import opsd
    opsd.main()
