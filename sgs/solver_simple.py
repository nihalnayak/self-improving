"""Solver-only baselines for SGS (sorting, multiplication, lcm or preorder, --task): verl's own PPO trainer with
adv_estimator=grpo (GRPO, default) or reinforce_plus_plus (REINFORCE++).

Only the Solver is trained, on the same targets, prompt and verifier as
sgs_simple (shared through sgs/common.py). Everything else (rollout, group
advantages, KL loss, validation, wandb/console logging, checkpoints) is verl's
`main_ppo` path, as in https://verl.readthedocs.io/en/latest/start/quickstart.html.

    jsonl -> out_dir/data/train.parquet   one row per target, also used as val
    reward = compute_score below          1 if the task verifier accepts (sgs/tasks.py)
    --val-dataset jsonl -> data/val.parquet  held-out targets (data_source val_<sort|mult|lcm|preorder>_len_{L})
    eval/pass@5/{len_L,all}               exact pass@5 on the targets, as in sgs_simple
    val/pass@5/{len_L,all}                exact pass@5 on the val targets
    eval_samples/{step}.jsonl             every val generation (verl validation_data_dir)
    val-core/<sort|mult|lcm|preorder>_len_{L}/acc/...         verl's own val metrics: mean@5 ~ pass@1; best@5 is a
                                          bootstrap (with replacement) estimate, biased low

Notes:
    - trainer.use_v1=false: verl's V1 trainer needs `transfer_queue`, which this
      env does not have, so we use the v0 TaskRunner + RayPPOTrainer.
    - The TaskRunner registers sgs.common.PatchedActorRolloutRefWorker instead of
      verl's stock worker (trl 1.16 + tied lm_head fixes).
    - Unknown CLI args are passed through as Hydra overrides, e.g.
      `trainer.total_training_steps=3`.

Usage (4 GPUs; see files/scripts/sgs/{grpo,reinforce}_sorting_qwen3_*.sbatch):
    python -m sgs.solver_simple --dataset files/data/sorting/sorting_1000.jsonl \
        --out-dir runs/solver_simple --n-gpus 4 [--adv-estimator reinforce_plus_plus]
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import ray

from sgs.common import PatchedActorRolloutRefWorker, ray_runtime_env
from sgs.tasks import TASK_BY_SOURCE, TASKS


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="GRPO / REINFORCE++ Solver baseline with verl")
    ap.add_argument("--task", choices=sorted(TASKS), default="sorting",
                    help="prompts, verifier and data format (sgs/tasks.py)")
    ap.add_argument("--dataset", required=True,
                    help="jsonl of targets (sorting: `list`; multiplication, lcm: `number1`, `number2`; "
                         "preorder: `inorder`, `postorder`)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument("--adv-estimator", choices=["grpo", "reinforce_plus_plus"], default="grpo")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--train-batch-size", type=int, default=32, help="prompts per training step")
    ap.add_argument("--k", type=int, default=8, help="rollouts per prompt (GRPO group size)")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--mini-batch-size", type=int, default=32,
                    help="prompts per optimizer step (x k sequences)")
    ap.add_argument("--micro-batch-size-per-gpu", type=int, default=4)
    ap.add_argument("--kl-coef", type=float, default=0.001, help="low_var_kl loss coefficient")
    ap.add_argument("--thinking", action="store_true",
                    help="enable_thinking=True in the chat template")
    ap.add_argument("--prompt-length", type=int, default=2048)
    ap.add_argument("--response-length", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    ap.add_argument("--no-offload", action="store_true",
                    help="keep FSDP params and optimizer on GPU (no CPU offload)")
    ap.add_argument("--model-dtype", choices=["fp32", "bf16"], default="fp32",
                    help="dtype of FSDP master weights, grads and optimizer state")
    ap.add_argument("--save-every", type=int, default=0, help="training steps; 0 = only at the end")
    ap.add_argument("--eval-every", type=int, default=None,
                    help="pass@5 on targets every N training steps (and before training); "
                         "default: once per epoch (like sgs_simple's once per iteration); 0 = off")
    ap.add_argument("--val-dataset", default=None,
                    help="jsonl of held-out targets; pass@5 on it at each eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="sgs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    return ap.parse_known_args(argv)


def write_parquet(targets, path, task, source_prefix=""):
    """verl RL dataset: chat prompt, data_source {prefix}{task.source}_len_{level}
    (e.g. sort_len_8, val_mult_len_4), task.answer as ground truth."""
    from datasets import Dataset
    rows = [{"prompt": task.solver_messages(xs),
             "data_source": f"{source_prefix}{task.source}_len_{task.level(xs)}",
             "reward_model": {"style": "rule", "ground_truth": task.answer(xs)},
             "extra_info": {"index": i}}
            for i, xs in enumerate(targets)]
    Dataset.from_list(rows).to_parquet(str(path))


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """verl custom reward: the task's binary verifier (same as sgs_simple). The task is
    recovered from data_source, the only task info verl passes here."""
    source = data_source.removeprefix("val_").rsplit("_len_", 1)[0]
    return float(TASK_BY_SOURCE[source].check(solution_str, ground_truth))


def build_config(args, data_path, val_paths, hydra_overrides):
    """verl's default ppo_trainer config with GRPO + our overrides."""
    import verl
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    a, r, ref = "actor_rollout_ref.actor", "actor_rollout_ref.rollout", "actor_rollout_ref.ref"
    offload = not args.no_offload
    loggers = "[console]" if args.no_wandb else "[console,wandb]"
    overrides = [
        # --- data ---
        f"data.train_files={data_path}",
        f"data.val_files=[{','.join(map(str, val_paths))}]",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.max_prompt_length={args.prompt_length}",
        f"data.max_response_length={args.response_length}",
        f"data.seed={args.seed}",
        f"+data.apply_chat_template_kwargs.enable_thinking={args.thinking}",
        # --- reward: binary verifier from this file ---
        f"reward.custom_reward_function.path={os.path.abspath(__file__)}",
        "reward.custom_reward_function.name=compute_score",
        # --- algorithm: GRPO or REINFORCE++ ---
        f"algorithm.adv_estimator={args.adv_estimator}",
        "algorithm.use_kl_in_reward=false",
        # --- model ---
        f"actor_rollout_ref.model.path={args.model}",
        # flash-attn is not installed in this env: use SDPA, and keep padding
        # (verl's packed "remove padding" path is only correct with flash-attn).
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.model.use_remove_padding=false",
        # --- optimizer: same as sgs_simple (Adam b=(0.9,0.95), clip 1.0, constant lr) ---
        f"{a}.optim.lr={args.lr}",
        f"{a}.optim.betas=[0.9,0.95]",
        f"{a}.optim.weight_decay=0.0",
        f"{a}.optim.clip_grad=1.0",
        f"{a}.optim.lr_scheduler_type=constant",
        # --- loss: verl GRPO example defaults ---
        f"{a}.use_kl_loss=true",
        f"{a}.kl_loss_coef={args.kl_coef}",
        f"{a}.kl_loss_type=low_var_kl",
        f"{a}.entropy_coeff=0",
        f"{a}.ppo_epochs=1",
        f"{a}.ppo_mini_batch_size={args.mini_batch_size}",
        f"{a}.ppo_micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        f"{a}.use_dynamic_bsz=false",
        # --- memory ---
        f"{a}.fsdp_config.param_offload={offload}",
        f"{a}.fsdp_config.optimizer_offload={offload}",
        f"{a}.fsdp_config.model_dtype={args.model_dtype}",
        f"{a}.checkpoint.save_contents=[model,optimizer,extra,hf_model]",
        f"{ref}.fsdp_config.param_offload={offload}",
        f"{ref}.fsdp_config.model_dtype={args.model_dtype}",
        f"{ref}.log_prob_micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        # --- rollout: hybrid vLLM, one replica per GPU ---
        f"{r}.name=vllm",
        f"{r}.mode=async",
        f"{r}.tensor_model_parallel_size=1",
        f"{r}.gpu_memory_utilization={args.gpu_memory_utilization}",
        f"{r}.n={args.k}",
        f"{r}.temperature=1.0",
        f"{r}.top_p=1.0",
        f"{r}.log_prob_micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        f"{r}.prompt_length={args.prompt_length}",
        f"{r}.response_length={args.response_length}",
        # --- validation: 5 samples per target at T=1 (pass@5, as in sgs_simple) ---
        f"{r}.val_kwargs.n=5",
        f"{r}.val_kwargs.do_sample=true",
        f"{r}.val_kwargs.temperature=1.0",
        f"{r}.val_kwargs.top_p=1.0",
        # --- trainer ---
        "trainer.use_v1=false",
        "trainer.critic_warmup=0",
        f"trainer.logger={loggers}",
        f"trainer.project_name={args.wandb_project}",
        f"trainer.experiment_name={args.run_name or Path(args.out_dir).name}",
        f"trainer.default_local_dir={Path(args.out_dir).resolve() / 'ckpt'}",
        f"trainer.validation_data_dir={Path(args.out_dir).resolve() / 'eval_samples'}",
        f"trainer.n_gpus_per_node={args.n_gpus}",
        "trainer.nnodes=1",
        f"trainer.total_epochs={args.epochs}",
        # verl saves at the last step whenever save_freq > 0, so "0" -> never mid-run.
        f"trainer.save_freq={args.save_every or 10**9}",
        f"trainer.test_freq={args.eval_every or -1}",
        f"trainer.val_before_train={bool(args.eval_every)}",
        *hydra_overrides,
    ]
    config_dir = os.path.join(os.path.dirname(verl.__file__), "trainer", "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


from verl.trainer.main_ppo_v0 import TaskRunner  # noqa: E402
from verl.trainer.ppo.ray_trainer import RayPPOTrainer  # noqa: E402

_orig_val_metrics_update = RayPPOTrainer._val_metrics_update


def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
    """verl's val metrics + exact pass@n per length and overall, as in sgs_simple.evaluate."""
    m = _orig_val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns)
    acc = reward_extra_infos_dict.get("acc") or reward_extra_infos_dict["reward"]
    passed, src = defaultdict(bool), {}
    for ds, uid, a in zip(data_sources, sample_uids, acc):
        passed[uid] |= bool(a > 0.5)
        src[uid] = ds
    n = self.config.actor_rollout_ref.rollout.val_kwargs.n
    # data_source is <task>_len_{L} (train targets -> eval/) or val_<task>_len_{L} (-> val/)
    level = {u: int(s.rsplit("_", 1)[-1]) for u, s in src.items()}
    for is_val, split in ((False, "eval"), (True, "val")):
        uids = [u for u, s in src.items() if s.startswith("val_") == is_val]
        if not uids:
            continue
        for L in sorted({level[u] for u in uids}):
            m[f"{split}/pass@{n}/len_{L}"] = float(np.mean(
                [passed[u] for u in uids if level[u] == L]))
        m[f"{split}/pass@{n}/all"] = float(np.mean([passed[u] for u in uids]))
    return m


# Patched in the TaskRunner actor too: Ray imports this module there to load SolverTaskRunner.
RayPPOTrainer._val_metrics_update = _val_metrics_update


@ray.remote(num_cpus=1)
class SolverTaskRunner(TaskRunner.__ray_metadata__.modified_class):
    """verl's v0 TaskRunner, but registering our patched actor/rollout/ref worker."""

    def add_actor_rollout_worker(self, config):
        _, wg_cls = super().add_actor_rollout_worker(config)
        for role in self.role_worker_mapping:
            self.role_worker_mapping[role] = ray.remote(PatchedActorRolloutRefWorker)
        return PatchedActorRolloutRefWorker, wg_cls


def main():
    from verl.trainer.main_ppo import run_ppo

    args, hydra_overrides = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    task = TASKS[args.task]
    targets = task.load(args.dataset)
    if args.eval_every is None:
        # verl's train dataloader drops the last partial batch
        args.eval_every = max(1, len(targets) // args.train_batch_size)
    (out_dir / "args.json").write_text(json.dumps(
        {**vars(args), "hydra_overrides": hydra_overrides}, indent=2))
    data_path = (out_dir / "data" / "train.parquet").resolve()
    write_parquet(targets, data_path, task)
    print(f"[data] {len(targets)} targets from {args.dataset} -> {data_path}", flush=True)
    val_paths = [data_path]
    if args.val_dataset:
        val_targets = task.load(args.val_dataset)
        val_paths.append((out_dir / "data" / "val.parquet").resolve())
        write_parquet(val_targets, val_paths[-1], task, source_prefix="val_")
        print(f"[data] {len(val_targets)} val targets from {args.val_dataset} -> {val_paths[-1]}",
              flush=True)

    cfg = build_config(args, data_path, val_paths, hydra_overrides)
    ray.init(runtime_env=ray_runtime_env())
    run_ppo(cfg, task_runner_class=SolverTaskRunner)


if __name__ == "__main__":
    # Re-import under the package name so Ray pickles our classes by reference
    # (`sgs.solver_simple.X`) instead of by value from `__main__`.
    from sgs import solver_simple
    solver_simple.main()
