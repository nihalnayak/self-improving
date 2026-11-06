"""SGS (Self-Guided Self-Play, arXiv 2604.20209) on sorting, built on verl.

Three separate Qwen3-4B models, all full fine-tuning (no LoRA):
    Solver       (trained)  sorts lists           -- prompt from misc/sort_benchmark.py
    Conjecturer  (trained)  proposes easier lists -- prompt from misc/sort_proposer.py
    Guide        (frozen)   scores conjectures    -- paper §E.3 rubric, adapted to sorting

Paper Algorithm 1 -> code (one call of `sgs_iteration` = one iteration t):

    B <- batch of targets (full D by default, as in paper §4.1)   step 1
    for x in B_unsolved: x~ <- Conjecturer(x)                      step 2 (conjecture)
    for p in B + B_synth: k Solver rollouts, v(y) = verifier        step 3 (solve)
    R_guide(x, x~) <- Guide                                         step 4 (guide)
    R_solve(x~) = 1[s!=0 and s in bottom 70%] * (1 - s)             step 5 (rewards)
    R_synth = minmax_normalize(R_solve * R_guide)                   (§G.5)
    Solver      <- REINFORCE on problems with s <= 0.5 (REINFORCE½)  step 6
    Conjecturer <- REINFORCE with advantage R_synth                  step 7

Both updates use verl's `gpg` policy loss (exactly -A * log pi, no ratio/clip)
with `seq-mean-token-mean` aggregation (= paper Eq. 4's 1/|y| sum_t).

File map:
    §1 Args                 CLI flags (paper App. C defaults, lr=1e-6)
    §2 Prompts              Conjecturer / Solver / Guide user messages
    §3 Parsing + rewards    pure-python verifiers and reward functions
    §4 verl plumbing        `Policy`: one model = FSDP actor + hybrid vLLM server
    §5 SGS iteration        steps 1-7 above
    §6 Runner + main        Ray actor that owns the three Policies and the loop

Chat template: prompts are never pre-templated. Each prompt goes to verl as a
message list [{"role": "user", ...}]; verl's agent loop applies the tokenizer's
chat template with `data.apply_chat_template_kwargs` (enable_thinking=False by
default).

Not implemented (vs paper): soft-overlong length penalty, Guide SFT for format.

Usage (4 GPUs; see files/scripts/sgs/sgs_simple.sbatch):
    python -m sgs.sgs_simple --dataset files/data/sorting/sorting_1000.jsonl \
        --out-dir runs/sgs_simple --n-gpus 4
"""

import argparse
from contextlib import contextmanager
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import ray
import torch

from sgs.common import (PatchedActorRolloutRefWorker, _BRACKET_RE, _ints, load_targets,
                        parse_sorted, ray_runtime_env, solver_messages, user_msg)


# =============================================================================
# §1 Args
# =============================================================================

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SGS on sorting with verl")
    ap.add_argument("--dataset", required=True, help="jsonl with a `list` field per row")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B",
                    help="init for Solver, Conjecturer and Guide (separate copies)")
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=0,
                    help="targets per iteration; 0 = full dataset (paper §4.1)")
    ap.add_argument("--k", type=int, default=8, help="Solver rollouts per problem")
    ap.add_argument("--lr", type=float, default=1e-6, help="Solver and Conjecturer lr")
    ap.add_argument("--mini-batch-size", type=int, default=32,
                    help="sequences per optimizer step (paper App. C)")
    ap.add_argument("--micro-batch-size-per-gpu", type=int, default=4)
    ap.add_argument("--solve-rate-max", type=float, default=0.5, help="REINFORCE½ cutoff")
    ap.add_argument("--too-easy-frac", type=float, default=0.3,
                    help="top fraction of synth solve rates that gets R_solve=0")
    ap.add_argument("--lo", type=int, default=0, help="integer range shown to Conjecturer")
    ap.add_argument("--hi", type=int, default=10000)
    ap.add_argument("--thinking", action="store_true",
                    help="enable_thinking=True in the chat template (all three models)")
    ap.add_argument("--prompt-length", type=int, default=2048)
    ap.add_argument("--solver-response-length", type=int, default=4096)
    ap.add_argument("--conj-response-length", type=int, default=2048)
    ap.add_argument("--guide-response-length", type=int, default=2048)
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="sgs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    return ap.parse_args(argv)


# =============================================================================
# §2 Prompts
# =============================================================================

# Conjecturer: verbatim from misc/sort_proposer.py. `example_list` is the
# unsolved target x, which is how the Conjecturer is conditioned on x.
CONJ_PROMPT = (
    "I am learning to sort lists of integers in [{lo}, {hi}] in "
    "ascending order. This is the task I ultimately want to solve. "
    "Here is one example sorting problem for reference:\n\n"
    "List: {example_list}\n\n"
    "To make progress, I want to first practice on a simpler version: "
    "a shorter and easier sorting problem. "
    "Learning to solve this simpler task will be a "
    "stepping stone toward solving the more challenging "
    "task.\n\n"
    "Generate one such simpler sorting problem and its solution. "
    "The input list must NOT already be sorted and must NOT be an "
    "obvious canonical example. The solution must contain exactly "
    "the same integers, including duplicates, in ascending order.\n\n"
    "Return exactly the following two blocks, replacing the placeholder "
    "lists with actual integers:\n\n"
    "<problem>\n"
    "Sort the numbers in the list in ascending order.\n"
    "List: [a, b, c, ...]\n"
    "</problem>\n"
    "<solution>\n"
    "[sorted integers]\n"
    "</solution>\n\n"
    "Write each list on a single line using comma-separated integers. "
    "Do not include reasoning, explanations, Markdown fences, LaTeX, "
    "or any text outside these two blocks."
)

# Guide: paper §E.3 (relevance 0-5, redundancy 0/1, complexity 0-4), with the
# Lean "lemma / premises / conclusion" wording mapped onto sorting problems.
GUIDE_PROMPT = (
    "You are a sorting expert. Here is a seed sorting problem:\n\n"
    "<seed>\n{seed}\n</seed>\n\n"
    "Here is a simpler sorting problem that is supposed to be useful for "
    "solving the above problem. It can be useful in that either it is a "
    "smaller version of the seed problem that can be directly used as a "
    "stepping stone, or it is a related problem that plausibly requires "
    "similar skills to solve.\n\n"
    "<problem>\n{conjecture}\n</problem>\n\n"
    "Please rate the relevance of the problem to the seed problem on a scale "
    "of 0 to 5, where 0 is \"not at all related\" and 5 is \"very useful for "
    "solving the seed problem.\" If the problem is trivial, you should give "
    "it a low score.\n\n"
    "Here is a rubric for how to score the problem:\n"
    "- 0: The problem is not at all related to the seed problem and is "
    "trivial to solve (e.g. the list has two or fewer numbers or is already "
    "sorted). OR the problem is identical to the seed problem (the same "
    "numbers, possibly reordered).\n"
    "- 1: The problem is not at all related to the seed problem (e.g. it is "
    "not a problem of sorting a list of integers in ascending order).\n"
    "- 2: The problem is related to the seed problem in that it is a sorting "
    "problem, but it is not directly useful for solving the seed problem "
    "(e.g. it is not simpler than the seed problem, or its numbers look "
    "nothing like the seed's).\n"
    "- 3: The problem is related to the seed problem and may be useful for "
    "solving the seed problem.\n"
    "- 4: The problem is directly useful for solving the seed problem. That "
    "is, if the problem was solved, the seed problem would be easier to solve "
    "(e.g. a shorter list whose numbers have similar size, duplicates and "
    "ordering patterns).\n"
    "- 5: The problem is very useful for solving the seed problem, and "
    "solving it will dramatically reduce the difficulty of the seed "
    "problem.\n\n"
    "Next decide how redundant the problem statement is. Rate this as 0 or 1:\n"
    "- 0: There is nothing redundant in the problem statement.\n"
    "- 1: There are redundant parts. That is, instructions, constraints or "
    "text that are not needed to state the sorting task.\n\n"
    "Next decide if the problem statement is overly complex. Rate this on a "
    "score of 0 to 4:\n"
    "- 0: The problem is minimally complex. A single instruction to sort a "
    "single list that is maximally clear.\n"
    "- 1: The problem has low complexity. Multiple related parts, but they "
    "form a cohesive sorting task.\n"
    "- 2: The problem has moderate complexity. Contains a few extra "
    "conditions or alternatives that are all on-theme.\n"
    "- 3: The problem has high complexity. Contains multiple unrelated tasks "
    "or conditions, or nested instructions that obscure the main sorting "
    "task.\n"
    "- 4: The problem has very high complexity. Many largely unrelated tasks "
    "or deeply nested instructions that are hard to parse.\n\n"
    "End your response with the three scores in exactly this format:\n"
    "<relevance>N</relevance>\n"
    "<redundancy>N</redundancy>\n"
    "<complexity>N</complexity>"
)

# How a target is shown to the Guide (same statement format the Conjecturer emits).
SEED_STATEMENT = "Sort the numbers in the list in ascending order.\nList: {list_str}"


def conj_messages(target, lo, hi):
    return user_msg(CONJ_PROMPT.format(lo=lo, hi=hi, example_list=str(target)))


def guide_messages(target, conj_statement):
    seed = SEED_STATEMENT.format(list_str=str(target))
    return user_msg(GUIDE_PROMPT.format(seed=seed, conjecture=conj_statement))


# =============================================================================
# §3 Parsing + rewards
# =============================================================================

_PROBLEM_RE = re.compile(r"<problem>(.*?)</problem>", re.DOTALL | re.IGNORECASE)
def parse_conjecture(text):
    """(statement, list) from the last non-empty <problem> block (misc/sort_proposer.py).

    Returns (None, None) if there is no block or no bracketed list inside it.
    """
    blocks = [b.strip() for b in _PROBLEM_RE.findall(text) if b.strip()]
    if not blocks:
        return None, None
    brackets = _BRACKET_RE.findall(blocks[-1])
    xs = _ints(brackets[-1]) if brackets else None
    return (blocks[-1], xs) if xs else (None, None)


def parse_guide(text):
    """(relevance, redundancy, complexity) from the last tag of each kind, or None."""
    out = []
    for tag, hi in (("relevance", 5), ("redundancy", 1), ("complexity", 4)):
        found = re.findall(rf"<{tag}>\s*(\d+)\s*</{tag}>", text)
        if not found or not 0 <= int(found[-1]) <= hi:
            return None
        out.append(int(found[-1]))
    return tuple(out)


def guide_score(scores):
    """Paper Eq. 3: 0 if complexity >= 3 else max(0, rel + (2 - cx) + (1 - red)) in [0, 8]."""
    if scores is None:
        return 0.0
    rel, red, cx = scores
    if cx >= 3:
        return 0.0
    return float(max(0, rel + (2 - cx) + (1 - red)))


def solve_rate_rewards(rates, too_easy_frac, rng):
    """R_solve per synthetic problem (Alg. 1 lines 12-14).

    0 if s == 0 (too hard) or s is in the top `too_easy_frac` of the batch's
    solve rates (too easy); otherwise 1 - s. Ties at the cut are broken randomly.
    """
    rates = np.asarray(rates, dtype=float)
    n = len(rates)
    order = sorted(range(n), key=lambda i: (-rates[i], rng.random()))
    too_easy = set(order[:math.ceil(too_easy_frac * n)])
    return np.array([0.0 if (rates[i] == 0 or i in too_easy) else 1.0 - rates[i]
                     for i in range(n)])


def minmax(r):
    """§G.5: linearly normalize rewards to [0, 1] within the batch (all-equal -> 0)."""
    r = np.asarray(r, dtype=float)
    span = r.max() - r.min() if len(r) else 0.0
    return (r - r.min()) / span if span > 0 else np.zeros_like(r)


# =============================================================================
# §4 verl plumbing
# =============================================================================
# One `Policy` = one model: a verl ActorRolloutRefWorker group (FSDP training
# engine + hybrid vLLM server on the same GPUs), an AgentLoopManager that turns
# chat messages into rollouts, and a CheckpointEngineManager that syncs FSDP
# weights into vLLM (wake) and frees vLLM memory (sleep).
#
# verl names each hybrid vLLM server Ray actor "vllm_server_{replica}_{node}",
# so three models in one Ray job would collide. We give each model its own
# prefix ("vllm_{tag}_") on both sides of the lookup: the replica that creates
# the server (driver side) and the ServerAdapter that finds it (worker side).

from verl.single_controller.base.decorator import Dispatch, register  # noqa: E402


class TaggedActorRolloutWorker(PatchedActorRolloutRefWorker):
    """Patched worker whose vLLM ServerAdapter looks up `vllm_{tag}_server_*`."""

    def __init__(self, config, role, tag):
        super().__init__(config, role)
        self.server_tag = tag

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        prefix = f"vllm_{self.server_tag}_"
        self.rollout._get_server_name_prefix = lambda: prefix

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_optimizer(self):
        # torch AdamW re-creates per-param state lazily when it's empty, and verl's
        # optimizer offload/load helpers skip an empty state.
        self.actor.engine.optimizer.state.clear()


def build_verl_config(args, tag, response_length):
    """verl's default ppo_trainer config with our overrides, one copy per model."""
    import verl
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    a, r = "actor_rollout_ref.actor", "actor_rollout_ref.rollout"
    overrides = [
        f"actor_rollout_ref.model.path={args.model}",
        # flash-attn is not installed in this env: use SDPA, and keep padding
        # (verl's packed "remove padding" path is only correct with flash-attn).
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.model.use_remove_padding=false",
        # --- optimizer: paper App. C (Adam b=(0.9,0.95), clip 1.0, constant lr) ---
        f"{a}.optim.lr={args.lr}",
        f"{a}.optim.betas=[0.9,0.95]",
        f"{a}.optim.weight_decay=0.0",
        f"{a}.optim.clip_grad=1.0",
        f"{a}.optim.lr_scheduler_type=constant",
        # --- loss: REINFORCE = -A * log pi, mean over tokens then sequences ---
        f"{a}.policy_loss.loss_mode=gpg",
        f"{a}.loss_agg_mode=seq-mean-token-mean",
        f"{a}.use_kl_loss=false",
        f"{a}.entropy_coeff=0",
        f"{a}.ppo_epochs=1",
        f"{a}.ppo_mini_batch_size={args.mini_batch_size}",
        f"{a}.ppo_micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        f"{a}.use_dynamic_bsz=false",
        # --- memory: three models share the GPUs, so offload when idle (unless --no-offload) ---
        f"{a}.fsdp_config.param_offload={not args.no_offload}",
        f"{a}.fsdp_config.optimizer_offload={not args.no_offload}",
        f"{a}.fsdp_config.model_dtype={args.model_dtype}",
        f"{a}.checkpoint.save_contents=[model,optimizer,extra,hf_model]",
        # --- rollout: hybrid vLLM, one replica per GPU ---
        f"{r}.name=vllm",
        f"{r}.mode=async",
        f"{r}.tensor_model_parallel_size=1",
        f"{r}.gpu_memory_utilization={args.gpu_memory_utilization}",
        f"{r}.n=1",  # we repeat prompts ourselves
        f"{r}.temperature=1.0",
        f"{r}.top_p=1.0",
        f"{r}.calculate_log_probs=true",
        f"{r}.free_cache_engine=true",
        f"{r}.checkpoint_engine.backend=naive",
        f"{r}.log_prob_micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        f"{r}.agent.num_workers=8",
        f"{r}.prompt_length={args.prompt_length}",
        f"{r}.response_length={response_length}",
        f"data.max_prompt_length={args.prompt_length}",
        f"data.max_response_length={response_length}",
        # --- chat template kwargs used by verl's agent loop ---
        f"+data.apply_chat_template_kwargs.enable_thinking={args.thinking}",
        f"trainer.n_gpus_per_node={args.n_gpus}",
        "trainer.nnodes=1",
        f"trainer.experiment_name=sgs_{tag}",
    ]
    config_dir = os.path.join(os.path.dirname(verl.__file__), "trainer", "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


class Policy:
    """One model (Solver, Conjecturer or Guide) living on the shared GPU pool."""

    def __init__(self, tag, cfg, pool, tokenizer):
        from verl.checkpoint_engine import CheckpointEngineManager
        from verl.experimental.agent_loop import AgentLoopManager
        from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.rollout.llm_server import LLMServerManager
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

        self.tag, self.cfg, self.tok = tag, cfg, tokenizer
        self.num_agent_workers = cfg.actor_rollout_ref.rollout.agent.num_workers
        self.mini_batch_size = cfg.actor_rollout_ref.actor.ppo_mini_batch_size

        # 1) FSDP actor + rollout adapter in one worker group.
        self.wg = RayWorkerGroup(
            resource_pool=pool,
            ray_cls_with_init=RayClassWithInitArgs(
                cls=ray.remote(TaggedActorRolloutWorker),
                config=cfg.actor_rollout_ref, role="actor_rollout", tag=tag),
            name_prefix=f"{tag}_",
            device_name="cuda",
        )
        self.wg.init_model()

        # 2) vLLM servers (one per GPU), named with this model's prefix.
        prefix = f"vllm_{tag}_"
        replica_cls = type(f"vLLMReplica_{tag}", (vLLMReplica,),
                           {"_get_server_name_prefix": lambda self: prefix})
        manager_cls = type(f"LLMServerManager_{tag}", (LLMServerManager,),
                           {"rollout_replica_class": replica_cls})
        self.servers = manager_cls.create(config=cfg, worker_group=self.wg,
                                          rollout_resource_pool=pool)

        # 3) Agent loop (tokenizes chat messages, calls vLLM, pads outputs).
        self.agents = AgentLoopManager.create(
            config=cfg, llm_client=self.servers.get_client(),
            teacher_client=None, reward_loop_worker_handles=None)

        # 4) FSDP -> vLLM weight sync; start asleep so the next model has room.
        self.ckpt = CheckpointEngineManager(
            config=omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout.checkpoint_engine),
            actor_wg=self.wg, replicas=self.servers.get_replicas())
        self.ckpt.sleep_replicas()

    # ---- generation --------------------------------------------------------
    def generate(self, messages, n, step):
        """Sample `n` responses per chat. Returns (DataProto rows, texts), interleaved
        so row i*n + j is sample j of messages[i]."""
        from verl import DataProto
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        chats = [m for m in messages for _ in range(n)]
        raw = np.empty(len(chats), dtype=object)
        for i, m in enumerate(chats):
            raw[i] = m
        batch = DataProto.from_single_dict(
            {"dummy_tensor": torch.zeros(len(chats), 1, dtype=torch.uint8), "raw_prompt": raw},
            meta_info={"global_steps": step})
        batch, pad = pad_dataproto_to_divisor(batch, self.num_agent_workers)

        self.ckpt.update_weights(step)   # wake: push current FSDP weights into vLLM
        out = self.agents.generate_sequences(batch)
        self.ckpt.sleep_replicas()       # free vLLM memory for the other models
        out = unpad_dataproto(out, pad)

        P = out.batch["prompts"].shape[1]
        resp_mask = out.batch["attention_mask"][:, P:].bool()
        texts = [self.tok.decode(out.batch["responses"][i][resp_mask[i]], skip_special_tokens=True)
                 for i in range(len(out))]
        return out, texts

    # ---- training ----------------------------------------------------------
    def update(self, rollouts, rows, advantages):
        """One REINFORCE pass over `rollouts[rows]` with per-sequence `advantages`.

        Rows are padded up to a multiple of the optimizer mini-batch with
        zero-advantage copies (no gradient). Returns verl's actor metrics.
        """
        from verl import DataProto
        from verl.protocol import pad_dataproto_to_divisor
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import left_right_2_no_padding

        sel = rollouts.select_idxs(np.asarray(rows, dtype=np.int64))
        keep = ["prompts", "responses", "response_mask", "input_ids",
                "attention_mask", "position_ids"]
        tensors = {k: sel.batch[k] for k in keep}
        adv = torch.tensor(advantages, dtype=torch.float32)[:, None]
        tensors["advantages"] = adv * tensors["response_mask"].float()
        # gpg ignores old_log_probs (no ratio), but verl's ppo_loss requires the key.
        tensors["old_log_probs"] = sel.batch["rollout_log_probs"].float()
        data = DataProto.from_dict(tensors=tensors,
                                   meta_info={"temperature": 1.0, "multi_turn": False})
        data, pad = pad_dataproto_to_divisor(data, self.mini_batch_size)
        if pad:
            data.batch["advantages"][-pad:] = 0.0

        td = left_right_2_no_padding(data.to_tensordict())
        tu.assign_non_tensor(
            td, calculate_entropy=False,
            global_batch_size=self.mini_batch_size, mini_batch_size=self.mini_batch_size,
            epochs=1, seed=42, dataloader_kwargs={"shuffle": True}, compute_loss=True)
        metrics = tu.get(self.wg.update_actor(td), "metrics")
        return {k: float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v)
                for k, v in metrics.items()}

    def reset_optimizer(self):
        self.wg.reset_optimizer()

    def save(self, path, step):
        self.wg.save_checkpoint(local_path=str(path), global_step=step, max_ckpt_to_keep=None)


# =============================================================================
# §5 SGS iteration (paper Algorithm 1)
# =============================================================================

from verl.utils.profiler.performance import simple_timer  # noqa: E402


@contextmanager
def phase(t, name, timing, msg):
    """Log start/end of one phase and accumulate its seconds into timing[name]."""
    print(f"[iter {t}] {name}: {msg}", flush=True)
    with simple_timer(name, timing):
        yield
    print(f"[iter {t}] {name} done in {timing[name]:.1f}s", flush=True)


def response_stats(rollouts):
    """(mean response length, fraction that hit the max response length)."""
    lens = rollouts.batch["response_mask"].sum(-1).float()
    max_len = rollouts.batch["responses"].shape[1]
    return float(lens.mean()), float((lens >= max_len).float().mean())


def sgs_iteration(t, args, targets, solved, solver, conj, guide, rng):
    """Run one SGS iteration in place on `solved`. Returns (metrics, sample records)."""
    m, timing = {}, {}

    # --- step 1: batch B, split into solved / unsolved -----------------------
    ids = list(range(len(targets)))
    if args.batch_size:
        ids = rng.sample(ids, min(args.batch_size, len(ids)))
    unsolved = [i for i in ids if not solved[i]]

    # --- step 2: one conjecture per unsolved target --------------------------
    conj_rollouts, conj_texts = None, []
    if unsolved:
        with phase(t, "conj_gen", timing, f"{len(unsolved)} prompts"):
            conj_rollouts, conj_texts = conj.generate(
                [conj_messages(targets[i], args.lo, args.hi) for i in unsolved], n=1, step=t)
    conjs = []   # one record per unsolved target, aligned with conj_rollouts rows
    for i, text in zip(unsolved, conj_texts):
        statement, xs = parse_conjecture(text)
        valid = xs is not None and len(xs) >= 2
        conjs.append({"target_idx": i, "target": targets[i], "completion": text,
                      "statement": statement, "problem": xs if valid else None})
    synth = [c for c in conjs if c["problem"] is not None]

    # --- step 3: k Solver rollouts on B ∪ B_synth, binary verification -------
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

    # --- step 6: Solver REINFORCE½ (problems with s <= 0.5, A = v) -----------
    hard = np.repeat(rates <= args.solve_rate_max, k)
    rows = np.nonzero(hard)[0]
    m["solver/n_opt_steps"] = 0
    if v[rows].sum() > 0:
        steps = math.ceil(len(rows) / solver.mini_batch_size)
        with phase(t, "solver_update", timing, f"{len(rows)} rows -> {steps} optimizer steps "
                   f"(mini-batch {solver.mini_batch_size})"):
            if args.reset_optimizer:
                solver.reset_optimizer()
            m.update({f"solver/{key}": val for key, val in solver.update(sol_rollouts, rows, v[rows]).items()})
        m["solver/n_opt_steps"] = steps
    m["solver/n_train_rollouts"] = len(rows)
    m["solver/n_train_correct"] = int(v[rows].sum())

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

    # --- metrics -------------------------------------------------------------
    n_t = len(ids)
    m.update({f"timing_s/{key}": val for key, val in timing.items()})
    m["solver/response_len_mean"], m["solver/response_clip_frac"] = response_stats(sol_rollouts)
    if conj_rollouts is not None:
        m["conj/response_len_mean"], m["conj/response_clip_frac"] = response_stats(conj_rollouts)
    m.update({
        "cumulative_solve_rate": float(np.mean(solved)),
        "targets/batch_size": n_t,
        "targets/n_unsolved_before": len(unsolved),
        "targets/mean_solve_rate": float(rates[:n_t].mean()) if n_t else 0.0,
        "targets/frac_any_correct": float((rates[:n_t] > 0).mean()) if n_t else 0.0,
        "conj/n_generated": len(conjs),
        "conj/parse_rate": len(synth) / max(1, len(conjs)),
        "conj/mean_len": float(np.mean([len(c["problem"]) for c in synth])) if synth else 0.0,
        "synth/mean_solve_rate": float(synth_rates.mean()) if synth else 0.0,
        "synth/frac_s0": float((synth_rates == 0).mean()) if synth else 0.0,
        "synth/frac_s1": float((synth_rates == 1).mean()) if synth else 0.0,
        "synth/mean_r_solve": float(r_solve.mean()) if synth else 0.0,
        "synth/mean_r_guide": float(np.mean([c["r_guide"] for c in synth])) if synth else 0.0,
        "synth/mean_r_synth_raw": float(raw.mean()) if len(raw) else 0.0,
        "guide/parse_rate": (float(np.mean([c["guide_scores"] is not None for c in synth]))
                             if synth else 0.0),
    })
    solver_examples = [{"problem": problems[j // k], "completion": sol_texts[j], "correct": bool(v[j])}
                       for j in range(0, len(sol_texts), max(1, len(sol_texts) // 8))]
    return m, {"conjectures": conjs, "solver_examples": solver_examples}


def evaluate(t, targets, solver, n=5):
    """Solver pass@n on all targets, overall and per list length (misc/sort_benchmark.py)."""
    _, texts = solver.generate([solver_messages(x) for x in targets], n=n, step=t)
    v = np.array([parse_sorted(texts[j]) == sorted(targets[j // n]) for j in range(len(texts))])
    passed = v.reshape(len(targets), n).any(axis=1)
    lens = np.array([len(x) for x in targets])
    m = {f"eval/pass@{n}/len_{L}": float(passed[lens == L].mean()) for L in sorted(set(lens.tolist()))}
    m[f"eval/pass@{n}/all"] = float(passed.mean())
    return m


# =============================================================================
# §6 Runner + main
# =============================================================================

@ray.remote(num_cpus=1)
class SGSRunner:
    """Owns the three models and the training loop (verl's managers expect to
    be driven from inside a Ray actor, cf. verl.trainer.main_ppo_v0.TaskRunner)."""

    def run(self, args):
        from tqdm import tqdm
        from transformers import AutoTokenizer
        from verl.single_controller.ray import RayResourcePool

        rng = random.Random(args.seed)
        out_dir = Path(args.out_dir)
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

        targets = load_targets(args.dataset)
        solved = [False] * len(targets)
        print(f"[data] {len(targets)} targets from {args.dataset}", flush=True)

        tok = AutoTokenizer.from_pretrained(args.model)
        # All three models share every GPU (sequential phases + vLLM sleep + FSDP offload).
        pool = RayResourcePool(process_on_nodes=[args.n_gpus], use_gpu=True,
                               max_colocate_count=3, name_prefix="sgs")
        policies = {}
        for tag, resp_len in (("solver", args.solver_response_length),
                              ("conj", args.conj_response_length),
                              ("guide", args.guide_response_length)):
            print(f"[init] {tag}", flush=True)
            policies[tag] = Policy(tag, build_verl_config(args, tag, resp_len), pool, tok)
        solver, conj, guide = policies["solver"], policies["conj"], policies["guide"]

        wandb = None
        if not args.no_wandb:
            import wandb
            wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args),
                       dir=str(out_dir))
            print(f"[wandb] {wandb.run.url}", flush=True)

        def run_eval(t):
            ev = evaluate(t, targets, solver)
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
            metrics, samples = sgs_iteration(t, args, targets, solved, solver, conj, guide, rng)
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
    ray.get(SGSRunner.remote().run.remote(args))


if __name__ == "__main__":
    # Re-import under the package name so Ray pickles our classes by reference
    # (`sgs.sgs_simple.X`) instead of by value from `__main__`.
    from sgs import sgs_simple
    sgs_simple.main()
