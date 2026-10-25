"""Inner loop: RLOO steps on a teacher-generated dataset X_k.

Follows paper §3.3: `cfg.inner_steps` steps (paper uses 10), each with a batch of
`cfg.student_group_size` completions (paper uses 32) on `batch_problems` problems
sampled from X_k (paper's inner-loop batch is 8 problems per gradient step).

Groups are per-problem of size `cfg.student_group_size`. `batch_problems` is a
runtime kwarg so train_pq.py can drive it from `cfg.pq_batch_problems`.
"""

import random
from typing import Any, Dict, List, Optional

import torch

from soar import dist_utils
from soar.config import SoarConfig
from soar.data import MathProblem
from soar.models import student_chat_prompt
from soar.rloo import rloo_advantages, rloo_loss, tokenize_rollouts
from soar.rollout import sync_adapter_to_disk, vllm_generate
from soar.verify import score_math_answer


# ---- LoRA snapshot/restore (undo inner training per candidate dataset) ----

def snapshot_student_lora(peft_model) -> Dict[str, torch.Tensor]:
    """Clone of only the student adapter's parameters.

    Under FSDP: all ranks enter `summon_full_params` together to all-gather the
    student LoRA weights (small — a few MB), then each rank keeps an identical
    full-shape snapshot. Keeping snapshots on all ranks avoids rank-conditional
    code in the caller and matches `restore_student_lora`'s all-rank writeback.
    """
    snap: Dict[str, torch.Tensor] = {}
    with dist_utils.summon_full_params(peft_model, writeback=False, rank0_only=False):
        for n, p in peft_model.named_parameters():
            if "lora_" in n and ".student." in n:
                snap[n] = p.detach().clone()
    return snap


def restore_student_lora(peft_model, snapshot: Dict[str, torch.Tensor]) -> None:
    """Overwrite student LoRA params from a snapshot.

    Under FSDP: all ranks summon full params with writeback=True, copy into the
    full tensors, FSDP re-shards on context exit.
    """
    with dist_utils.summon_full_params(peft_model, writeback=True, rank0_only=False):
        state = dict(peft_model.named_parameters())
        with torch.no_grad():
            for n, v in snapshot.items():
                state[n].copy_(v)


# ---- inner loop over a dataset X_k ----

def inner_train_student_on_dataset(
    peft_model,
    tok,
    vllm_engine,
    cfg: SoarConfig,
    optimizer: torch.optim.Optimizer,
    problems: List[MathProblem],
    tag: str,
    rng: random.Random | None = None,
    *,
    batch_problems: int = 1,
    inner_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run RLOO steps on `problems`.

    Each step samples `batch_problems` problems and generates `cfg.student_group_size`
    completions per problem (one RLOO group per problem). Advantages are computed
    within each group; all `batch_problems * student_group_size` sequences form a
    single policy-gradient batch per step (one gradient update per step).

    - `inner_steps` overrides `cfg.inner_steps` when provided (useful for callers that
      want to drive the step count externally, e.g. train_pq.py).
    """
    assert len(problems) > 0, "inner_train_student_on_dataset: empty problem list"
    assert batch_problems >= 1
    rng = rng or random.Random()
    n_steps = inner_steps if inner_steps is not None else cfg.inner_steps
    group_size = cfg.student_group_size
    peft_model.set_adapter("student")
    stats: Dict[str, Any] = {"mean_reward": 0.0}
    n_rollouts_total = 0
    n_correct_total = 0
    reward_sum_total = 0.0

    for k in range(n_steps):
        # Rank 0 drives problem selection + vLLM generation + reward scoring;
        # results are broadcast so every rank sees the same batch (RLOO groups
        # must agree across ranks for correct advantage computation).
        if dist_utils.is_main():
            if batch_problems <= len(problems):
                batch = rng.sample(problems, batch_problems)
            else:
                batch = [rng.choice(problems) for _ in range(batch_problems)]
        else:
            batch = None
        batch = dist_utils.broadcast_object(batch, src=0)

        lora_path = sync_adapter_to_disk(peft_model, "student", cfg.out_dir, f"{tag}_k{k}")
        prompt_strs = [student_chat_prompt(tok, p.problem) for p in batch]
        # Flatten so group of `group_size` completions per problem is contiguous.
        gen_prompts: List[str] = []
        for pstr in prompt_strs:
            gen_prompts.extend([pstr] * group_size)

        if dist_utils.is_main():
            completions = vllm_generate(
                vllm_engine, gen_prompts,
                lora_path=lora_path, lora_name="student",
                max_new_tokens=cfg.student_max_new_tokens,
                temperature=cfg.student_temperature,
            )
            rewards: List[float] = []
            for pi, p in enumerate(batch):
                for ci in range(group_size):
                    rewards.append(score_math_answer(completions[pi * group_size + ci], p.answer))
            payload = (completions, rewards)
        else:
            payload = None
        completions, rewards = dist_utils.broadcast_object(payload, src=0)

        n_rollouts_total += len(rewards)
        n_correct_total += sum(1 for r in rewards if r >= 120.0)
        reward_sum_total += float(sum(rewards))

        dev = next(peft_model.parameters()).device
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=dev)

        # Skip the step only if NO group has any within-group reward signal.
        rwv = rewards_t.view(batch_problems, group_size)
        group_ranges = (rwv.max(dim=-1).values - rwv.min(dim=-1).values)
        if (group_ranges < 1e-6).all():
            stats = {"skipped": True, "mean_reward": float(rewards_t.mean().item())}
            continue

        adv = rloo_advantages(rewards_t, group_size=group_size)

        batch_tokens = tokenize_rollouts(
            tok, gen_prompts, completions,
            device=dev.type,
            max_prompt_len=cfg.max_prompt_len,
            max_total_len=cfg.max_prompt_len + cfg.student_max_new_tokens,
        )
        batch_tokens.rewards = rewards_t

        # Split the batch across ranks for data-parallel forward/backward.
        # Advantages are computed from the full group above; each rank only runs
        # loss on its slice. FSDP's grad reduce-scatter averages across ranks, so
        # local `.mean()` + grad-average = global mean over all B sequences.
        total = batch_tokens.input_ids.shape[0]
        local_idx = dist_utils.shard_indices(total)
        local_idx_t = torch.tensor(local_idx, dtype=torch.long, device=dev)
        local_batch = type(batch_tokens)(
            input_ids=batch_tokens.input_ids.index_select(0, local_idx_t),
            attention_mask=batch_tokens.attention_mask.index_select(0, local_idx_t),
            completion_mask=batch_tokens.completion_mask.index_select(0, local_idx_t),
            rewards=batch_tokens.rewards.index_select(0, local_idx_t),
        )
        local_adv = adv.index_select(0, local_idx_t)

        peft_model.train()
        out = rloo_loss(peft_model, local_batch, advantages=local_adv, kl_beta=cfg.kl_beta)
        out["loss"].backward()
        if dist_utils.is_distributed():
            # FSDP method handles sharded grads; only student grads are populated here
            # (teacher adapter grads are None), so clipping-all is equivalent to
            # clipping-student.
            peft_model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(
                [p for n, p in peft_model.named_parameters()
                 if ".student." in n and p.requires_grad], 1.0,
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        stats = {k2: (v.item() if isinstance(v, torch.Tensor) else v) for k2, v in out.items()}
        stats["mean_reward"] = float(rewards_t.mean().item())
    stats["n_rollouts"] = n_rollouts_total
    stats["n_correct"] = n_correct_total
    stats["mean_reward_run"] = reward_sum_total / max(n_rollouts_total, 1)
    return stats
