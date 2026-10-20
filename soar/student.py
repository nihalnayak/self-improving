"""Inner loop: RLOO steps on a teacher-generated dataset X_k.

Follows paper §3.3: `cfg.inner_steps` steps (paper uses 10), each with a batch of
`cfg.student_rollouts` completions (paper uses 8) on one problem sampled from X_k.

The `batch_problems` kwarg generalises the step to `batch_problems` problems per
gradient update (SOAR Table 3 sets batch=8 for PQ eval runs); groups remain
per-problem of size `cfg.student_rollouts`. Default `batch_problems=1` preserves
the single-problem SOAR inner-loop behavior exactly.
"""

import random
from typing import Any, Dict, List, Optional

import torch

from soar.config import SoarConfig
from soar.data import MathProblem
from soar.models import student_chat_prompt
from soar.rloo import rloo_advantages, rloo_loss, tokenize_rollouts
from soar.rollout import sync_adapter_to_disk, vllm_generate
from soar.verify import score_math_answer


# ---- LoRA snapshot/restore (undo inner training per candidate dataset) ----

def snapshot_student_lora(peft_model) -> Dict[str, torch.Tensor]:
    """CPU-independent clone of only the student adapter's parameters."""
    snap = {}
    for n, p in peft_model.named_parameters():
        if "lora_" in n and ".student." in n:
            snap[n] = p.detach().clone()
    return snap


def restore_student_lora(peft_model, snapshot: Dict[str, torch.Tensor]) -> None:
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

    Each step samples `batch_problems` problems and generates `cfg.student_rollouts`
    completions per problem (one RLOO group per problem). Advantages are computed
    within each group; all `batch_problems * student_rollouts` sequences form a
    single policy-gradient batch per step (one gradient update per step).

    - `batch_problems=1` matches the SOAR inner loop (one problem, one gradient step).
    - `inner_steps` overrides `cfg.inner_steps` when provided (useful for callers that
      want to drive the step count externally, e.g. train_pq.py).
    """
    assert len(problems) > 0, "inner_train_student_on_dataset: empty problem list"
    assert batch_problems >= 1
    rng = rng or random.Random()
    n_steps = inner_steps if inner_steps is not None else cfg.inner_steps
    group_size = cfg.student_rollouts
    peft_model.set_adapter("student")
    stats: Dict[str, Any] = {"mean_reward": 0.0}

    for k in range(n_steps):
        if batch_problems <= len(problems):
            batch = rng.sample(problems, batch_problems)
        else:
            batch = [rng.choice(problems) for _ in range(batch_problems)]

        lora_path = sync_adapter_to_disk(peft_model, "student", cfg.out_dir, f"{tag}_k{k}")
        prompt_strs = [student_chat_prompt(tok, p.problem) for p in batch]
        # Flatten so group of `group_size` completions per problem is contiguous.
        gen_prompts: List[str] = []
        for pstr in prompt_strs:
            gen_prompts.extend([pstr] * group_size)
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
        rewards_t = torch.tensor(rewards, dtype=torch.float32,
                                 device=next(peft_model.parameters()).device)

        # Skip the step only if NO group has any within-group reward signal.
        rwv = rewards_t.view(batch_problems, group_size)
        group_ranges = (rwv.max(dim=-1).values - rwv.min(dim=-1).values)
        if (group_ranges < 1e-6).all():
            stats = {"skipped": True, "mean_reward": float(rewards_t.mean().item())}
            continue

        adv = rloo_advantages(rewards_t, group_size=group_size)

        batch_tokens = tokenize_rollouts(
            tok, gen_prompts, completions,
            device=rewards_t.device.type,
            max_prompt_len=cfg.max_prompt_len,
            max_total_len=cfg.max_prompt_len + cfg.student_max_new_tokens,
        )
        batch_tokens.rewards = rewards_t

        peft_model.train()
        out = rloo_loss(peft_model, batch_tokens, advantages=adv, kl_beta=cfg.kl_beta)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for n, p in peft_model.named_parameters()
             if ".student." in n and p.requires_grad], 1.0,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        stats = {k2: (v.item() if isinstance(v, torch.Tensor) else v) for k2, v in out.items()}
        stats["mean_reward"] = float(rewards_t.mean().item())
    return stats
