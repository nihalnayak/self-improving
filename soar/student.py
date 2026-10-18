"""Inner loop: RLOO steps on a teacher-generated dataset X_k.

Follows paper §3.3: `cfg.inner_steps` steps (paper uses 10), each with a batch of
`cfg.student_rollouts` completions (paper uses 8) on one problem sampled from X_k.
"""

import random
from typing import Any, Dict, List

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
) -> Dict[str, Any]:
    """Run `cfg.inner_steps` RLOO steps. Each step samples ONE problem from `problems` and
    rolls out `cfg.student_rollouts` completions, then takes a single gradient step.
    Returns last-step stats.
    """
    assert len(problems) > 0, "inner_train_student_on_dataset: empty problem list"
    rng = rng or random.Random()
    peft_model.set_adapter("student")
    stats: Dict[str, Any] = {"mean_reward": 0.0}

    for k in range(cfg.inner_steps):
        problem = rng.choice(problems)
        lora_path = sync_adapter_to_disk(peft_model, "student", cfg.out_dir, f"{tag}_k{k}")
        prompt_str = student_chat_prompt(tok, problem.problem)
        completions = vllm_generate(
            vllm_engine, [prompt_str] * cfg.student_rollouts,
            lora_path=lora_path, lora_name="student",
            max_new_tokens=cfg.student_max_new_tokens,
            temperature=cfg.student_temperature,
        )

        rewards = [score_math_answer(c, problem.answer) for c in completions]
        rewards_t = torch.tensor(rewards, dtype=torch.float32,
                                 device=next(peft_model.parameters()).device)
        if (rewards_t.max() - rewards_t.min()).item() < 1e-6:
            # No signal within this group -> skip the gradient step.
            stats = {"skipped": True, "mean_reward": float(rewards_t.mean().item())}
            continue

        batch = tokenize_rollouts(
            tok, [prompt_str] * cfg.student_rollouts, completions,
            device=rewards_t.device.type,
            max_prompt_len=cfg.max_prompt_len,
            max_total_len=cfg.max_prompt_len + cfg.student_max_new_tokens,
        )
        batch.rewards = rewards_t
        adv = rloo_advantages(rewards_t, group_size=cfg.student_rollouts)

        peft_model.train()
        out = rloo_loss(peft_model, batch, advantages=adv, kl_beta=cfg.kl_beta)
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
