"""Inner loop: K GRPO steps on a single candidate problem."""

import copy
from typing import Dict, Any

import torch

from soar.config import SoarConfig
from soar.data import MCQ, format_mcq_prompt
from soar.grpo import grpo_loss, tokenize_rollouts
from soar.models import chat_prompt
from soar.rollout import sync_adapter_to_disk, vllm_generate
from soar.verify import score_answer


# ---- LoRA snapshot/restore (so we can undo inner training per candidate) ----

def snapshot_student_lora(peft_model) -> Dict[str, torch.Tensor]:
    """Return a CPU deepcopy of only the student adapter's parameters."""
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


# ---- the inner training step ----

def inner_train_student(
    peft_model,
    tok,
    vllm_engine,
    cfg: SoarConfig,
    optimizer: torch.optim.Optimizer,
    problem: MCQ,
    tag: str,
) -> Dict[str, Any]:
    """Run K GRPO steps on this single problem. Returns stats from the last step."""
    peft_model.set_adapter("student")
    stats = {}
    for k in range(cfg.inner_steps):
        # 1. Push current student LoRA to disk, generate N completions via vLLM
        lora_path = sync_adapter_to_disk(peft_model, "student", cfg.out_dir, f"{tag}_k{k}")
        user_prompt = format_mcq_prompt(problem)
        prompt_str = chat_prompt(tok, user_prompt)
        completions = vllm_generate(
            vllm_engine, [prompt_str] * cfg.student_rollouts,
            lora_path=lora_path, lora_name="student",
            max_new_tokens=cfg.student_max_new_tokens,
            temperature=cfg.student_temperature,
        )

        # 2. Reward each completion (letter match)
        rewards = [score_answer(c, problem.answer) for c in completions]
        if max(rewards) - min(rewards) < 1e-6:
            # All completions scored the same -> zero advantage, skip the update.
            stats = {"skipped": True, "mean_reward": sum(rewards) / len(rewards)}
            continue

        # 3. Tokenize + GRPO step
        batch = tokenize_rollouts(
            tok, [prompt_str] * cfg.student_rollouts, completions,
            device=next(peft_model.parameters()).device.type,
            max_prompt_len=cfg.max_prompt_len,
            max_total_len=cfg.max_prompt_len + cfg.student_max_new_tokens,
        )
        batch.rewards = torch.tensor(rewards, dtype=torch.float32, device=batch.input_ids.device)

        peft_model.train()
        out = grpo_loss(peft_model, batch, group_size=cfg.student_rollouts, kl_beta=cfg.kl_beta)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for n, p in peft_model.named_parameters() if ".student." in n and p.requires_grad], 1.0
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        stats = {k2: (v.item() if isinstance(v, torch.Tensor) else v) for k2, v in out.items()}
        stats["mean_reward"] = sum(rewards) / len(rewards)
    return stats
