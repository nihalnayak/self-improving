"""Single GRPO loss used by BOTH teacher and student loops.

Steps (vanilla GRPO + KL to base):
  1. Within-group z-score of rewards -> advantages (one scalar per sequence).
  2. logp_new = per-token log prob of each completion under current policy (adapter ON).
  3. logp_ref = same, but with adapters disabled -> base model (gives us a free KL ref).
  4. loss = -(mean_token(logp_new) * advantage) + kl_beta * mean_token(logp_new - logp_ref)
"""

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class RolloutBatch:
    """Tokenized rollouts ready for GRPO.

    Shapes:
      input_ids:       [B, L]  — full sequence (prompt tokens + completion tokens, left-padded)
      attention_mask:  [B, L]
      completion_mask: [B, L-1] — 1 for positions whose logp belongs to the completion
      rewards:         [B]
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    rewards: torch.Tensor

def tokenize_rollouts(tok, prompts: List[str], completions: List[str], device: str,
                      max_prompt_len: int, max_total_len: int) -> RolloutBatch:
    """Encode (prompt, completion) pairs with left-padding and compute completion_mask."""
    assert len(prompts) == len(completions)
    seqs, prompt_lens = [], []
    for p, c in zip(prompts, completions):
        p_ids = tok(p, add_special_tokens=False).input_ids[-max_prompt_len:]
        c_ids = tok(c, add_special_tokens=False).input_ids
        total = (p_ids + c_ids)[:max_total_len]
        seqs.append(total)
        prompt_lens.append(min(len(p_ids), max_total_len))

    # Left-pad to common length.
    L = max(len(s) for s in seqs)
    pad_id = tok.pad_token_id
    input_ids, attention_mask, completion_mask = [], [], []
    for s, pl in zip(seqs, prompt_lens):
        pad = L - len(s)
        row = [pad_id] * pad + s
        amask = [0] * pad + [1] * len(s)
        # completion mask on positions whose target token is a completion token.
        # per-token logp at index t predicts token at index t+1 of the sequence.
        # So completion_mask[t] = 1 iff position t+1 is a completion token.
        shifted = [0] * (L - 1)
        for t in range(L - 1):
            orig_pos = t + 1 - pad  # position in the un-padded sequence that t+1 refers to
            if orig_pos >= pl and orig_pos < len(s):
                shifted[t] = 1
        input_ids.append(row)
        attention_mask.append(amask)
        completion_mask.append(shifted)

    return RolloutBatch(
        input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
        attention_mask=torch.tensor(attention_mask, dtype=torch.long, device=device),
        completion_mask=torch.tensor(completion_mask, dtype=torch.float32, device=device),
        rewards=torch.empty(0, device=device),  # rewards set externally
    )


def _per_token_logp(model, input_ids, attention_mask) -> torch.Tensor:
    """Return per-token log-prob of the actual next token at each position: shape [B, L-1]."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :].float()           # [B, L-1, V]
    targets = input_ids[:, 1:]                       # [B, L-1]
    logp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return logp


def grpo_loss(peft_model, batch: RolloutBatch, group_size: int, kl_beta: float) -> dict:
    """Compute GRPO loss. Returns dict with 'loss', 'adv_mean', 'kl', 'reward_mean'."""
    rewards = batch.rewards
    assert rewards.numel() > 0, "set batch.rewards before calling grpo_loss"
    assert rewards.numel() % group_size == 0, "batch size must be divisible by group_size"

    # 1. advantages (within-group z-score)
    r = rewards.view(-1, group_size)
    adv = (r - r.mean(dim=-1, keepdim=True)) / (r.std(dim=-1, keepdim=True) + 1e-6)
    adv = adv.view(-1)  # [B]

    # 2. logp under current adapters
    logp_new = _per_token_logp(peft_model, batch.input_ids, batch.attention_mask)   # [B, L-1]

    # 3. logp under base (adapters off) — free KL ref
    with torch.no_grad():
        with peft_model.disable_adapter():
            logp_ref = _per_token_logp(peft_model, batch.input_ids, batch.attention_mask)

    # Masked per-sequence stats
    m = batch.completion_mask                                        # [B, L-1]
    denom = m.sum(dim=-1).clamp(min=1.0)
    mean_logp_new = (logp_new * m).sum(dim=-1) / denom               # [B]
    mean_logp_ref = (logp_ref * m).sum(dim=-1) / denom               # [B]
    kl = (mean_logp_new - mean_logp_ref)                             # [B]

    pg_loss = -(adv * mean_logp_new).mean()
    kl_term = kl.mean()
    loss = pg_loss + kl_beta * kl_term

    return {
        "loss": loss,
        "pg_loss": pg_loss.detach(),
        "kl": kl_term.detach(),
        "reward_mean": rewards.mean().detach(),
        "reward_std": rewards.std().detach() if rewards.numel() > 1 else torch.tensor(0.0),
        "adv_abs_mean": adv.abs().mean().detach(),
    }
