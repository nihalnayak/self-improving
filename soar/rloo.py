"""RLOO loss (Ahmadian et al., 2024) used by BOTH student (inner) and teacher (outer) loops.

Difference from GRPO: no std-normalization. Per-sequence advantage is leave-one-out:
    adv[i] = reward[i] - mean(reward[j] for j != i in same group)
           = (G * r[i] - sum_group(r)) / (G - 1)

For the student (inner loop): group = `cfg.student_group_size` completions on one problem.
For the teacher (outer loop): advantages are computed at the **dataset** level over `g`
datasets (one scalar per X_k) and broadcast to each X_k's `n` constituent rollouts.

To accommodate both, `rloo_loss` takes **precomputed per-sequence advantages** (not raw
rewards). Call `rloo_advantages(rewards, group_size)` to get the standard RLOO scaling.
"""

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class RolloutBatch:
    """Tokenized rollouts.

    Shapes:
      input_ids:       [B, L]
      attention_mask:  [B, L]
      completion_mask: [B, L-1]  (1 where position's target token belongs to the completion)
      rewards:         [B]       (raw per-sequence rewards; optional metadata)
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

    L = max(len(s) for s in seqs)
    pad_id = tok.pad_token_id
    input_ids, attention_mask, completion_mask = [], [], []
    for s, pl in zip(seqs, prompt_lens):
        pad = L - len(s)
        row = [pad_id] * pad + s
        amask = [0] * pad + [1] * len(s)
        shifted = [0] * (L - 1)
        for t in range(L - 1):
            orig_pos = t + 1 - pad
            if orig_pos >= pl and orig_pos < len(s):
                shifted[t] = 1
        input_ids.append(row)
        attention_mask.append(amask)
        completion_mask.append(shifted)

    return RolloutBatch(
        input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
        attention_mask=torch.tensor(attention_mask, dtype=torch.long, device=device),
        completion_mask=torch.tensor(completion_mask, dtype=torch.float32, device=device),
        rewards=torch.empty(0, device=device),
    )


def rloo_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Leave-one-out advantages. Input shape [B] with B divisible by group_size.

    adv[i] = (G * r[i] - sum_group(r)) / (G - 1)
    """
    assert rewards.ndim == 1 and rewards.numel() % group_size == 0
    if group_size < 2:
        raise ValueError("RLOO requires group_size >= 2")
    r = rewards.view(-1, group_size)
    s = r.sum(dim=-1, keepdim=True)
    adv = (group_size * r - s) / (group_size - 1)
    return adv.view(-1)


def _per_token_logp(model, input_ids, attention_mask) -> torch.Tensor:
    """Per-token log-prob of the actual next token at each position: shape [B, L-1]."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    logp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return logp


def rloo_loss(peft_model, batch: RolloutBatch, advantages: torch.Tensor,
              kl_beta: float) -> dict:
    """RLOO policy-gradient loss + KL to the adapter-disabled policy.

    `advantages` is [B] precomputed per sequence (so the caller controls how groups are
    formed). KL is a free reference: logp under the base model (adapter off).
    """
    assert advantages.ndim == 1 and advantages.numel() == batch.input_ids.shape[0]

    logp_new = _per_token_logp(peft_model, batch.input_ids, batch.attention_mask)   # [B, L-1]
    with torch.no_grad():
        with peft_model.disable_adapter():
            logp_ref = _per_token_logp(peft_model, batch.input_ids, batch.attention_mask)

    m = batch.completion_mask
    denom = m.sum(dim=-1).clamp(min=1.0)
    mean_logp_new = (logp_new * m).sum(dim=-1) / denom                              # [B]
    mean_logp_ref = (logp_ref * m).sum(dim=-1) / denom                              # [B]
    kl = (mean_logp_new - mean_logp_ref)                                            # [B]

    pg_loss = -(advantages * mean_logp_new).mean()
    kl_term = kl.mean()
    loss = pg_loss + kl_beta * kl_term

    return {
        "loss": loss,
        "pg_loss": pg_loss.detach(),
        "kl": kl_term.detach(),
        "adv_abs_mean": advantages.abs().mean().detach(),
    }


if __name__ == "__main__":
    # Sanity checks on rloo_advantages.
    r = torch.tensor([1.0, 0.0, 0.0, 0.0])
    a = rloo_advantages(r, group_size=4)
    assert torch.allclose(a, torch.tensor([1.0, -1/3, -1/3, -1/3])), a
    assert torch.allclose(a.sum(), torch.tensor(0.0), atol=1e-6)

    # Two groups
    r = torch.tensor([1.0, 0.0, 0.0, 2.0, 0.0, 1.0])
    a = rloo_advantages(r, group_size=3)
    # Group 1: [1, 0, 0] -> adv_i = (3*r_i - 1) / 2 = [1.0, -0.5, -0.5]
    # Group 2: [2, 0, 1] -> sum=3, adv_i = (3*r_i - 3) / 2 = [1.5, -1.5, 0.0]
    assert torch.allclose(a, torch.tensor([1.0, -0.5, -0.5, 1.5, -1.5, 0.0])), a

    # Each group's advantages sum to zero.
    for i in range(0, 6, 3):
        assert abs(a[i:i+3].sum().item()) < 1e-6

    # Group size < 2 should error
    try:
        rloo_advantages(torch.tensor([1.0, 2.0]), group_size=1)
        assert False, "expected ValueError"
    except ValueError:
        pass

    print("rloo.py: all sanity checks passed")
