"""Self-distillation KL probe on sorting (OPSD-style, no training).

Student context: the sort prompt (misc/sort_benchmark.py).
Teacher context: the same prompt plus the gold sorted list, using the OPSD
reference-solution block and transition text verbatim
(github.com/siyan-zhao/OPSD, data_collator.py).

The student samples completions. Each completion is scored by the same
model under both contexts. We report per-token full-vocab KL on the
completion tokens: forward KL(teacher || student) (OPSD beta=0) and
reverse KL(student || teacher).

Usage:
    python -m misc.sort_opsd_kl
    python -m misc.sort_opsd_kl --lengths 8 --n-lists 2 --n-samples 2
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from misc.sort_benchmark import PROMPT_TEMPLATE, parse_sorted


TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you truly "
    "understand the reasoning behind each step — do not copy or paraphrase "
    "it. Now, using your own words and independent reasoning, derive the same "
    "final answer to the problem above. Think step by step, explore different "
    "approaches, and don't be afraid to backtrack or reconsider if something "
    "doesn't work out:\n"
)


def student_text(xs):
    return PROMPT_TEMPLATE.format(list_str=str(xs))


def teacher_text(xs, gold):
    return (
        student_text(xs) + "\n\n"
        "Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n\\boxed{{{gold}}}\n"
        "=== Reference Solution End ===\n"
        f"{TRANSITION_PROMPT}"
    )


def chat_ids(tok, content, thinking, device):
    text = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
    )
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


@torch.no_grad()
def completion_logprobs(model, prompt_ids, comp_ids):
    """Full-vocab log-probs at each completion position, shape [T, V]."""
    ids = torch.cat([prompt_ids, comp_ids[None]], dim=1)
    logits = model(ids).logits[0, prompt_ids.shape[1] - 1 : -1].float()
    return F.log_softmax(logits, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--lengths", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--n-lists", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=10000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="files/results/sort_opsd_kl")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir) / args.model.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    eos = set(model.generation_config.eos_token_id)

    rows = []
    for L in args.lengths:
        for i in range(args.n_lists):
            xs = [rng.randint(args.lo, args.hi) for _ in range(L)]
            gold = sorted(xs)
            s_ids = chat_ids(tok, student_text(xs), args.thinking, model.device)
            t_ids = chat_ids(tok, teacher_text(xs, gold), args.thinking, model.device)
            gen = model.generate(
                s_ids, do_sample=True, temperature=args.temperature, top_p=1.0,
                top_k=0, max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.n_samples,
            )[:, s_ids.shape[1]:]
            for k, comp in enumerate(gen):
                # keep up to and including the first eos
                stop = next((j for j, t in enumerate(comp.tolist()) if t in eos),
                            len(comp) - 1)
                comp = comp[: stop + 1]
                lp_s = completion_logprobs(model, s_ids, comp)
                lp_t = completion_logprobs(model, t_ids, comp)
                fwd = (lp_t.exp() * (lp_t - lp_s)).sum(-1)  # KL(t || s)
                rev = (lp_s.exp() * (lp_s - lp_t)).sum(-1)  # KL(s || t)
                text = tok.decode(comp, skip_special_tokens=True)
                row = {
                    "length": L, "idx": i, "sample": k, "list": xs, "gold": gold,
                    "completion": text, "correct": parse_sorted(text) == gold,
                    "n_tokens": len(comp),
                    "fwd_kl_mean": fwd.mean().item(),
                    "rev_kl_mean": rev.mean().item(),
                    "fwd_kl_tok": fwd.tolist(), "rev_kl_tok": rev.tolist(),
                }
                rows.append(row)
                print(f"[L={L} i={i} k={k}] correct={row['correct']} "
                      f"T={row['n_tokens']} fwdKL={row['fwd_kl_mean']:.4f} "
                      f"revKL={row['rev_kl_mean']:.4f}")

    print("\n=== mean per-token KL by length / correctness ===")
    for L in args.lengths:
        for c in (True, False):
            sub = [r for r in rows if r["length"] == L and r["correct"] == c]
            if sub:
                f = sum(r["fwd_kl_mean"] for r in sub) / len(sub)
                rv = sum(r["rev_kl_mean"] for r in sub) / len(sub)
                print(f"  L={L:>3} correct={c!s:5} n={len(sub):>3} "
                      f"fwdKL={f:.4f} revKL={rv:.4f}")

    path = out_dir / "kl.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
