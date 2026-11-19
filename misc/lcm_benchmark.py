"""Pass@5 least common multiple benchmark for Qwen3-4B via vLLM.

For each operand digit length in {2, 4, 6, 8, 10}, generate 100 random
integer pairs (seeded for determinism). Both operands have exactly that
many digits, with no leading zeros. Draw 5 samples per pair and compute
pass@5: at least one sample exactly matches lcm(number1, number2).
Use --exclude-coprime to resample pairs with GCD 1 while keeping pair counts.

Usage:
    python -m misc.lcm_benchmark
    python -m misc.lcm_benchmark --exclude-coprime
    python -m misc.lcm_benchmark --n-pairs 3 --lengths 2 4 6 8 10 --out-dir files/results/lcm/no-thinking
    python -m misc.lcm_benchmark --n-pairs 2 --lengths 2 4 6 8 10 --max-tokens 4096 --thinking --out-dir files/results/lcm/thinking4096
    python -m misc.lcm_benchmark --n-pairs 2 --lengths 2 4 6 8 10 --max-tokens 8192 --thinking --out-dir files/results/lcm/thinking8192
    python -m misc.lcm_benchmark --n-pairs 2 --lengths 2 4 6 8 10 --max-tokens 16384 --thinking --out-dir files/results/lcm/thinking16384
"""

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


PROMPT_TEMPLATE = (
    "Compute the least common multiple (LCM) of the following two integers.\n\n"
    "number1: {number1}\n"
    "number2: {number2}\n\n"
    "Return only the LCM on a single line, enclosed in \\boxed{{}}, "
    "e.g. \\boxed{{1234}}."
)

# Generate the set of number pairs
def gen_pairs(lengths, n_per_length, seed, exclude_coprime=False):
    rng = random.Random(seed)
    data = {}
    for L in lengths:
        lo, hi = 10 ** (L - 1), 10 ** L - 1
        pairs = []
        while len(pairs) < n_per_length:
            number1, number2 = rng.randint(lo, hi), rng.randint(lo, hi)
            if exclude_coprime and math.gcd(number1, number2) == 1:
                continue
            pairs.append((number1, number2))
        data[L] = pairs
    return data

# Generate the LCM prompt
def build_prompt(tok, number1, number2, enable_thinking=False):
    msg = [{"role": "user", "content": PROMPT_TEMPLATE.format(
        number1=number1, number2=number2)}]
    return tok.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_STANDALONE_INTEGER_RE = re.compile(r"(?<![\w.+-])[0-9]+(?!\w|\.[0-9])")
_INTEGER_RE = re.compile(r"[0-9]+")


def parse_lcm(text):
    """Parse the least common multiple from the model output.

    Prefer the first \\boxed{...}; fall back to the last complete integer. 
    Returns None if nothing parses to an integer.
    """

    # Only consider text after the thinking trace
    text = text.rsplit("</think>", 1)[-1].strip()

    # Get the first boxed answer
    m = _BOXED_RE.search(text)
    if m:
        inside = m.group(1).strip()
    else:
        integers = _STANDALONE_INTEGER_RE.findall(text)
        if not integers:
            return None
        inside = integers[-1]

    if not _INTEGER_RE.fullmatch(inside):
        return None
    try:
        return int(inside)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--lengths", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-coprime", action="store_true",
                    help="resample pairs with GCD 1 to keep only GCDs greater than 1")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--out-dir", default="files/results/lcm")
    args = ap.parse_args()

    # Verify arguments
    if any(L < 1 for L in args.lengths) or len(set(args.lengths)) != len(args.lengths):
        ap.error("--lengths must contain distinct positive digit counts")
    if args.n_pairs < 1 or args.n_samples < 1:
        ap.error("--n-pairs and --n-samples must be positive")

    out_dir = Path(args.out_dir) / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Deterministic integer pair generation.
    data = gen_pairs(args.lengths, args.n_pairs, args.seed,
                     exclude_coprime=args.exclude_coprime)
    gold = {L: [math.lcm(number1, number2) for number1, number2 in pairs] for L, pairs in data.items()}

    # 2) Build prompts (flattened, keep index mapping back to (L, i)).
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts, index = [], []
    for L in args.lengths:
        for i, (number1, number2) in enumerate(data[L]):
            prompts.append(build_prompt(tok, number1, number2, enable_thinking=args.thinking))
            index.append((L, i))

    # Show one fully-formatted prompt per length so we can eyeball the input.
    print("\n=== example prompts (one per length) ===")
    for L in args.lengths:
        # First prompt at this length is the first time we see L in `index`.
        j = next(k for k, (LL, _) in enumerate(index) if LL == L)
        number1, number2 = data[L][0]
        print(f"\n--- digits={L}  input = lcm({number1}, {number2})")
        print(f"--- gold          = {gold[L][0]}")
        print(f"--- prompt text:\n{prompts[j]}")
        print("--- end prompt ---")

    # 3) Load vLLM.
    n_gpus = max(1, torch.cuda.device_count())
    print(f"[vllm] loading {args.model} (tensor_parallel_size={n_gpus})")
    engine = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=n_gpus,
        enforce_eager=False,
        disable_log_stats=True,
        enable_prefix_caching=True,
    )

    sp = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    print(f"[gen] {len(prompts)} prompts x n={args.n_samples}")
    t0 = time.time()
    outs = engine.generate(prompts, sp)
    runtime = time.time() - t0

    # 4) Parse + score.
    n_pass = {L: 0 for L in args.lengths}
    n_total = {L: 0 for L in args.lengths}
    per_item = []
    for (L, i), req in zip(index, outs):
        number1, number2 = data[L][i] # NOTE
        truth = gold[L][i]
        completions = [o.text for o in req.outputs]
        parsed = [parse_lcm(c) for c in completions]
        oks = [p == truth for p in parsed]
        passed = any(oks)
        n_total[L] += 1
        n_pass[L] += int(passed)
        per_item.append({
            "length": L,
            "idx": i,
            "number1": number1,
            "number2": number2,
            "gold": truth,
            "completions": completions,
            "parsed": parsed,
            "oks": oks,
            "pass_at_n": passed,
        })

    pass_at_n = {L: n_pass[L] / n_total[L] for L in args.lengths}

    print(f"\n=== pass@{args.n_samples} (runtime {runtime:.1f}s) ===")
    for L in args.lengths:
        print(f"  digits={L:>3}: {pass_at_n[L]*100:5.1f}%  ({n_pass[L]}/{n_total[L]})")

    # Show a few input/completion/parse examples per length for sanity-checking.
    n_show_items = 2            # items per length
    n_show_samples = 2          # samples per item (of args.n_samples)
    print("\n=== example completions ===")
    by_length = {L: [r for r in per_item if r["length"] == L] for L in args.lengths}
    for L in args.lengths:
        print(f"\n--- digits={L} ---")
        for r in by_length[L][:n_show_items]:
            print(f"\n[digits={L} idx={r['idx']}] pass@{args.n_samples}={r['pass_at_n']}")
            print(f"  input      = lcm({r['number1']}, {r['number2']})")
            print(f"  gold       = {r['gold']}")
            for s, (c, p, ok) in enumerate(zip(r["completions"], r["parsed"], r["oks"])):
                if s >= n_show_samples:
                    break
                c_show = c if len(c) <= 400 else c[:400] + f"... [truncated, total {len(c)} chars]"
                print(f"  sample[{s}] ok={ok}")
                print(f"    completion = {c_show!r}")
                print(f"    parsed     = {p}")

    # 5) Persist.
    summary = {
        "model": args.model,
        "lengths": args.lengths,
        "n_pairs": args.n_pairs,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "seed": args.seed,
        "exclude_coprime": args.exclude_coprime,
        "pass_at_n": {str(L): pass_at_n[L] for L in args.lengths},
        "n_pass": {str(L): n_pass[L] for L in args.lengths},
        "n_total": {str(L): n_total[L] for L in args.lengths},
        "runtime_sec": runtime,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "per_item.jsonl").open("w") as f:
        for r in per_item:
            f.write(json.dumps(r) + "\n")
    print(f"\n[saved] {out_dir}/summary.json and per_item.jsonl")


if __name__ == "__main__":
    main()
