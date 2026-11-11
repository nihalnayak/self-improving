"""Pass@5 sorting benchmark for Qwen3-4B via vLLM.

For each list length L in {4, 8, 16, 32}, generate 100 random integer lists in
[0, 10000] (seeded for determinism), ask the model to sort each one, draw 5
samples per list in a single vLLM call, parse the sorted list from each
completion, and compute pass@5 (at least one of 5 samples exactly matches
`sorted(xs)`).

Usage:
    python -m misc.sort_benchmark
    python -m misc.sort_benchmark --n-lists 20 --lengths 4 8   # smoke
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


PROMPT_TEMPLATE = (
    "Sort the numbers in the list in ascending order.\n\n"
    "List: {list_str}\n\n"
    "Return only the sorted list on a single line, enclosed in \\boxed{{}}, "
    "e.g. \\boxed{{[1, 2, 3]}}."
)


def gen_lists(lengths, n_per_length, lo, hi, seed):
    rng = random.Random(seed)
    return {
        L: [[rng.randint(lo, hi) for _ in range(L)] for _ in range(n_per_length)]
        for L in lengths
    }


def build_prompt(tok, nums, enable_thinking=False):
    msg = [{"role": "user", "content": PROMPT_TEMPLATE.format(list_str=str(nums))}]
    return tok.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


_BOXED_RE = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)
_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_sorted(text):
    """Pull a list of ints from the model output.

    Prefer the first \\boxed{...}; fall back to the last [...] block.
    Returns None if nothing parses to ints.
    """
    m = _BOXED_RE.search(text)
    if m:
        inside = m.group(1)
    else:
        brackets = _BRACKET_RE.findall(text)
        if not brackets:
            return None
        inside = brackets[-1]
    nums = _NUM_RE.findall(inside)
    if not nums:
        return None
    try:
        return [float(n) for n in nums]
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--lengths", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--n-lists", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=10000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking", action="store_true",
                    help="Enable thinking mode")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--out-dir", default="files/results/sorting")
    args = ap.parse_args()

    # if args.thinking:
    #     args.max_tokens = 2048

    out_dir = Path(args.out_dir) / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Deterministic list generation.
    data = gen_lists(args.lengths, args.n_lists, args.lo, args.hi, args.seed)
    gold = {L: [sorted(xs) for xs in lists] for L, lists in data.items()}

    # 2) Build prompts (flattened, keep index mapping back to (L, i)).
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts, index = [], []
    for L in args.lengths:
        for i, xs in enumerate(data[L]):
            prompts.append(build_prompt(tok, xs, enable_thinking=args.thinking))
            index.append((L, i))

    # Show one fully-formatted prompt per length so we can eyeball the input.
    print("\n=== example prompts (one per length) ===")
    for L in args.lengths:
        # First prompt at this length is the first time we see L in `index`.
        j = next(k for k, (LL, _) in enumerate(index) if LL == L)
        print(f"\n--- len={L}  input list = {data[L][0]}")
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
        truth = gold[L][i]
        completions = [o.text for o in req.outputs]
        parsed = [parse_sorted(c) for c in completions]
        oks = [p == truth for p in parsed]
        passed = any(oks)
        n_total[L] += 1
        n_pass[L] += int(passed)
        per_item.append({
            "length": L,
            "idx": i,
            "input": data[L][i],
            "gold": truth,
            "completions": completions,
            "parsed": parsed,
            "oks": oks,
            "pass_at_n": passed,
        })

    pass_at_5 = {L: n_pass[L] / n_total[L] for L in args.lengths}

    print(f"\n=== pass@{args.n_samples} (runtime {runtime:.1f}s) ===")
    for L in args.lengths:
        print(f"  len={L:>3}: {pass_at_5[L]*100:5.1f}%  ({n_pass[L]}/{n_total[L]})")

    # Show a few input/completion/parse examples per length for sanity-checking.
    n_show_items = 2            # items per length
    n_show_samples = 2          # samples per item (of args.n_samples)
    print("\n=== example completions ===")
    by_length = {L: [r for r in per_item if r["length"] == L] for L in args.lengths}
    for L in args.lengths:
        print(f"\n--- len={L} ---")
        for r in by_length[L][:n_show_items]:
            print(f"\n[len={L} idx={r['idx']}] pass@{args.n_samples}={r['pass_at_n']}")
            print(f"  input      = {r['input']}")
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
        "n_lists": args.n_lists,
        "n_samples": args.n_samples,
        "lo": args.lo,
        "hi": args.hi,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "seed": args.seed,
        "pass_at_n": {str(L): pass_at_5[L] for L in args.lengths},
        "n_pass": {str(L): n_pass[L] for L in args.lengths},
        "n_total": {str(L): n_total[L] for L in args.lengths},
        "runtime_sec": runtime,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "per_item.jsonl", "w") as f:
        for r in per_item:
            f.write(json.dumps(r) + "\n")
    print(f"\n[saved] {out_dir}/summary.json  and per_item.jsonl")

if __name__ == "__main__":
    main()
