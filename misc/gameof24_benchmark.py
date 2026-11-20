"""Pass@5 Game of 24 benchmark via vLLM.

Puzzles come from the Tree of Thoughts CSV (ranked by human solve time, AMT).
They are split into --n-bins equal bins by rank (bin 0 = easiest), and
--n-per-bin puzzles are sampled from each bin (seeded). Draw 5 samples per
puzzle and compute pass@5: at least one sample is an expression that uses
each of the four numbers exactly once with + - * / and evaluates to 24.

Usage:
    python -m misc.gameof24_benchmark
    python -m misc.gameof24_benchmark --model Qwen/Qwen3-8B --n-per-bin 20
"""

import argparse
import ast
import csv
import io
import json
import random
import re
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from urllib.request import urlopen

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from misc.gen_gameof24_data import SOURCE_URL


PROMPT_TEMPLATE = (
    "Use the four numbers {numbers} and the basic arithmetic operations "
    "(+, -, *, /) to obtain 24. Each number must be used exactly once; "
    "parentheses are allowed.\n\n"
    "Return only the expression on a single line, enclosed in \\boxed{{}}, "
    "e.g. \\boxed{{(1 + 2 + 3) * 4}}."
)


def load_bins(n_bins, n_per_bin, seed):
    """Return [(bin_idx, rank_lo, rank_hi, amt_lo, amt_hi, [(numbers, rank, amt), ...]), ...]."""
    with urlopen(SOURCE_URL, timeout=30) as response:
        rows = list(csv.DictReader(io.StringIO(response.read().decode("utf-8"))))
    puzzles = sorted(((int(r["Rank"]), [int(x) for x in r["Puzzles"].split()],
                       float(r["AMT (s)"])) for r in rows))
    rng = random.Random(seed)
    bins = []
    for b in range(n_bins):
        chunk = puzzles[b * len(puzzles) // n_bins:(b + 1) * len(puzzles) // n_bins]
        sample = sorted(rng.sample(chunk, n_per_bin))
        bins.append((b, chunk[0][0], chunk[-1][0], chunk[0][2], chunk[-1][2],
                     [(nums, rank, amt) for rank, nums, amt in sample]))
    return bins


def build_prompt(tok, numbers, enable_thinking=False):
    msg = [{"role": "user", "content": PROMPT_TEMPLATE.format(
        numbers=", ".join(map(str, numbers)))}]
    return tok.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}


def parse_expr(text):
    """First \\boxed{...} after the thinking trace, with LaTeX operators normalized, else None."""
    text = text.rsplit("</think>", 1)[-1]
    m = _BOXED_RE.search(text)
    if not m:
        return None
    s = m.group(1)
    for a, b in (("\\times", "*"), ("\\cdot", "*"), ("\\div", "/"), ("×", "*"), ("÷", "/"),
                 ("\\left", ""), ("\\right", "")):
        s = s.replace(a, b)
    return s.split("=")[0].strip()


def check(expr, numbers):
    """True iff expr uses exactly `numbers` (as a multiset) with + - * / and equals 24."""
    if expr is None:
        return False
    try:
        tree = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return False
    used = []

    def ev(node):
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.Constant) and type(node.value) is int:
            used.append(node.value)
            return Fraction(node.value)
        raise ValueError

    try:
        value = ev(tree)
    except (ValueError, ZeroDivisionError):
        return False
    return Counter(used) == Counter(numbers) and value == 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--n-per-bin", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--out-dir", default="files/results/gameof24")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Rank-binned puzzle sample.
    bins = load_bins(args.n_bins, args.n_per_bin, args.seed)
    for b, rlo, rhi, alo, ahi, _ in bins:
        print(f"[bin {b}] ranks {rlo}-{rhi}  AMT {alo:.2f}-{ahi:.2f}s")

    # 2) Build prompts.
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts, index = [], []
    for b, *_, items in bins:
        for i, (nums, _, _) in enumerate(items):
            prompts.append(build_prompt(tok, nums, enable_thinking=args.thinking))
            index.append((b, i))
    print(f"\n--- example prompt:\n{prompts[0]}\n--- end prompt ---")

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
    bin_ids = [b for b, *_ in bins]
    n_pass = {b: 0 for b in bin_ids}
    n_total = {b: 0 for b in bin_ids}
    per_item = []
    for (b, i), req in zip(index, outs):
        nums, rank, amt = bins[b][5][i]
        completions = [o.text for o in req.outputs]
        parsed = [parse_expr(c) for c in completions]
        oks = [check(p, nums) for p in parsed]
        passed = any(oks)
        n_total[b] += 1
        n_pass[b] += int(passed)
        per_item.append({
            "bin": b,
            "idx": i,
            "numbers": nums,
            "rank": rank,
            "amt": amt,
            "completions": completions,
            "finish_reasons": [o.finish_reason for o in req.outputs],
            "parsed": parsed,
            "oks": oks,
            "pass_at_n": passed,
        })

    pass_at_n = {b: n_pass[b] / n_total[b] for b in bin_ids}
    print(f"\n=== pass@{args.n_samples} (runtime {runtime:.1f}s) ===")
    for b, rlo, rhi, *_ in bins:
        print(f"  bin={b} (ranks {rlo}-{rhi}): {pass_at_n[b]*100:5.1f}%  ({n_pass[b]}/{n_total[b]})")

    print("\n=== example completions ===")
    for b in bin_ids:
        for r in [r for r in per_item if r["bin"] == b][:2]:
            print(f"\n[bin={b} idx={r['idx']}] numbers={r['numbers']} pass@{args.n_samples}={r['pass_at_n']}")
            for s, (c, p, ok) in enumerate(zip(r["completions"][:2], r["parsed"], r["oks"])):
                c_show = c if len(c) <= 400 else c[:400] + f"... [truncated, total {len(c)} chars]"
                print(f"  sample[{s}] ok={ok}\n    completion = {c_show!r}\n    parsed     = {p}")

    # 5) Persist.
    summary = {
        "model": args.model,
        "n_bins": args.n_bins,
        "n_per_bin": args.n_per_bin,
        "bins": {str(b): {"rank_lo": rlo, "rank_hi": rhi, "amt_lo": alo, "amt_hi": ahi}
                 for b, rlo, rhi, alo, ahi, _ in bins},
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "seed": args.seed,
        "pass_at_n": {str(b): pass_at_n[b] for b in bin_ids},
        "n_pass": {str(b): n_pass[b] for b in bin_ids},
        "n_total": {str(b): n_total[b] for b in bin_ids},
        "runtime_sec": runtime,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "per_item.jsonl").open("w") as f:
        for r in per_item:
            f.write(json.dumps(r) + "\n")
    print(f"\n[saved] {out_dir}/summary.json and per_item.jsonl")


if __name__ == "__main__":
    main()
