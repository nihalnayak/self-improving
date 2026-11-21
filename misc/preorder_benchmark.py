"""Pass@5 preorder traversal benchmark via vLLM.

For each node count L in {8, 16, 32, 64, 128}, generate 100 random binary
trees (via misc.gen_preordertraversal_data.gen_sample, seeded). The model gets
the inorder and postorder traversals and must reconstruct the preorder.
Draw 5 samples per tree and compute pass@5: at least one sample exactly
matches the gold preorder list.

Usage:
    python -m misc.preorder_benchmark
    python -m misc.preorder_benchmark --model Qwen/Qwen3-8B --n-trees 20 --lengths 8 16
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

from misc.gen_preordertraversal_data import gen_sample


PROMPT_TEMPLATE = (
    "A binary tree has unique node labels. Given its inorder and postorder "
    "traversals, compute its preorder traversal.\n\n"
    "Inorder: {inorder}\n"
    "Postorder: {postorder}\n\n"
    "Return only the preorder traversal on a single line, enclosed in \\boxed{{}}, "
    "e.g. \\boxed{{[2, 0, 1]}}."
)


def build_prompt(tok, inorder, postorder, enable_thinking=False):
    msg = [{"role": "user", "content": PROMPT_TEMPLATE.format(
        inorder=inorder, postorder=postorder)}]
    return tok.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def parse_preorder(text):
    """Parse the first \\boxed{[...]} after the thinking trace into a list of ints, else None."""
    text = text.rsplit("</think>", 1)[-1]
    m = _BOXED_RE.search(text)
    if not m:
        return None
    inside = m.group(1).strip().strip("[]").strip()
    if not inside:
        return []
    try:
        return [int(x) for x in inside.split(",")]
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--lengths", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--n-trees", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--out-dir", default="files/results/preorder")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Deterministic tree generation.
    rng = random.Random(args.seed)
    data = {L: [gen_sample(L, rng) for _ in range(args.n_trees)] for L in args.lengths}

    # 2) Build prompts.
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts, index = [], []
    for L in args.lengths:
        for i, x in enumerate(data[L]):
            prompts.append(build_prompt(tok, x["inorder"], x["postorder"],
                                        enable_thinking=args.thinking))
            index.append((L, i))

    print("\n=== example prompts (one per length) ===")
    for L in args.lengths:
        j = next(k for k, (LL, _) in enumerate(index) if LL == L)
        print(f"\n--- nodes={L}  gold = {data[L][0]['preorder']}")
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
        x = data[L][i]
        completions = [o.text for o in req.outputs]
        parsed = [parse_preorder(c) for c in completions]
        oks = [p == x["preorder"] for p in parsed]
        passed = any(oks)
        n_total[L] += 1
        n_pass[L] += int(passed)
        per_item.append({
            "length": L,
            "idx": i,
            "inorder": x["inorder"],
            "postorder": x["postorder"],
            "gold": x["preorder"],
            "completions": completions,
            "finish_reasons": [o.finish_reason for o in req.outputs],
            "parsed": parsed,
            "oks": oks,
            "pass_at_n": passed,
        })

    pass_at_n = {L: n_pass[L] / n_total[L] for L in args.lengths}
    print(f"\n=== pass@{args.n_samples} (runtime {runtime:.1f}s) ===")
    for L in args.lengths:
        print(f"  nodes={L:>3}: {pass_at_n[L]*100:5.1f}%  ({n_pass[L]}/{n_total[L]})")

    print("\n=== example completions ===")
    for L in args.lengths:
        for r in [r for r in per_item if r["length"] == L][:2]:
            print(f"\n[nodes={L} idx={r['idx']}] pass@{args.n_samples}={r['pass_at_n']}  gold={r['gold']}")
            for s, (c, p, ok) in enumerate(zip(r["completions"][:2], r["parsed"], r["oks"])):
                c_show = c if len(c) <= 400 else c[:400] + f"... [truncated, total {len(c)} chars]"
                print(f"  sample[{s}] ok={ok}\n    completion = {c_show!r}\n    parsed     = {p}")

    # 5) Persist.
    summary = {
        "model": args.model,
        "lengths": args.lengths,
        "n_trees": args.n_trees,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "seed": args.seed,
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
