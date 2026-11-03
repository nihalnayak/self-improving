"""SGS-style sorting-problem proposer via vLLM.

Prompts Qwen3-4B (as a Conjecturer, cf. SGS §E.1, arxiv 2604.20209v1) to
propose a *simpler* sorting problem (a list of fewer than L ints in
[lo, hi]) as a stepping stone toward a target L-element sorting task, and
its solution. The final problem is emitted in the exact benchmark format
(see misc/sort_benchmark.py) followed by the solution in \\boxed{...}.

Draws 10 completions in a single vLLM call (SamplingParams(n=10)) and
reports how many were parsable, had a shorter-than-L length, and had a
boxed solution matching sorted(problem).

Usage:
    python -m misc.sort_proposer
    python -m misc.sort_proposer --thinking
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# Adapted from SGS §E.1 Conjecturer Prompt (Lean4 lemma generator): propose
# an easier sub-problem (shorter list) as a stepping stone toward the target
# L-element task.
PROMPT_TEMPLATE = (
    "I am learning to sort lists of {L} integers in [{lo}, {hi}] in "
    "ascending order. This is the task I ultimately want to solve.\n\n"
    "To make progress, I want to first practice on a simpler version: a "
    "shorter sorting problem with fewer than {L} integers (but at least 12) "
    "in [{lo}, {hi}]. Learning to solve this simpler task will be a "
    "stepping stone toward solving the more challenging {L}-element task.\n\n"
    "Please generate one such simpler sorting problem and its solution. "
    "The list should NOT already be sorted, and should NOT be an obvious "
    "canonical example.\n\n"
    "You may think step-by-step before producing the final output.\n\n"
    "Output the final problem and solution in exactly the following format, "
    "with nothing after the solution:\n\n"
    "<problem> Sort the numbers in the list in ascending order.\n\n"
    "List: [a, b, c, ...]\n\n</problem>\n\n"
    "Return only the sorted list on a single line, enclosed in \\boxed{{}}, "
    "e.g. \\boxed{{[1, 2, 3]}}.\n\n"
    "\\boxed{{[sorted, ...]}}\n\n"
    "All thinking should go before this block so the final output is cleanly separated.\n\n"
)


def build_prompt(tok, L, lo, hi, enable_thinking=False):
    msg = [{"role": "user", "content": PROMPT_TEMPLATE.format(L=L, lo=lo, hi=hi)}]
    return tok.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


_PROBLEM_RE = re.compile(r"<problem>(.*?)</problem>", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)
_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
_NUM_RE = re.compile(r"-?\d+")


def extract_problem(text):
    """Pull the list of ints from the last `<problem>...</problem>` block."""
    blocks = [b for b in _PROBLEM_RE.findall(text) if b.strip()]
    if not blocks:
        return None
    brackets = _BRACKET_RE.findall(blocks[-1])
    if not brackets:
        return None
    nums = _NUM_RE.findall(brackets[-1])
    if not nums:
        return None
    try:
        return [int(n) for n in nums]
    except ValueError:
        return None


def extract_solution(text):
    """Pull the sorted list from the last non-empty \\boxed{...}.

    Skips the `\\boxed{}` and example `\\boxed{[1, 2, 3]}` that appear in
    the prompt-echoed format block; falls back to the last [...] block.
    """
    boxed = [b for b in _BOXED_RE.findall(text) if b.strip()]
    if boxed:
        inside = boxed[-1]
    else:
        brackets = _BRACKET_RE.findall(text)
        if not brackets:
            return None
        inside = brackets[-1]
    nums = _NUM_RE.findall(inside)
    if not nums:
        return None
    try:
        return [int(n) for n in nums]
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--list-length", type=int, default=32)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=10000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking", action="store_true",
                    help="Enable thinking mode; caps max_tokens at 1024.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--out-dir", default="files/results/sorting/proposer")
    args = ap.parse_args()

    # if args.thinking:
    #     args.max_tokens = 1024

    out_dir = Path(args.out_dir) / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    prompt = build_prompt(
        tok, args.list_length, args.lo, args.hi, enable_thinking=args.thinking
    )

    print("=== proposer prompt ===")
    print(prompt)
    print("=== end prompt ===\n")

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

    print(f"[gen] 1 prompt x n={args.n_samples}")
    t0 = time.time()
    outs = engine.generate([prompt], sp)
    runtime = time.time() - t0

    completions = [o.text for o in outs[0].outputs]

    samples = []
    for i, c in enumerate(completions):
        problem = extract_problem(c)
        solution = extract_solution(c)
        problem_ok = problem is not None
        boxed_ok = solution is not None
        both_ok = problem_ok and boxed_ok
        len_ok = problem_ok and 2 <= len(problem) < args.list_length
        correct = both_ok and solution == sorted(problem)
        samples.append({
            "idx": i,
            "completion": c,
            "problem": problem,
            "solution": solution,
            "problem_parsed": problem_ok,
            "boxed_parsed": boxed_ok,
            "both_parsed": both_ok,
            "problem_len": len(problem) if problem_ok else None,
            "problem_len_ok": len_ok,
            "correct": correct,
        })

    # Per-sample display.
    print(f"\n=== generated samples (runtime {runtime:.1f}s) ===")
    for r in samples:
        c = r["completion"]
        c_show = c if len(c) <= 800 else c[:800] + f"... [truncated, total {len(c)} chars]"
        print(f"\n--- sample[{r['idx']}] "
              f"problem_parsed={r['problem_parsed']} "
              f"boxed_parsed={r['boxed_parsed']} "
              f"len={r['problem_len']} "
              f"len_ok={r['problem_len_ok']} "
              f"correct={r['correct']}")
        print(f"  completion: {c_show!r}")
        print(f"  extracted problem : {r['problem']}")
        print(f"  extracted solution: {r['solution']}")

    N = args.n_samples
    n_problem = sum(r["problem_parsed"] for r in samples)
    n_boxed = sum(r["boxed_parsed"] for r in samples)
    n_both = sum(r["both_parsed"] for r in samples)
    n_len_ok = sum(r["problem_len_ok"] for r in samples)
    n_correct = sum(r["correct"] for r in samples)

    print(f"\n=== stats (n={N}) ===")
    print(f"  <problem>...</problem> parsed : {n_problem}/{N}")
    print(f"  \\boxed{{...}}            parsed : {n_boxed}/{N}")
    print(f"  both parsed                   : {n_both}/{N}")
    print(f"  problem has 2 <= len < {args.list_length} ints: {n_len_ok}/{N}")
    print(f"  solution == sorted(problem)   : {n_correct}/{N}")

    summary = {
        "model": args.model,
        "n_samples": N,
        "list_length": args.list_length,
        "lo": args.lo,
        "hi": args.hi,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "seed": args.seed,
        "runtime_sec": runtime,
        "n_problem_parsed": n_problem,
        "n_boxed_parsed": n_boxed,
        "n_both_parsed": n_both,
        "n_problem_len_ok": n_len_ok,
        "n_correct": n_correct,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "samples.jsonl", "w") as f:
        for r in samples:
            f.write(json.dumps(r) + "\n")
    print(f"\n[saved] {out_dir}/summary.json  and samples.jsonl")


if __name__ == "__main__":
    main()
