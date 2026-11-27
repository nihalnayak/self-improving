"""Qwen3-4B avg pass@1 on reasoning-gym tasks at several difficulty levels.

Each task has one difficulty knob; DIFFICULTIES lists its values. For every
(task, value) we draw --n examples, sample --n-samples completions each
(thinking off), score with the dataset's own score_answer, and report avg
pass@1 = mean correctness over items x samples.

Prompt follows this repo's task style (user message only, final answer in
\\boxed{}, cf. misc/gameof24_benchmark.py) with step-by-step reasoning allowed;
temperature/top_p follow reasoning-gym eval/eval.py.

Usage:
    python -m misc.rg_examples
    python -m misc.rg_examples --tasks countdown --n 5 --n-samples 2
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from reasoning_gym.factory import create_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DIFFICULTIES = {
    "spell_backward": [3, 5, 7, 9],                  # word_len
    "count_primes":   [1000, 10000, 50000, 100000],  # max_n
    "countdown":      [3, 4, 5, 6],                  # num_numbers
}

PROMPT_TEMPLATE = (
    "Solve the following task.\n\n"
    "{question}\n\n"
    "You may think step by step. Then give your final answer on a single line, "
    "enclosed in \\boxed{{}}, e.g. \\boxed{{{example}}}."
)
EXAMPLES = {"spell_backward": "nus", "count_primes": "12", "countdown": "(1 + 2) * 3"}

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def parse_boxed(text):
    """First \\boxed{...} after the thinking trace, LaTeX operators normalized, else None."""
    m = _BOXED_RE.search(text.rsplit("</think>", 1)[-1])
    if not m:
        return None
    s = m.group(1)
    for a, b in (("\\times", "*"), ("\\cdot", "*"), ("\\div", "/"), ("×", "*"), ("÷", "/"),
                 ("\\left", ""), ("\\right", "")):
        s = s.replace(a, b)
    return s.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default=",".join(DIFFICULTIES))
    p.add_argument("--n", type=int, default=10, help="examples per level")
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--out-dir", default="files/results/rg_examples")
    args = p.parse_args()

    # 1) Build datasets per (task, difficulty value).
    levels = []  # (task, value, ds)
    for task in args.tasks.split(","):
        prev = 10
        for v in DIFFICULTIES[task]:
            if task == "spell_backward":
                cfg = dict(min_word_len=v, max_word_len=v)
            elif task == "count_primes":
                cfg = dict(min_n=prev, max_n=v)
                prev = v
            elif task == "countdown":
                cfg = dict(min_numbers=v, max_numbers=v)
            ds = create_dataset(task, **cfg, seed=args.seed, size=args.n)
            levels.append((task, v, ds))

    # 2) Prompts.
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts, index = [], []
    for li, (task, _, ds) in enumerate(levels):
        for i, ex in enumerate(ds):
            content = PROMPT_TEMPLATE.format(question=ex["question"], example=EXAMPLES[task])
            prompts.append(tok.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False))
            index.append((li, i))
    for task in dict.fromkeys(t for t, _, _ in levels):
        j = next(k for k, (li, _) in enumerate(index) if levels[li][0] == task)
        print(f"=== example prompt: {task} ===\n{prompts[j]}\n=== end prompt ===")

    # 3) Generate.
    engine = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        disable_log_stats=True,
        enable_prefix_caching=True,
    )
    sp = SamplingParams(n=args.n_samples, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed)
    t0 = time.time()
    outs = engine.generate(prompts, sp)
    print(f"[gen] {len(prompts)} prompts x n={args.n_samples} in {time.time() - t0:.1f}s")

    # 4) Score.
    rows = []
    for (li, i), req in zip(index, outs):
        task, v, ds = levels[li]
        ex = ds[i]
        comps = [o.text for o in req.outputs]
        parsed = [parse_boxed(c) for c in comps]
        oks = [ds.score_answer(a, ex) == 1.0 for a in parsed]
        rows.append({"task": task, "difficulty": v, "idx": i, "question": ex["question"],
                     "gold": ex["answer"], "completions": comps, "parsed": parsed,
                     "oks": oks, "pass1": sum(oks) / len(oks)})

    out_dir = Path(args.out_dir) / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "items.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = {}
    for task, v, _ in levels:
        rs = [r for r in rows if r["task"] == task and r["difficulty"] == v]
        summary.setdefault(task, {})[v] = sum(r["pass1"] for r in rs) / len(rs)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # 5) Report.
    print("\n=== examples ===")
    for task, v, _ in levels:
        rs = [r for r in rows if r["task"] == task and r["difficulty"] == v]
        print(f"\n--- {task} | difficulty={v} ---")
        for r in rs[:2]:
            q = r["question"] if len(r["question"]) <= 300 else r["question"][:300] + "..."
            print(f"Q: {q}\ngold: {r['gold']}\nparsed[0]: {r['parsed'][0]}\noks: {r['oks']}")

    print(f"\n=== avg pass@1 ({args.n} items x {args.n_samples} samples) ===")
    for task, vs in summary.items():
        for v, acc in vs.items():
            print(f"  {task:<15} difficulty={v:<7} {acc * 100:5.1f}%")
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
