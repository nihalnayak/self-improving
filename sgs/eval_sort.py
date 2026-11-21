"""Solver pass@n on sorting (or --task multiplication / lcm / preorder) datasets for any number of checkpoints, via vLLM.

Same prompt, verifier and sampling as the in-training evals of sgs_simple and
solver_simple (sgs/common.py; 5 samples at T=1), so the numbers line up with
their `eval/pass@5/*` (train) and `val/pass@5/*` (val) curves.

    --checkpoints   any mix of
                      HF model dir   (has config.json)
                      run out_dir    (every ckpt/iter_*/solver/huggingface   [sgs_simple]
                                      and ckpt/global_step_*/actor/huggingface [solver_simple])
                      HF hub id      (e.g. Qwen/Qwen3-1.7B, the base model)
    --datasets      split=jsonl pairs (default: the task's train + val sets)
    --task          sorting (default) | multiplication | lcm | preorder: prompt, verifier
                    and level (list length / operand digits / nodes) from sgs/tasks.py

Outputs (under --out-dir):
    summary.jsonl                   one row per (checkpoint, split): {split}/pass@n/{len_L,all}
                                    (L = task level)
    <label>/<split>_samples.jsonl   every completion with its parsed answer and correctness;
                                    `list` is the problem, `level` its task level

Usage:
    python -m sgs.eval_sort --checkpoints Qwen/Qwen3-1.7B /n/netscratch/.../runs/<run>
    python -m sgs.eval_sort --checkpoints <run> --datasets val=files/data/sorting/sorting_val_100.jsonl
    python -m sgs.eval_sort --task lcm --checkpoints Qwen/Qwen3-4B \
        --datasets train=data/lcm/lcm_1000_L2-32.jsonl val=data/lcm/lcm_val_100.jsonl
"""

import argparse
import gc
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from sgs.tasks import TASKS, parse_preorder, parse_product, parse_sorted

# default train + val sets per task
DATASETS = {
    "sorting": ["train=files/data/sorting/sorting_1000.jsonl", "val=files/data/sorting/sorting_val_100.jsonl"],
    "multiplication": ["train=data/multiplication/multiplication_1000.jsonl",
                       "val=data/multiplication/multiplication_val_100.jsonl"],
    "lcm": ["train=data/lcm/lcm_1000.jsonl", "val=data/lcm/lcm_val_100.jsonl"],
    "preorder": ["train=data/preorder/preorder_1000.jsonl", "val=data/preorder/preorder_val_100.jsonl"],
}
# parsed answer stored per sample (correctness comes from task.verify)
PARSE = {"sorting": parse_sorted, "multiplication": parse_product, "lcm": parse_product,
         "preorder": parse_preorder}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--task", choices=sorted(TASKS), default="sorting",
                    help="prompts, verifier and data format (sgs/tasks.py)")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="split=path.jsonl (default: the task's train + val sets)")
    ap.add_argument("--out-dir", default="files/results/sgs/eval_sort")
    ap.add_argument("--n", type=int, default=5, help="samples per target (pass@n)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking", action="store_true", help="enable_thinking=True in the chat template")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--last-only", action="store_true", help="run dirs: only the last saved step")
    args = ap.parse_args(argv)
    args.datasets = args.datasets or DATASETS[args.task]
    return args


def expand_checkpoints(specs, last_only=False):
    """-> [(label, model_path)], run dirs expanded to every saved step (or the last) in step order."""
    out = []
    for spec in specs:
        p = Path(spec)
        if (p / "config.json").exists():
            out.append((p.resolve().as_posix().strip("/").replace("/", "_")[-120:], str(p)))
        elif (p / "ckpt").is_dir():
            found = list(p.glob("ckpt/iter_*/solver/huggingface")) + \
                    list(p.glob("ckpt/global_step_*/actor/huggingface"))
            assert found, f"no HF checkpoints under {p}/ckpt"
            step = lambda d: int(re.search(r"(?:iter_|global_step_)(\d+)", str(d)).group(1))
            if last_only:
                found = [max(found, key=step)]
            out +=[(f"{p.name}/step_{step(d):04d}", str(d)) for d in sorted(found, key=step)]
        else:
            out.append((spec.replace("/", "_"), spec))  # HF hub id
    return out


def evaluate(llm, tok, targets, args):
    """pass@n per task level (sorting: list length) and overall + per-sample rows."""
    task = TASKS[args.task]
    prompts = [tok.apply_chat_template(task.solver_messages(xs), tokenize=False, add_generation_prompt=True,
                                       enable_thinking=args.thinking) for xs in targets]
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.generate(prompts, sp)
    rows, passed = [], []
    for xs, req in zip(targets, outs):
        texts = [o.text for o in req.outputs]
        parsed = [PARSE[args.task](t) for t in texts]
        oks = [task.verify(t, xs) for t in texts]
        passed.append(any(oks))
        rows.append({"list": xs, "level": task.level(xs), "completions": texts, "parsed": parsed,
                     "correct": oks})
    passed, lens = np.array(passed), np.array([task.level(x) for x in targets])
    m = {f"pass@{args.n}/len_{L}": float(passed[lens == L].mean()) for L in sorted(set(lens.tolist()))}
    m[f"pass@{args.n}/all"] = float(passed.mean())
    return m, rows


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {s: TASKS[args.task].load(p) for s, p in (d.split("=", 1) for d in args.datasets)}
    ckpts = expand_checkpoints(args.checkpoints, args.last_only)
    print(f"[eval] {len(ckpts)} checkpoints x splits {[(s, len(t)) for s, t in splits.items()]}", flush=True)

    for label, path in ckpts:
        print(f"\n[load] {label}: {path}", flush=True)
        tok = AutoTokenizer.from_pretrained(path)
        llm = LLM(model=path, dtype="bfloat16", max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  tensor_parallel_size=max(1, torch.cuda.device_count()),
                  enable_prefix_caching=True, disable_log_stats=True, seed=args.seed)
        (out_dir / label).mkdir(parents=True, exist_ok=True)
        for split, targets in splits.items():
            m, rows = evaluate(llm, tok, targets, args)
            print(f"[{label}] {split}: " + " ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
            with open(out_dir / label / f"{split}_samples.jsonl", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            with open(out_dir / "summary.jsonl", "a") as f:
                f.write(json.dumps({"checkpoint": label, "path": path, "split": split,
                                    **{f"{split}/{k}": v for k, v in m.items()}}) + "\n")
        del llm  # shuts down the vLLM engine before loading the next checkpoint
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
