"""Build the fail@128 subset of MATH, following SOAR paper §B.5 exactly.

Pipeline:
    1. Load the official MATH *test* split (5000 problems). The paper deliberately uses the
       test split (not train) because Llama-3.2-3B-Instruct scores 60% on MATH train vs 37%
       on test, suggesting partial training-set contamination (§B.5 "MATH" paragraph).
    2. For each problem, generate 128 solutions with the base model at T=1.0, top_p=1.0,
       max_tokens=1024 (§B.5 "Fail@128 Filtering").
    3. Keep problems with 0/128 success → fail@128 subset.
    4. Random 50:50 split into our internal train/test. All downstream training and
       evaluation uses only these two files; the HF train split is never touched.

Usage:
    python -m misc.build_fail128
    python -m misc.build_fail128 --limit-problems 32 --n-samples 8 --out-dir files/data/math_smoke
"""

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

from soar.data import MathProblem
from soar.models import student_chat_prompt
from soar.verify import extract_boxed, success_math_answer
from transformers import AutoTokenizer


# ----- MATH test split loader -----

# EleutherAI's maintained mirror of Hendrycks MATH. Published as 7 per-subject configs
# (not a single dataset); concatenating their test splits gives the 5000-problem test set
# referenced by SOAR paper §B.5.
_MATH_REPO = "EleutherAI/hendrycks_math"
_MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def _row_to_math_problem(row: dict, idx: int) -> Optional[MathProblem]:
    """Convert one EleutherAI/hendrycks_math row to a MathProblem.
    Returns None only if the \\boxed{} answer can't be extracted."""
    problem = row["problem"]
    solution = row["solution"]
    gold = extract_boxed(solution)
    if gold is None:
        return None
    qid = hashlib.sha1(problem.encode("utf-8")).hexdigest()[:16]
    return MathProblem(
        qid=qid,
        problem=problem.strip(),
        answer=gold.strip(),
        solution=solution,
        level=str(row["level"]),
        subject=row["type"],
        idx=idx,
    )


def load_math_test_split() -> List[MathProblem]:
    """Load the full 5000-problem MATH test split from EleutherAI/hendrycks_math by
    concatenating per-subject configs. Each row carries an `idx` in [0, 5000) that
    tags its position in the concatenated test set (skipped rows still burn their
    index slot so `idx` uniquely identifies the source row)."""
    out: List[MathProblem] = []
    skipped = 0
    global_idx = 0
    for cfg in _MATH_CONFIGS:
        print(f"[data] loading {_MATH_REPO} config={cfg} split=test")
        ds = load_dataset(_MATH_REPO, cfg, split="test")
        if "idx" not in ds.column_names:
            ds = ds.add_column("idx", list(range(global_idx, global_idx + len(ds))))
        for row in ds:
            p = _row_to_math_problem(row, int(row["idx"]))
            if p is None:
                skipped += 1
                continue
            out.append(p)
        global_idx += len(ds)
    print(f"[data] loaded {len(out)} problems across {len(_MATH_CONFIGS)} configs "
          f"(skipped {skipped})")
    if len(out) != 5000:
        print(f"[data] WARNING: expected 5000 problems, got {len(out)}")
    return out


# ----- Main pass@128 pipeline -----

def build_fail128(args):
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = load_math_test_split()
    if args.limit_problems:
        problems = problems[: args.limit_problems]
        print(f"[data] limited to {len(problems)} problems (--limit-problems)")
    n_source = len(problems)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    n_gpus = torch.cuda.device_count()
    print(f"[vllm] loading engine: {args.base_model} (tensor_parallel_size={n_gpus})")
    llm_kwargs = dict(
        model=args.base_model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=False,
        disable_log_stats=True,
        enable_prefix_caching=True,
    )
    if n_gpus > 1:
        llm_kwargs["tensor_parallel_size"] = n_gpus
    engine = LLM(**llm_kwargs)

    sp = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    successes_per_problem: List[int] = []
    kept: List[MathProblem] = []
    t0 = time.time()

    B = args.batch_problems
    with tqdm(total=len(problems), desc=f"pass@{args.n_samples}") as pbar:
        for i in range(0, len(problems), B):
            chunk = problems[i:i + B]
            prompts = [student_chat_prompt(tok, p.problem) for p in chunk]
            outs = engine.generate(prompts, sp)
            for p, out in zip(chunk, outs):
                completions = [o.text for o in out.outputs]
                n_ok = sum(1 for c in completions if success_math_answer(c, p.answer))
                successes_per_problem.append(n_ok)
                if n_ok == 0:
                    kept.append(p)
            pbar.update(len(chunk))
            dt = time.time() - t0
            done = i + len(chunk)
            print(
                f"[pass@{args.n_samples}] {done}/{len(problems)}  "
                f"kept(fail@{args.n_samples})={len(kept)}  elapsed={dt:.0f}s"
            )

    # 50:50 split of the fail@128 subset.
    rng.shuffle(kept)
    half = len(kept) // 2
    train_split = kept[:half]
    test_split = kept[half:]
    print(f"[split] fail@{args.n_samples}={len(kept)}  train={len(train_split)}  test={len(test_split)}")

    with open(out_dir / "train.jsonl", "w") as f:
        for p in train_split:
            f.write(p.to_json() + "\n")
    with open(out_dir / "test.jsonl", "w") as f:
        for p in test_split:
            f.write(p.to_json() + "\n")

    hist = dict(Counter(successes_per_problem))
    meta = {
        "base_model": args.base_model,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "source": "MATH test split",
        "source_split": "test",
        "n_source": n_source,
        "n_fail": len(kept),
        "n_train": len(train_split),
        "n_test": len(test_split),
        "pass_at_n_histogram": {str(k): v for k, v in sorted(hist.items())},
        "runtime_sec": time.time() - t0,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[write] {out_dir}/{{train,test}}.jsonl and meta.json")


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--out-dir", default="files/data/math_fail128")
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42,
                    help="paper §5.1: must differ from training/eval --seed")
    ap.add_argument("--limit-problems", type=int, default=0,
                    help="If >0, only process the first N problems (smoke test).")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--batch-problems", type=int, default=100,
                    help="Number of problems to pool into one engine.generate() call.")
    return ap.parse_args()


if __name__ == "__main__":
    build_fail128(_parse_args())
