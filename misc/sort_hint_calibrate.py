"""Calibrate natural-language hints so a 128-length sort hits t/5 solve rate.

One random list of `--length` ints in [lo, hi]. The solver (Qwen3-4B, prompt
from misc/sort_benchmark.py) can't sort it unaided. For each target
t in {1..5}, a hint-writer LLM (same engine, sees the gold answer) rewrites a
natural-language hint; we check the solver's solves/n with that hint and feed
the result back to the writer, until solves == t or `--max-rounds` is hit.
The best hint per target is then re-checked on a fresh draw (final result).

Usage:
    python -m misc.sort_hint_calibrate
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from misc.sort_benchmark import PROMPT_TEMPLATE, parse_sorted


HINT_WRITER_TEMPLATE = (
    "A language model is asked to sort this list of {L} integers in "
    "ascending order:\n\n"
    "List: {xs}\n\n"
    "The correct sorted answer is:\n\n"
    "Sorted: {gold}\n\n"
    "The model sees the list plus a natural-language hint. We sample it "
    "{n} times. We want it to get the exact sorted list right in exactly "
    "{t} of {n} attempts: not more, not less.\n\n"
    "Previous attempts (hint -> correct answers out of {n}):\n{history}\n\n"
    "Write a new hint. If the model solved it too rarely, reveal more about "
    "the answer; if too often, reveal less. Useful clues include: the "
    "smallest and largest values, how many numbers fall in each range "
    "(e.g. below 1000, 1000-1999, ...), some of the sorted values at the "
    "start, middle or end, which values are duplicated, or a good "
    "bucketing strategy.{full_rule} Write the hint as plain natural-language "
    "sentences addressed to the model.\n\n"
    "Return only the new hint inside <hint>...</hint>."
)

_HINT_RE = re.compile(r"<hint>(.*?)</hint>", re.DOTALL)


def chat(tok, content, thinking):
    return tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
    )


def solver_text(xs, hint):
    text = PROMPT_TEMPLATE.format(list_str=str(xs))
    return text + f"\n\nHint: {hint}" if hint else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=10000)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--writer-max-tokens", type=int, default=8192)
    ap.add_argument("--thinking", action="store_true", help="solver thinking")
    ap.add_argument("--no-writer-thinking", dest="writer_thinking",
                    action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--out-dir", default="files/results/sorting_hints")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.model.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    xs = [rng.randint(args.lo, args.hi) for _ in range(args.length)]
    gold = sorted(xs)
    n = args.n_samples

    tok = AutoTokenizer.from_pretrained(args.model)
    engine = LLM(
        model=args.model, dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        disable_log_stats=True, enable_prefix_caching=True,
    )
    call = [0]  # distinct vLLM seed per call

    def solve(hint):
        call[0] += 1
        sp = SamplingParams(n=n, temperature=args.temperature,
                            max_tokens=args.max_tokens, seed=args.seed + call[0])
        out = engine.generate([chat(tok, solver_text(xs, hint), args.thinking)], sp)
        return sum(parse_sorted(o.text) == gold for o in out[0].outputs)

    def write_hint(t, history):
        call[0] += 1
        hist = "\n".join(f"- {h!r} -> {s}/{n}" for h, s in history)
        prompt = HINT_WRITER_TEMPLATE.format(
            L=args.length, xs=xs, gold=gold, n=n, t=t, history=hist,
            full_rule="" if t == n else
            " Do not write out the full sorted list.",
        )
        sp = SamplingParams(n=1, temperature=0.7, top_p=0.95,
                            max_tokens=args.writer_max_tokens,
                            seed=args.seed + call[0])
        raw = engine.generate([chat(tok, prompt, args.writer_thinking)], sp)[0].outputs[0].text
        body = raw.split("</think>")[-1]
        m = _HINT_RE.search(body)
        return (m.group(1).strip() if m else ""), raw

    trace_f = open(out_dir / "trace.jsonl", "w")
    results = []
    start_hint = ""
    for t in range(1, n + 1):
        hint = start_hint
        solves = solve(hint)
        history = [(hint, solves)]
        trace_f.write(json.dumps({"target": t, "round": 0, "hint": hint,
                                  "solves": solves, "writer_raw": None}) + "\n")
        print(f"[t={t}/{n} r=0] solves={solves}/{n} hint={hint[:200]!r}")
        r = 0
        while solves != t and r < args.max_rounds:
            r += 1
            hint, raw = write_hint(t, history)
            solves = solve(hint)
            history.append((hint, solves))
            trace_f.write(json.dumps({"target": t, "round": r, "hint": hint,
                                      "solves": solves, "writer_raw": raw}) + "\n")
            trace_f.flush()
            print(f"[t={t}/{n} r={r}] solves={solves}/{n} hint={hint[:200]!r}")
        # closest to t, later round wins ties
        best_i = min(range(len(history)),
                     key=lambda i: (abs(history[i][1] - t), -i))
        best_hint, best_solves = history[best_i]
        start_hint = best_hint
        results.append({"target": t, "hint": best_hint,
                        "prompt": solver_text(xs, best_hint),
                        "calib_solves": best_solves, "rounds": r})
    trace_f.close()

    print(f"\n=== final check (fresh draw, n={n}) ===")
    for res in results:
        res["final_solves"] = solve(res["hint"])
        print(f"target {res['target']}/{n} | calib {res['calib_solves']}/{n} "
              f"| final {res['final_solves']}/{n} | rounds {res['rounds']}")
        print(f"  hint: {res['hint']}")

    (out_dir / "calibrated.json").write_text(json.dumps(
        {"args": vars(args), "list": xs, "gold": gold, "targets": results},
        indent=2))
    print(f"\n[saved] {out_dir}/calibrated.json and trace.jsonl")


if __name__ == "__main__":
    main()
