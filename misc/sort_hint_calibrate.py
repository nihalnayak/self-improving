"""Calibrate answer-span hints so a 128-length sort hits t/5 solve rate.

One random list of `--length` ints in [lo, hi]. The solver (Qwen3-4B, prompt
from misc/sort_benchmark.py) can't sort it unaided. A hint is a literal piece
of the gold sorted list; the target t fixes which piece (PATTERN):
    t=5,4 -> first + middle + last k values   (easy)
    t=3,2 -> middle k values                  (medium)
    t=1   -> last k values                    (hard)
A hint-writer LLM (same engine, sees the gold answer) picks only the span
length k; the hint text is rendered from gold, so every value is correct. We
check the solver's solves/n with that hint and feed k -> solves back to the
writer, until solves == t or `--max-rounds` is hit. The best hint per target
is then re-checked on a fresh draw (final result).

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


PATTERN = {5: "fml", 4: "fml", 3: "m", 2: "m", 1: "l"}   # f=first, m=middle, l=last
PATTERN_TEXT = {
    "fml": "the first K, the middle K and the last K values of the sorted answer",
    "m": "K consecutive values from the middle of the sorted answer",
    "l": "the last K values of the sorted answer",
}

HINT_WRITER_TEMPLATE = (
    "A language model is asked to sort this list of {L} integers in "
    "ascending order:\n\n"
    "List: {xs}\n\n"
    "The correct sorted answer is:\n\n"
    "Sorted: {gold}\n\n"
    "The model sees the list plus a hint. We sample it {n} times. We want it "
    "to get the exact sorted list right in exactly {t} of {n} attempts: not "
    "more, not less.\n\n"
    "The hint reveals {what}, written out exactly. You choose K. A larger K "
    "reveals more of the answer, so the model will solve it more often; a "
    "smaller K reveals less. K must be an integer from 1 to {cap}.\n\n"
    "Previous attempts (K -> correct answers out of {n}; K=none means no "
    "hint):\n{history}\n\n"
    "Return only the new K inside <hint>...</hint>, e.g. <hint>8</hint>."
)

_HINT_RE = re.compile(r"<hint>(.*?)</hint>", re.DOTALL)


def chat(tok, content, thinking):
    return tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
    )


def span_cap(L, pattern, full_ok):
    """Largest K; below the full list unless full_ok (t == n)."""
    if pattern == "fml":
        return max(1, L // 3 if full_ok else (L - 1) // 3)
    return L if full_ok else L - 1


def render_hint(gold, pattern, k):
    """Hint text with the requested spans of gold (1-indexed positions)."""
    L = len(gold)
    i = (L - k) // 2
    parts = {
        "f": f"The sorted answer starts with: {gold[:k]}.",
        "m": f"Positions {i + 1}-{i + k} of the sorted answer are: {gold[i:i + k]}.",
        "l": f"The sorted answer ends with: {gold[L - k:]}.",
    }
    return " ".join(parts[c] for c in pattern)


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
        """Returns (k or None, cap, raw writer text)."""
        call[0] += 1
        hist = "\n".join(f"- K={'none' if k is None else k} -> {s}/{n}"
                         for k, _, s in history)
        cap = span_cap(args.length, PATTERN[t], t == n)
        prompt = HINT_WRITER_TEMPLATE.format(
            L=args.length, xs=xs, gold=gold, n=n, t=t, history=hist,
            what=PATTERN_TEXT[PATTERN[t]], cap=cap,
        )
        sp = SamplingParams(n=1, temperature=0.7, top_p=0.95,
                            max_tokens=args.writer_max_tokens,
                            seed=args.seed + call[0])
        raw = engine.generate([chat(tok, prompt, args.writer_thinking)], sp)[0].outputs[0].text
        body = raw.split("</think>")[-1]
        m = _HINT_RE.search(body)
        km = re.search(r"\d+", m.group(1)) if m else None
        k = min(max(int(km.group()), 1), cap) if km else None
        return k, raw

    trace_f = open(out_dir / "trace.jsonl", "w")
    results = []
    for t in range(1, n + 1):
        pattern = PATTERN[t]
        k, hint = None, ""
        solves = solve(hint)
        history = [(k, hint, solves)]
        trace_f.write(json.dumps({"target": t, "round": 0, "pattern": pattern, "k": k,
                                  "hint": hint, "solves": solves, "writer_raw": None}) + "\n")
        print(f"[t={t}/{n} {pattern} r=0] k={k} solves={solves}/{n}")
        r = 0
        while solves != t and r < args.max_rounds:
            r += 1
            k, raw = write_hint(t, history)
            hint = "" if k is None else render_hint(gold, pattern, k)
            solves = solve(hint)
            history.append((k, hint, solves))
            trace_f.write(json.dumps({"target": t, "round": r, "pattern": pattern, "k": k,
                                      "hint": hint, "solves": solves, "writer_raw": raw}) + "\n")
            trace_f.flush()
            print(f"[t={t}/{n} {pattern} r={r}] k={k} solves={solves}/{n}")
        # closest to t, later round wins ties
        best_i = min(range(len(history)),
                     key=lambda i: (abs(history[i][2] - t), -i))
        best_k, best_hint, best_solves = history[best_i]
        results.append({"target": t, "pattern": pattern, "k": best_k, "hint": best_hint,
                        "prompt": solver_text(xs, best_hint),
                        "calib_solves": best_solves, "rounds": r})
    trace_f.close()

    print(f"\n=== final check (fresh draw, n={n}) ===")
    for res in results:
        res["final_solves"] = solve(res["hint"])
        print(f"target {res['target']}/{n} | {res['pattern']} k={res['k']} "
              f"| calib {res['calib_solves']}/{n} "
              f"| final {res['final_solves']}/{n} | rounds {res['rounds']}")
        print(f"  hint: {res['hint']}")

    (out_dir / "calibrated.json").write_text(json.dumps(
        {"args": vars(args), "list": xs, "gold": gold, "targets": results},
        indent=2))
    print(f"\n[saved] {out_dir}/calibrated.json and trace.jsonl")


if __name__ == "__main__":
    main()
