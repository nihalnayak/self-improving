"""LLM-as-judge difficulty rating of SOAR promotion_questions.

Loads `<soar-run>/promotion_questions.jsonl` (synthetic teacher-generated math
problems, 64 per promotion event) together with `<soar-run>/promotions/index.jsonl`
(metadata linking each PQ to its promotion), runs a local vLLM judge on every PQ
to rate difficulty on the MATH 1–5 scale, and persists per-question judgments +
aggregate stats. Also prints 3 sample PQs per promotion and 5 random fail@128
test examples for visual grounding against the target distribution.

CLI:
    python -m analysis.judge_promotion_questions \\
        --soar-run /path/to/soar/run \\
        [--judge-model Qwen/Qwen3-32B] \\
        [--fail128-dir /override/path]
"""

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from soar.data import load_fail128_split


JUDGE_PROMPT = """You are judging a math problem's difficulty on the MATH dataset scale.

Scale:
  1 = Prealgebra       (basic arithmetic, simple linear equations)
  2 = Algebra          (polynomials, systems, basic word problems)
  3 = Intermediate Algebra / Geometry (quadratics, trig, basic geometry proofs)
  4 = Precalculus / Number Theory     (sequences, complex numbers, modular arithmetic)
  5 = AMC/AIME/Olympiad               (contest problems requiring insight)

Subjects (pick one): Prealgebra, Algebra, Intermediate Algebra, Geometry, \
Number Theory, Counting & Probability, Precalculus, Other

Problem:
{problem}

Reference answer: {answer}

Respond with exactly one JSON object on a single line, no preamble:
{{"level": <1-5>, "subject": "<subject>", "rationale": "<one sentence>"}}"""


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_judge(text: str) -> dict:
    """Pull the first {...} block out of the judge's reply and json.loads it.
    On failure returns {level: None, subject: None, rationale: text[:200]}."""
    m = _JSON_RE.search(text)
    if not m:
        return {"level": None, "subject": None, "rationale": text[:200].strip()}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"level": None, "subject": None, "rationale": text[:200].strip()}
    lvl = obj.get("level")
    if isinstance(lvl, str) and lvl.strip().isdigit():
        lvl = int(lvl.strip())
    return {
        "level": lvl if isinstance(lvl, int) and 1 <= lvl <= 5 else None,
        "subject": obj.get("subject"),
        "rationale": obj.get("rationale", "")[:400],
    }


def _trunc(s: str, n: int = 400) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--soar-run", required=True,
                    help="path to a completed SOAR run")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-32B",
                    help="HF repo id for the vLLM judge")
    ap.add_argument("--fail128-dir", default=None,
                    help="override; default = fail128_dir recorded in <soar-run>/config.json")
    ap.add_argument("--results-root", default="files/analysis/results")
    ap.add_argument("--n-fail128-examples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    soar_run = Path(args.soar_run)
    run_cfg = json.loads((soar_run / "config.json").read_text())
    fail128_dir = args.fail128_dir or run_cfg["fail128_dir"]

    # 1) Load. PQs loaded raw (not via _read_jsonl) to keep promotion_idx.
    pqs = list(load_dataset(
        "json", data_files=str(soar_run / "promotion_questions.jsonl"), split="train"
    ))
    promos_path = soar_run / "promotions" / "index.jsonl"
    promos = list(load_dataset("json", data_files=str(promos_path), split="train"))
    promos_by_idx = {int(p["promotion_idx"]): p for p in promos}

    _, fail128_test = load_fail128_split(fail128_dir)

    print(f"[judge] soar run      = {soar_run}")
    print(f"[judge] base model    = {run_cfg['base_model']}")
    print(f"[judge] fail128 dir   = {fail128_dir}")
    print(f"[judge] n_promotions  = {len(promos)}")
    print(f"[judge] n_pqs         = {len(pqs)}")
    print(f"[judge] n_fail128_test= {len(fail128_test)}")
    print(f"[judge] judge model   = {args.judge_model}")

    # 2) Judge via vLLM.
    print(f"[judge] loading judge {args.judge_model} …")
    llm = LLM(
        model=args.judge_model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=args.max_model_len,
    )
    tok = AutoTokenizer.from_pretrained(args.judge_model)

    prompts = [
        tok.apply_chat_template(
            [{"role": "user",
              "content": JUDGE_PROMPT.format(problem=p["problem"], answer=p["answer"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3: skip <think>…</think>
        )
        for p in pqs
    ]
    sp = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, seed=args.seed,
    )
    outs = llm.generate(prompts, sp)
    judgments = [_parse_judge(o.outputs[0].text) for o in outs]
    raw_texts = [o.outputs[0].text for o in outs]

    n_parse_fail = sum(1 for j in judgments if j["level"] is None)
    print(f"[judge] parse failures: {n_parse_fail} / {len(pqs)}")

    # 3) Persist.
    out_dir = Path(args.results_root) / soar_run.name
    out_dir.mkdir(parents=True, exist_ok=True)

    judgments_path = out_dir / "judgments.jsonl"
    with judgments_path.open("w") as f:
        for pq, j, raw in zip(pqs, judgments, raw_texts):
            rec = {**pq,
                   "judge_level": j["level"],
                   "judge_subject": j["subject"],
                   "judge_rationale": j["rationale"],
                   "judge_raw": raw}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[judge] wrote {judgments_path}")

    # Per-promotion aggregates.
    per_promotion = []
    for pidx in sorted({int(p["promotion_idx"]) for p in pqs}):
        group = [(p, j) for p, j in zip(pqs, judgments)
                 if int(p["promotion_idx"]) == pidx]
        levels = [j["level"] for _, j in group if j["level"] is not None]
        subjects = [j["subject"] for _, j in group if j["subject"]]
        meta = promos_by_idx.get(pidx, {})
        per_promotion.append({
            "promotion_idx": pidx,
            "outer_step": meta.get("outer_step"),
            "k_star": meta.get("k_star"),
            "j_star": meta.get("j_star"),
            "R_bar_t": meta.get("R_bar_t"),
            "moving_avg": meta.get("moving_avg"),
            "n_pqs": len(group),
            "n_judged": len(levels),
            "mean_level": statistics.mean(levels) if levels else None,
            "median_level": statistics.median(levels) if levels else None,
            "level_hist": dict(sorted(Counter(levels).items())),
            "subject_hist": dict(Counter(subjects).most_common()),
        })

    all_levels = [j["level"] for j in judgments if j["level"] is not None]
    all_subjects = [j["subject"] for j in judgments if j["subject"]]
    summary = {
        "soar_run": str(soar_run),
        "judge_model": args.judge_model,
        "n_pqs": len(pqs),
        "n_parse_fail": n_parse_fail,
        "overall": {
            "mean_level": statistics.mean(all_levels) if all_levels else None,
            "median_level": statistics.median(all_levels) if all_levels else None,
            "level_hist": dict(sorted(Counter(all_levels).items())),
            "subject_hist": dict(Counter(all_subjects).most_common()),
        },
        "per_promotion": per_promotion,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[judge] wrote {summary_path}")

    # 4) Printed report.
    print("\n" + "=" * 80)
    print("OVERALL")
    print("=" * 80)
    print(f"mean level : {summary['overall']['mean_level']}")
    print(f"level hist : {summary['overall']['level_hist']}")
    print(f"subjects   : {summary['overall']['subject_hist']}")

    for row in per_promotion:
        print("\n" + "=" * 80)
        print(f"PROMOTION {row['promotion_idx']}  (outer_step={row['outer_step']}, "
              f"k*={row['k_star']}, j*={row['j_star']}, R̄_t={row['R_bar_t']})")
        print("=" * 80)
        print(f"mean level : {row['mean_level']}  median: {row['median_level']}")
        print(f"level hist : {row['level_hist']}")
        print(f"subjects   : {row['subject_hist']}")

        group = [(p, j) for p, j in zip(pqs, judgments)
                 if int(p["promotion_idx"]) == row["promotion_idx"]
                 and j["level"] is not None]
        if not group:
            continue
        group_sorted = sorted(group, key=lambda x: x[1]["level"])
        picks: List[tuple] = []
        picks.append(("LOWEST ", group_sorted[0]))
        picks.append(("MEDIAN ", group_sorted[len(group_sorted) // 2]))
        picks.append(("HIGHEST", group_sorted[-1]))
        for tag, (p, j) in picks:
            print(f"\n[{tag} level={j['level']}  subject={j['subject']}]")
            print(f"  Q: {_trunc(p['problem'])}")
            print(f"  A: {_trunc(str(p['answer']), 120)}")
            print(f"  R: {_trunc(j['rationale'], 300)}")

    # fail@128 samples.
    print("\n" + "=" * 80)
    print(f"FAIL@128 TEST SAMPLES  ({args.n_fail128_examples} random, seed={args.seed})")
    print("=" * 80)
    rng = random.Random(args.seed)
    k = min(args.n_fail128_examples, len(fail128_test))
    for q in rng.sample(fail128_test, k):
        print(f"\n[qid={q.qid}  level={q.level}  subject={q.subject}]")
        print(f"  Q: {_trunc(q.problem)}")
        print(f"  A: {_trunc(str(q.answer), 120)}")


if __name__ == "__main__":
    main()
