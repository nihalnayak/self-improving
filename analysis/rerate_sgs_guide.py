"""Re-rate an SGS run's synthetic problems with the Guide LLM, offline via vLLM.

Loads every (target, conjecture statement) pair the Guide scored during the run
(`<run>/samples/iter_*.json`, key `conjectures`), regenerates the Guide rating
with the same model, prompt (sgs_simple.guide_messages) and sampling (T=1,
top_p=1, 2048 tokens, no thinking), and parses it two ways:
    strict   sgs_simple.parse_guide: needs <tag>N</tag> with matching close tag
    lenient  <tag>N followed by any closing tag (the Guide often writes
             <complexity>2</redundancy>)
R_guide = sgs_simple.guide_score(scores) for both.

Outputs (under --out-dir):
    ratings.jsonl   one row per pair: iter, target, statement, solve_rate,
                    completion, strict/lenient scores and R_guide
    summary.jsonl   one row per iter: parse rates, mean R_guide, score means

Usage:
    python -m analysis.rerate_sgs_guide --run /n/netscratch/.../runs/<sgs run>
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from sgs.sgs_simple import guide_messages, guide_score, parse_guide

_LENIENT = {tag: re.compile(rf"<{tag}>\s*(\d+)\s*</?\w*>?", re.IGNORECASE)
            for tag in ("relevance", "redundancy", "complexity")}


def parse_guide_lenient(text):
    """(relevance, redundancy, complexity) from the last `<tag>N` of each kind, or None."""
    out = []
    for tag, hi in (("relevance", 5), ("redundancy", 1), ("complexity", 4)):
        found = _LENIENT[tag].findall(text)
        if not found or not 0 <= int(found[-1]) <= hi:
            return None
        out.append(int(found[-1]))
    return tuple(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True, help="sgs_simple out_dir")
    ap.add_argument("--model", default=None, help="Guide model (default: the run's --model)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_args = json.loads((args.run / "args.json").read_text())
    model = args.model or run_args["model"]
    out_dir = args.out_dir or Path("files/results/sgs/guide_rerate") / args.run.name
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for f in sorted(args.run.glob("samples/iter_*.json")):
        it = int(f.stem.split("_")[1])
        for c in json.loads(f.read_text())["conjectures"]:
            if c.get("statement") and c.get("problem"):  # same filter as sgs_iteration's `synth`
                pairs.append({"iter": it, "target": c["target"], "statement": c["statement"],
                              "problem": c["problem"], "solve_rate": c.get("solve_rate"),
                              "run_r_guide": c.get("r_guide")})
    print(f"[data] {len(pairs)} (target, conjecture) pairs from {args.run.name}", flush=True)

    tok = AutoTokenizer.from_pretrained(model)
    prompts = [tok.apply_chat_template(guide_messages(p["target"], p["statement"]), tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=run_args.get("thinking", False))
               for p in pairs]
    llm = LLM(model=model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85,
              enable_prefix_caching=True, disable_log_stats=True, seed=args.seed)
    sp = SamplingParams(n=1, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.generate(prompts, sp)

    for p, o in zip(pairs, outs):
        text = o.outputs[0].text
        strict, lenient = parse_guide(text), parse_guide_lenient(text)
        p.update(completion=text, strict_scores=strict, lenient_scores=lenient,
                 strict_r_guide=guide_score(strict), lenient_r_guide=guide_score(lenient))
    with open(out_dir / "ratings.jsonl", "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in pairs)

    rows = []
    for it in sorted({p["iter"] for p in pairs}) + ["all"]:
        ps = [p for p in pairs if it == "all" or p["iter"] == it]
        ok = [p["lenient_scores"] for p in ps if p["lenient_scores"]]
        rel, red, cx = (np.array(x) for x in zip(*ok)) if ok else ([np.nan],) * 3
        rows.append({"iter": it, "n": len(ps),
                     "strict_parse_rate": np.mean([p["strict_scores"] is not None for p in ps]),
                     "lenient_parse_rate": np.mean([p["lenient_scores"] is not None for p in ps]),
                     "strict_mean_r_guide": np.mean([p["strict_r_guide"] for p in ps]),
                     "lenient_mean_r_guide": np.mean([p["lenient_r_guide"] for p in ps]),
                     "mean_relevance": float(np.mean(rel)), "mean_redundancy": float(np.mean(red)),
                     "mean_complexity": float(np.mean(cx)),
                     "frac_complexity_ge3": float(np.mean(np.asarray(cx) >= 3))})
    with open(out_dir / "summary.jsonl", "w") as f:
        f.writelines(json.dumps({k: (float(v) if isinstance(v, np.floating) else v)
                                 for k, v in r.items()}) + "\n" for r in rows)
    for r in rows:
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    print(f"[done] -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
