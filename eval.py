"""Unbiased pass@k eval of a student LoRA (or base model) on the fail@128 test split.

Metric is the paper's unbiased pass@k estimator (SOAR §4.3; Chen et al. 2021),
computed in `soar.verify.pass_at_k`. For a single greedy run at T=0 just pass
`--n-samples 1 --temperature 0.0 --ks 1`.

Exposes `run_vllm_eval(engine, questions, tok, *, temperature, n_samples,
max_new_tokens, ks, seed, lora_path=None)` for every vLLM-based student eval in
the repo (HO/PQ val callbacks, promoted_student scorer, single-GPU TRL eval,
and this script's main()). `seed` MUST differ from the fail@128 filter seed
(SOAR §5.1: *"stochastic sampling with different seeds than were used for the
initial fail@128 filtering"*). Filter default is 42; training/eval seeds are
0/1/2 per the sweep, so they're disjoint by construction.

Usage:
    python eval.py                                           # base model, pass@{1,4,8,16,32}
    python eval.py --student_lora runs/.../promo_003/student # chosen student LoRA
    python eval.py --n-samples 32 --temperature 1.0          # paper (Table 3)
    python eval.py --ks 1,4,8 --output-json /tmp/eval.json   # save JSON report
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vllm import SamplingParams
from vllm.lora.request import LoRARequest

from soar.config import SoarConfig
from soar.data import MathProblem, load_fail128_split
from soar.models import load_vllm_engine, student_chat_prompt
from soar.verify import pass_at_k
from transformers import AutoTokenizer


_FAIL128_FILTER_SEED = 42  # must stay in sync with misc/build_fail128.py default
_LORA_EVAL_COUNTER = itertools.count(10_000_000)


def run_vllm_eval(
    engine,
    questions: List[MathProblem],
    tok,
    *,
    temperature: float,
    n_samples: int,
    max_new_tokens: int,
    ks: Tuple[int, ...] = (1,),
    seed: int,
    lora_path: Optional[str] = None,
) -> Dict[str, float]:
    """Chat-template `questions`, draw `n_samples` completions each at
    `temperature` with vLLM `SamplingParams(seed=seed)`, score, return
    `{str(k): pass@k}` for each k.

    `seed` must differ from the fail@128 filter seed (paper §5.1). Callers
    forward their training/eval script's own `--seed` verbatim.

    If `lora_path` is set, applies the adapter via `LoRARequest` (fresh vLLM
    with `enable_lora=True`). If `None`, generates from the engine as-is —
    covers both base-model eval and the TRL colocate engine (where
    `sync_weights()` has already merged the LoRA into base params).
    """
    assert seed != _FAIL128_FILTER_SEED, (
        f"eval seed {seed} collides with fail@128 filter seed; see SOAR §5.1"
    )
    prompts = [student_chat_prompt(tok, q.problem) for q in questions]
    sp = SamplingParams(
        n=n_samples, temperature=temperature, top_p=1.0,
        max_tokens=max_new_tokens, seed=seed,
    )
    kw = {}
    if lora_path is not None:
        kw["lora_request"] = LoRARequest(
            lora_name="student",
            lora_int_id=next(_LORA_EVAL_COUNTER),
            lora_path=lora_path,
        )
    outs = engine.generate(prompts, sp, use_tqdm=False, **kw)
    completions_by_q = [[c.text for c in o.outputs] for o in outs]
    golds = [q.answer for q in questions]
    pk = pass_at_k(completions_by_q, golds, ks=ks)
    return {str(k): v for k, v in pk.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail128-dir", default="files/data/math_fail128")
    ap.add_argument("--n-samples", type=int, default=32,
                    help="completions per problem")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="paper Table 3 default = 1.0")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="truncate test set for a quick run")
    ap.add_argument("--seed", type=int, default=0,
                    help="vLLM SamplingParams seed; MUST differ from fail@128 filter seed (42)")
    ap.add_argument("--student_lora", type=str, default=None,
                    help="path to a PEFT adapter dir; if None, eval base model")
    ap.add_argument("--ks", type=str, default="1,4,8,16,32",
                    help="comma-separated K values for pass@k")
    ap.add_argument("--output-json", type=str, default=None,
                    help="optional path to save a pass@k report JSON")
    args = ap.parse_args()

    ks = tuple(int(k) for k in args.ks.split(","))

    cfg = SoarConfig(seed=args.seed, student_max_new_tokens=args.max_tokens)
    _, test = load_fail128_split(args.fail128_dir)
    if args.limit:
        test = test[: args.limit]
    print(f"[eval] test problems: {len(test)}")

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    engine = load_vllm_engine(cfg)
    pk_str = run_vllm_eval(
        engine, test, tok,
        temperature=args.temperature, n_samples=args.n_samples,
        max_new_tokens=args.max_tokens, ks=ks, seed=args.seed,
        lora_path=args.student_lora,
    )
    print("[eval] pass@k = " + ", ".join(f"{k}:{float(v):.4f}" for k, v in pk_str.items()))

    if args.output_json:
        report = {
            "student_lora": args.student_lora,
            "fail128_dir": args.fail128_dir,
            "n_test": len(test),
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "ks": list(ks),
            "pass_at_k": pk_str,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
