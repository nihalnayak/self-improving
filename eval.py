"""Unbiased pass@k eval of a student LoRA (or base model) on the fail@128 test split.

Metric is the paper's unbiased pass@k estimator (SOAR §4.3; Chen et al. 2021),
computed in `soar.verify.pass_at_k`. For a single greedy run at T=0 just pass
`--n-samples 1 --temperature 0.0 --ks 1`.

Usage:
    python eval.py                                           # base model, pass@{1,4,8,16,32}
    python eval.py --student_lora runs/.../promo_003/student # chosen student LoRA
    python eval.py --n-samples 32 --temperature 1.0          # paper (Table 3)
    python eval.py --ks 1,4,8 --output-json /tmp/eval.json   # save JSON report
"""

import argparse
import json
from pathlib import Path

from vllm import SamplingParams

from soar.config import SoarConfig
from soar.data import load_fail128_split
from soar.models import load_vllm_engine, student_chat_prompt
from soar.rollout import vllm_generate
from soar.verify import pass_at_k
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail128-dir", default="files/data/math_fail128")
    ap.add_argument("--n-samples", type=int, default=32,
                    help="completions per problem")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="paper Table 3 default = 1.0")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="truncate test set for a quick run")
    ap.add_argument("--seed", type=int, default=0)
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
    prompts = [student_chat_prompt(tok, q.problem) for q in test]
    golds = [q.answer for q in test]

    if args.student_lora is None:
        sp = SamplingParams(
            n=args.n_samples, temperature=args.temperature, top_p=1.0,
            max_tokens=args.max_tokens, seed=args.seed,
        )
        outs = engine.generate(prompts, sp, use_tqdm=False)
        completions_by_q = [[c.text for c in o.outputs] for o in outs]
    else:
        # vllm_generate yields one completion per prompt; repeat prompts to get N samples.
        rep_prompts = [p for p in prompts for _ in range(args.n_samples)]
        flat = vllm_generate(
            engine, rep_prompts, lora_path=args.student_lora, lora_name="student",
            max_new_tokens=args.max_tokens, temperature=args.temperature,
        )
        completions_by_q = [
            flat[i * args.n_samples:(i + 1) * args.n_samples] for i in range(len(test))
        ]

    pk = pass_at_k(completions_by_q, golds, ks=ks)
    pk_str = {str(k): v for k, v in pk.items()}
    print("[eval] pass@k = " + ", ".join(f"{k}:{v:.4f}" for k, v in pk.items()))

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
