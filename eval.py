"""Standalone eval: student LoRA (or base) on the fail@128 held-out test split.

Usage:
    python eval.py                                           # eval base model
    python eval.py --student_lora runs/.../ckpt_outer*/baseline_student
    python eval.py --fail128-dir files/data/math_fail128 --n-samples 32 --temperature 0.7
"""

import argparse
from statistics import mean

from soar.config import SoarConfig
from soar.data import load_fail128_split
from soar.models import load_vllm_engine, student_chat_prompt
from soar.rollout import vllm_generate
from soar.verify import success_math_answer
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail128-dir", default="files/data/math_fail128")
    ap.add_argument("--n-samples", type=int, default=1,
                    help="completions per problem; use 1 for greedy pass@1 at T=0")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="truncate test set for a quick run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--student_lora", type=str, default=None,
                    help="path to a PEFT adapter dir; if None, eval base model")
    args = ap.parse_args()

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

    if args.student_lora is None:
        from vllm import SamplingParams
        sp = SamplingParams(
            n=args.n_samples, temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens,
        )
        outs = engine.generate(prompts, sp, use_tqdm=False)
        # For n>1: pass@k success per problem = any correct
        scores = []
        for o, q in zip(outs, test):
            ok = any(success_math_answer(c.text, q.answer) for c in o.outputs)
            scores.append(1.0 if ok else 0.0)
    else:
        if args.n_samples != 1:
            print(f"[eval] WARNING: vllm_generate helper takes one completion per prompt; "
                  f"repeating prompts {args.n_samples}× for pass@{args.n_samples}")
        reps = args.n_samples
        rep_prompts = [p for p in prompts for _ in range(reps)]
        completions = vllm_generate(
            engine, rep_prompts, lora_path=args.student_lora, lora_name="student",
            max_new_tokens=args.max_tokens, temperature=args.temperature,
        )
        scores = []
        for i, q in enumerate(test):
            group = completions[i * reps:(i + 1) * reps]
            ok = any(success_math_answer(c, q.answer) for c in group)
            scores.append(1.0 if ok else 0.0)

    pass_at_k = float(mean(scores)) if scores else 0.0
    print(f"n={len(scores)}  pass@{args.n_samples}={pass_at_k:.3f}")
    for i in range(min(3, len(test))):
        q = test[i]
        print("---")
        print(f"Q[{i}]: {q.problem[:200]}...")
        print(f"gold: {q.answer}")


if __name__ == "__main__":
    main()
