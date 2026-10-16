"""Standalone eval: student LoRA (or base) on MMLU-Pro math.

Usage:
    python eval.py --n 32                            # eval base Qwen (no LoRA)
    python eval.py --n 100 --student_lora path/to/   # eval a trained student adapter
"""

import argparse
from statistics import mean

from soar.config import SoarConfig
from soar.data import format_mcq_prompt, load_mmlu_pro_split
from soar.models import chat_prompt, load_vllm_engine
from soar.rollout import vllm_generate
from soar.verify import score_answer
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--subject", type=str, default="math")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--student_lora", type=str, default=None,
                    help="path to a PEFT adapter dir; if None, eval base model")
    args = ap.parse_args()

    cfg = SoarConfig(n_eval_questions=args.n, subject=args.subject, seed=args.seed)
    q_eval, _ = load_mmlu_pro_split(args.subject, args.n, args.seed)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    engine = load_vllm_engine(cfg)
    prompts = [chat_prompt(tok, format_mcq_prompt(q)) for q in q_eval]

    if args.student_lora is None:
        # No LoRA: use vLLM without a lora_request.
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=cfg.student_max_new_tokens)
        outs = engine.generate(prompts, sp, use_tqdm=False)
        completions = [o.outputs[0].text for o in outs]
    else:
        completions = vllm_generate(
            engine, prompts, lora_path=args.student_lora, lora_name="student",
            max_new_tokens=cfg.student_max_new_tokens, temperature=0.0,
        )

    scores = [score_answer(c, q.answer) for c, q in zip(completions, q_eval)]
    print(f"n={len(scores)}  acc={mean(scores):.3f}  (random baseline ≈ 1/{len(q_eval[0].options)})")
    for i in range(min(3, len(q_eval))):
        print("---")
        print(f"Q: {q_eval[i].question[:200]}...")
        print(f"gold={q_eval[i].answer}  gen={completions[i][:200]}...")


if __name__ == "__main__":
    main()
