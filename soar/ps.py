"""PS baseline (SOAR paper §4.3): Promoted Student.

Enumerate all promoted student LoRAs saved by `soar.train` to
`{soar_run}/promotions/promo_NNN/student/`, evaluate each greedy pass@1 on the
fail@128 train split, select the best, and report its pass@k (k ∈ {1,4,8,16,32})
on the fail@128 test split using 32 samples per problem at T=1.0 (Table 3, B.7).

Paper §4.3: *"For training runs that reached multiple promotions, we evaluate
the student model with the best validation performance (i.e., best 𝒟_train
greedy accuracy) on the test set."*

CLI: python -m soar.ps --soar-run runs/soar-v1 [--out runs/ps-v1] [--smoke]
"""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import List, Optional

from transformers import AutoTokenizer

from soar.config import SoarConfig, smoke_config
from soar.data import MathProblem, load_fail128_split
from soar.models import load_vllm_engine, student_chat_prompt
from soar.rollout import vllm_generate
from soar.verify import pass_at_k, success_math_answer


def _resolve_lora_path(promo_dir: Path) -> Optional[Path]:
    """`save_pretrained(..., selected_adapters=["student"])` writes either
    `{promo_dir}/adapter_config.json` or `{promo_dir}/student/adapter_config.json`
    depending on the PEFT version. Return the directory that contains the
    adapter_config.json, or None if neither exists.
    """
    if (promo_dir / "adapter_config.json").exists():
        return promo_dir
    nested = promo_dir / "student"
    if (nested / "adapter_config.json").exists():
        return nested
    return None


def _discover_promotions(soar_run: Path) -> List[Path]:
    pdir = soar_run / "promotions"
    if not pdir.exists():
        return []
    out: List[Path] = []
    for d in sorted(pdir.glob("promo_*")):
        if not d.is_dir():
            continue
        lp = _resolve_lora_path(d)
        if lp is not None:
            out.append(lp)
    return out


def _greedy_pass1(engine, questions: List[MathProblem], lora_path: Path, tok,
                  cfg: SoarConfig) -> float:
    prompts = [student_chat_prompt(tok, q.problem) for q in questions]
    completions = vllm_generate(
        engine, prompts, lora_path=str(lora_path), lora_name="student",
        max_new_tokens=cfg.student_max_new_tokens, temperature=0.0,
    )
    scores = [1.0 if success_math_answer(c, q.answer) else 0.0
              for c, q in zip(completions, questions)]
    return float(mean(scores)) if scores else 0.0


def _sample_per_question(engine, questions: List[MathProblem], lora_path: Path, tok,
                         *, n_samples: int, temperature: float,
                         max_new_tokens: int) -> List[List[str]]:
    """Return `[len(questions)] x [n_samples]` completions. One vLLM call."""
    flat_prompts: List[str] = []
    for q in questions:
        flat_prompts.extend([student_chat_prompt(tok, q.problem)] * n_samples)
    comps = vllm_generate(
        engine, flat_prompts, lora_path=str(lora_path), lora_name="student",
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    return [comps[i * n_samples:(i + 1) * n_samples] for i in range(len(questions))]


def main(cfg: SoarConfig, soar_run: str, out_dir: str) -> None:
    soar_run_p = Path(soar_run)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print(f"[ps] loading fail@128 from {cfg.fail128_dir}")
    train_set, test_set = load_fail128_split(cfg.fail128_dir)
    print(f"[ps] |train|={len(train_set)}  |test|={len(test_set)}")

    promo_paths = _discover_promotions(soar_run_p)
    assert promo_paths, (
        f"[ps] no promoted student LoRAs found under {soar_run_p / 'promotions'}. "
        "Run soar.train first (it saves to promotions/promo_NNN/ on each promotion)."
    )
    print(f"[ps] found {len(promo_paths)} promoted student checkpoints")

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print("[ps] loading vLLM engine")
    engine = load_vllm_engine(cfg)

    candidates = []
    for lp in promo_paths:
        # promo_idx from parent dir name: either "promo_001" (nested layout) or itself.
        promo_dir = lp if lp.name.startswith("promo_") else lp.parent
        promo_idx = int(promo_dir.name.split("_")[-1])
        acc = _greedy_pass1(engine, train_set, lp, tok, cfg)
        candidates.append({"promo_idx": promo_idx, "path": str(lp),
                           "train_acc_greedy": acc})
        print(f"[ps] promo_{promo_idx:03d}: train greedy pass@1 = {acc:.4f}")

    best = max(candidates, key=lambda c: c["train_acc_greedy"])
    print(f"[ps] selected promo_{best['promo_idx']:03d} "
          f"(train_acc_greedy={best['train_acc_greedy']:.4f})")

    print(f"[ps] test pass@k eval: {cfg.pq_pass_at_k_samples} samples/problem at T={cfg.pq_temperature}")
    comps_by_q = _sample_per_question(
        engine, test_set, Path(best["path"]), tok,
        n_samples=cfg.pq_pass_at_k_samples,
        temperature=cfg.pq_temperature,
        max_new_tokens=cfg.pq_max_new_tokens,
    )
    golds = [q.answer for q in test_set]
    pk = pass_at_k(comps_by_q, golds, ks=tuple(cfg.pq_pass_at_k_ks))
    pk_str = {str(k): v for k, v in pk.items()}
    print("[ps] test pass@k = " + ", ".join(f"{k}:{v:.4f}" for k, v in pk.items()))

    report = {
        "soar_run": str(soar_run_p),
        "fail128_dir": cfg.fail128_dir,
        "n_candidates": len(candidates),
        "candidates": candidates,
        "selected": {
            "promo_idx": best["promo_idx"],
            "path": best["path"],
            "train_acc_greedy": best["train_acc_greedy"],
            "test_pass_at_k": pk_str,
        },
        "eval": {
            "n_samples": cfg.pq_pass_at_k_samples,
            "temperature": cfg.pq_temperature,
            "max_new_tokens": cfg.pq_max_new_tokens,
            "ks": list(cfg.pq_pass_at_k_ks),
        },
    }
    with open(out_p / "ps_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[ps] wrote {out_p / 'ps_report.json'}")


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soar-run", type=str, required=True,
                    help="path to a completed SOAR run output dir (must contain promotions/)")
    ap.add_argument("--out", type=str, default=None, help="override out_dir")
    ap.add_argument("--fail128-dir", type=str, default=None, help="override fail128_dir")
    ap.add_argument("--smoke", action="store_true", help="tiny config for a sanity run")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = smoke_config() if args.smoke else SoarConfig()
    if args.fail128_dir:
        cfg.fail128_dir = args.fail128_dir
    out_dir = args.out or (str(Path(args.soar_run) / "ps_eval"))
    main(cfg, args.soar_run, out_dir)
