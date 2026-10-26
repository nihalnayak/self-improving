"""PS baseline (SOAR paper §4.3): Promoted Student.

Enumerate all promoted student LoRAs saved by SOAR training to
`<out_dir>/promotions/promo_NNN/student/`, pick the one with the best
greedy pass@1 on D_train, then report its pass@k (k ∈ {1,4,8,16,32}) on
D_test via a subprocess call to `eval.py` (32 samples at T=1.0, Table 3).

CLI:
    python -m baselines.promoted_student \\
        --out-dir /path/to/soar/run \\
        --fail128-dir files/data/math_fail128 \\
        [--results-root files/results/baselines] \\
        [--base-model meta-llama/Llama-3.2-3B-Instruct]
"""

import argparse
import gc
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean
from typing import List, Optional

import torch
from transformers import AutoTokenizer

from soar.config import SoarConfig
from soar.data import MathProblem, load_fail128_split
from soar.models import load_vllm_engine, student_chat_prompt
from soar.rollout import vllm_generate
from soar.verify import success_math_answer


def _resolve_lora_path(promo_dir: Path) -> Optional[Path]:
    """`save_pretrained(..., selected_adapters=["student"])` writes either
    `{promo_dir}/adapter_config.json` or `{promo_dir}/student/adapter_config.json`
    depending on the PEFT version. Return whichever contains the adapter_config.json,
    or None if neither does."""
    if (promo_dir / "adapter_config.json").exists():
        return promo_dir
    nested = promo_dir / "student"
    if (nested / "adapter_config.json").exists():
        return nested
    return None


def find_lora_students(soar_run: Path) -> List[Path]:
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


def greedy_pass1(engine, questions: List[MathProblem], lora_path: Path, tok,
                 cfg: SoarConfig) -> float:
    prompts = [student_chat_prompt(tok, q.problem) for q in questions]
    completions = vllm_generate(
        engine, prompts, lora_path=str(lora_path), lora_name="student",
        max_new_tokens=cfg.student_max_new_tokens, temperature=0.0,
    )
    scores = [1.0 if success_math_answer(c, q.answer) else 0.0
              for c, q in zip(completions, questions)]
    return float(mean(scores)) if scores else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True,
                    help="SOAR run output dir (must contain promotions/)")
    ap.add_argument("--fail128-dir", type=str, required=True)
    ap.add_argument("--results-root", type=str, default="files/results/baselines")
    ap.add_argument("--base-model", type=str, default=None,
                    help="override; default = base_model recorded in <out-dir>/config.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    run_name = out_dir.name
    with open(out_dir / "config.json") as f:
        run_cfg = json.load(f)

    cfg = SoarConfig()
    cfg.fail128_dir = args.fail128_dir
    cfg.base_model = args.base_model or run_cfg["base_model"]
    model_short = Path(cfg.base_model).name.lower()

    print(f"[ps] loading fail@128 from {cfg.fail128_dir}")
    train_set, test_set = load_fail128_split(cfg.fail128_dir)
    print(f"[ps] |train|={len(train_set)}  |test|={len(test_set)}")

    promo_paths = find_lora_students(out_dir)
    assert promo_paths, (
        f"[ps] no promoted student LoRAs under {out_dir / 'promotions'}."
    )
    print(f"[ps] found {len(promo_paths)} promoted student checkpoints")

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print("[ps] loading vLLM engine for selection")
    engine = load_vllm_engine(cfg)

    candidates = []
    for lp in promo_paths:
        promo_dir = lp if lp.name.startswith("promo_") else lp.parent
        promo_idx = int(promo_dir.name.split("_")[-1])
        acc = greedy_pass1(engine, train_set, lp, tok, cfg)
        candidates.append({"promo_idx": promo_idx, "path": str(lp),
                           "train_acc_greedy": acc})
        print(f"[ps] promo_{promo_idx:03d}: train greedy pass@1 = {acc:.4f}")

    best = max(candidates, key=lambda c: c["train_acc_greedy"])
    print(f"[ps] selected promo_{best['promo_idx']:03d} "
          f"(train_acc_greedy={best['train_acc_greedy']:.4f})")

    # Release engine before eval.py spins up its own.
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tmp_dir = Path(tempfile.mkdtemp(prefix="ps_eval_"))
    tmp_json = tmp_dir / "eval.json"
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable, str(repo_root / "eval.py"),
        "--student_lora", best["path"],
        "--fail128-dir", args.fail128_dir,
        "--n-samples", str(cfg.pq_pass_at_k_samples),
        "--temperature", str(cfg.pq_temperature),
        "--max-tokens", str(cfg.pq_max_new_tokens),
        "--ks", ",".join(str(k) for k in cfg.pq_pass_at_k_ks),
        "--output-json", str(tmp_json),
    ]
    print(f"[ps] running final eval: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(repo_root))
    eval_report = json.loads(tmp_json.read_text())

    result = {
        "baseline": "promoted_student",
        "baseline_tag": "PS",
        "run_name": run_name,
        "out_dir": str(out_dir),
        "base_model": cfg.base_model,
        "model_short": model_short,
        "selection_metric": "train_pass@1_greedy",
        "selection_scores": candidates,
        "chosen_promotion_idx": best["promo_idx"],
        "chosen_train_acc": best["train_acc_greedy"],
        "test_pass_at_k": eval_report["pass_at_k"],
        "n_test": eval_report["n_test"],
        "n_samples": eval_report["n_samples"],
        "temperature": eval_report["temperature"],
        "max_tokens": eval_report["max_tokens"],
        "ks": eval_report["ks"],
    }
    out_path = Path(args.results_root) / model_short / run_name / "promoted_student.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[ps] wrote {out_path}")
    print("[ps] test pass@k = " + ", ".join(f"{k}:{v}" for k, v in result["test_pass_at_k"].items()))


if __name__ == "__main__":
    main()
