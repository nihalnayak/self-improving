"""Print model / pass@K / n_samples for every fail@K dataset in the sweep grid.

The (model, K) grid mirrors `files/scripts/misc/run_build_fail128.sh`. Combos
whose `meta.json` is missing are reported as `(not built)`.

Usage:
    python -m misc.stats_fail128
    python -m misc.stats_fail128 --root /n/netscratch/dam_lab/Lab/nnayak1/soar/data/math_fail
"""

import argparse
import json
from pathlib import Path


DEFAULT_ROOT = "/n/netscratch/dam_lab/Lab/nnayak1/soar/data/math_fail"

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]
K_LIST = [32, 64, 128]


def dataset_dir(root: Path, model: str, k: int) -> Path:
    return root / model.split("/")[-1].lower() / f"pass{k}"


def n_samples(meta_path: Path) -> int | None:
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    return int(meta.get("n_fail", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="Parent dir holding <model_short>/pass<K>/meta.json trees.")
    args = ap.parse_args()
    root = Path(args.root)

    cols = ["model", "pass@K", "n_samples"]
    rows = []
    for model in MODELS:
        for k in K_LIST:
            n = n_samples(dataset_dir(root, model, k) / "meta.json")
            rows.append([model, f"pass@{k}", str(n) if n is not None else "(not built)"])

    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    print("  ".join(f"{c:<{widths[i]}}" for i, c in enumerate(cols)))
    print("  ".join("-" * widths[i] for i in range(len(cols))))
    for r in rows:
        print("  ".join(f"{r[i]:<{widths[i]}}" for i in range(len(cols))))


if __name__ == "__main__":
    main()
