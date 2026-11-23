"""Histogram of list lengths in each sorting_sgs_onpolicy_5iters_synth train file.

One subplot per file; lengths binned 2-10, 11-20, ..., 121-128. Lengths outside
[2, 128] are dropped as outliers (title shows visualized/total).

Usage:
    python -m analysis.synth_length_hist
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

EDGES = [2, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 129]  # left-inclusive bin starts
# file stem -> train lengths, in plot order (sorting_1000 is sort_curves.py's "all" config)
TRAIN_LENGTHS = {"sorting_1000_L8-128": "8, 128", "sorting_1000_L8-16-128": "8, 16, 128",
                 "sorting_1000_L8-16-32-128": "8, 16, 32, 128", "sorting_1000": "8, 16, 32, 64, 128"}
LABELS = ["2-10"] + [f"{a}-{a + 9}" for a in range(11, 121, 10)] + ["121-128"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=Path("files/data/sorting_sgs_onpolicy_5iters_synth"))
    ap.add_argument("--out-dir", type=Path, default=Path("files/results/analysis/synth_length_hist"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    files = [args.data_dir / f"{n}.jsonl" for n in TRAIN_LENGTHS]
    fig, axes = plt.subplots(1, len(files), figsize=(5 * len(files), 4), sharey=True)
    for ax, f in zip(axes, files):
        lens = np.array([len(json.loads(line)["list"]) for line in open(f)])
        kept = lens[(lens >= EDGES[0]) & (lens < EDGES[-1])]
        counts, _ = np.histogram(kept, bins=EDGES)
        sns.barplot(x=LABELS, y=counts, ax=ax, color="C0")
        ax.set_title(f"Train Lengths\n{TRAIN_LENGTHS[f.stem]}\n{len(kept)}/{len(lens)} shown ({len(lens) - len(kept)} outliers removed)",
                     fontsize=10)
        ax.set_xlabel("list length")
        ax.tick_params(axis="x", rotation=60)
    axes[0].set_ylabel("count")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"synth_length_hist.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
