"""Histogram of list lengths in each sorting_sgs_onpolicy_5iters_synth train file.

One subplot per file; one bar per length 1-128. The first 1000 rows of each file
are the original train split (red, bottom), the rest are synthetic (blue, stacked
on top). Lengths > 128 are dropped as outliers (title shows visualized/total).

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

MAX_LEN = 128
N_ORIG = 1000  # rows [:N_ORIG] are the original split, the rest are synthetic
# file stem -> train lengths, in plot order (sorting_1000 is sort_curves.py's "all" config)
TRAIN_LENGTHS = {"sorting_1000_L8-128": "8, 128", "sorting_1000_L8-16-128": "8, 16, 128",
                 "sorting_1000_L8-16-32-128": "8, 16, 32, 128", "sorting_1000": "8, 16, 32, 64, 128"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=Path("files/data/sorting_sgs_onpolicy_5iters_synth"))
    ap.add_argument("--out-dir", type=Path, default=Path("files/results/analysis/synth_length_hist"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    files = [args.data_dir / f"{n}.jsonl" for n in TRAIN_LENGTHS]
    edges = np.arange(1, MAX_LEN + 2)
    xs = edges[:-1]
    fig, axes = plt.subplots(1, len(files), figsize=(5 * len(files), 4), sharey=True)
    for ax, f in zip(axes, files):
        lens = np.array([len(json.loads(line)["list"]) for line in open(f)])
        orig, synth = lens[:N_ORIG], lens[N_ORIG:]
        n_kept = (lens <= MAX_LEN).sum()
        orig_counts, _ = np.histogram(orig, bins=edges)
        synth_counts, _ = np.histogram(synth, bins=edges)
        ax.bar(xs, orig_counts, width=1.0, color="tab:red", label="original")
        ax.bar(xs, synth_counts, width=1.0, bottom=orig_counts, color="C0", label="synthetic")
        ax.set_title(f"Train Lengths\n{TRAIN_LENGTHS[f.stem]}\n{n_kept}/{len(lens)} shown ({len(lens) - n_kept} outliers removed)",
                     fontsize=10)
        ax.set_xlim(0.5, MAX_LEN + 0.5)
        ax.set_xticks(range(8, MAX_LEN + 1, 8))
        ax.set_xlabel("list length")
        ax.tick_params(axis="x", rotation=60)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("count (log)")
    axes[0].legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"synth_length_hist.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
