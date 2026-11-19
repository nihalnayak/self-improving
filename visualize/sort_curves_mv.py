"""Per-iter pass@5 curves for SGS with the gold verifier vs majority-vote (MV) variants.

One row per train config, one column per list length. Methods (all SGS-OnPolicy, so
curves come from each run's eval_pass5.jsonl; see sort_curves.py):
    SGS (Gold)                     sgs-onpolicy-grpo-reset-<config>
    SGS (MV≥r, Percentile)         sgs-onpolicy-grpo-mv<r>-reset-<config>
    SGS (MV≥r, R-Zero)             sgs-onpolicy-grpo-mv<r>-rzero-reset-<config>
    SGS (MV≥r, Learnability)       sgs-onpolicy-grpo-mv<r>-learnability-reset-<config>
A config row shows whichever of these labels exist in --results-dir. Base model
pass@5 is the dotted line. --metric pass1 plots pass@1 instead (see sort_curves.py);
outputs get a `_pass1` suffix.

Writes to --results-dir:
    sort_curves_mv<r>_{train,val}.pdf / .png
    sort_curves_mv<r>.csv                 split, config, method, epoch, length, pass

Usage:
    python -m visualize.sort_curves_mv --results-dir files/results/sgs/eval_sorting_qwen3-4b_5ep
    python -m visualize.sort_curves_mv --results-dir files/results/sgs/eval_sorting_qwen3-4b_5ep --metric pass1
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from visualize.sort_curves import FONT_SIZE, LENGTHS, base_pass, run_dir, sgs_rows  # noqa: E402

CONFIGS = ("L8-128", "L8-16-128", "L8-16-32-128", "all")
_TAB10 = sns.color_palette("tab10", 10)


def methods(mv_ratio):
    """label prefix -> (display name, color); SGS-OnPolicy keeps its sort_curves.py color."""
    return {"sgs-onpolicy-grpo-reset": ("SGS (Gold)", _TAB10[4]),
            f"sgs-onpolicy-grpo-mv{mv_ratio}-reset": (f"SGS (MV≥{mv_ratio}, Percentile)", _TAB10[0]),
            f"sgs-onpolicy-grpo-mv{mv_ratio}-rzero-reset": (f"SGS (MV≥{mv_ratio}, R-Zero)", _TAB10[2]),
            f"sgs-onpolicy-grpo-mv{mv_ratio}-learnability-reset": (f"SGS (MV≥{mv_ratio}, Learnability)", _TAB10[3])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, default=Path("/n/netscratch/dam_lab/Lab/nnayak1/sgs/runs"))
    ap.add_argument("--mv-ratio", default="0.75")
    ap.add_argument("--metric", choices=("pass5", "pass1"), default="pass5")
    args = ap.parse_args()
    suffix = "" if args.metric == "pass5" else f"_{args.metric}"
    ylabel = "Pass@5" if args.metric == "pass5" else "Pass@1"

    METHODS = methods(args.mv_ratio)
    labels = {p.name for p in args.results_dir.iterdir() if p.is_dir()}
    configs = [c for c in CONFIGS if any(f"{m}-{c}" in labels for m in METHODS)]

    rows = []
    for config in configs:
        for m, (name, _) in METHODS.items():
            if f"{m}-{config}" not in labels:
                continue
            run = run_dir(args.results_dir, f"{m}-{config}", args.runs_root)
            rows += [{"config": config, "method": name, **r} for r in sgs_rows(run, args.metric)]
    df = pd.DataFrame(rows).sort_values(["split", "config", "method", "length", "epoch"])
    df.to_csv(args.results_dir / f"sort_curves_mv{args.mv_ratio}{suffix}.csv", index=False)

    sns.set_theme(style="whitegrid", rc={"font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE,
                                         "axes.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE,
                                         "xtick.labelsize": FONT_SIZE - 4, "ytick.labelsize": FONT_SIZE - 4})
    palette = {name: color for name, color in METHODS.values()}
    for split in ("train", "val"):
        fig, axes = plt.subplots(len(configs), len(LENGTHS), figsize=(4.5 * len(LENGTHS), 4 * len(configs)),
                                 sharex=True, sharey=True, squeeze=False)
        titled = set()  # column title on the first non-empty panel (train skips lengths)
        for i, config in enumerate(configs):
            base = base_pass(args.results_dir, config, args.metric)
            for j, length in enumerate(LENGTHS):
                ax = axes[i][j]
                d = df[(df["split"] == split) & (df["config"] == config) & (df["length"] == length)]
                if d.empty:  # length not in this train config
                    ax.axis("off")
                    continue
                sns.lineplot(data=d, x="epoch", y="pass", hue="method", palette=palette,
                             marker="o", markersize=10, linewidth=2.5, ax=ax, legend=False)
                ax.axhline(base[split, length], ls=":", color="gray", lw=2)
                ax.set_ylim(-0.03, 1.03)
                ax.set_xticks(range(6))
                if length not in titled:
                    ax.set_title(f"Length {length}")
                    titled.add(length)
                ax.set_xlabel("Iteration" if i == len(configs) - 1 else "")
                ax.set_ylabel(f"{split.capitalize()} {ylabel}" if j == 0 else "")
                ax.tick_params(labelbottom=True, labelleft=True)
            train_lengths = LENGTHS if config == "all" else config[1:].split("-")
            axes[i][0].annotate("Train Lengths\n" + ", ".join(train_lengths), xy=(-0.42, 0.5),
                                xycoords="axes fraction", rotation=90, ha="right", va="center", multialignment="center",
                                fontsize=FONT_SIZE)
        handles = [plt.Line2D([], [], color=c, marker="o", markersize=10, lw=2.5, label=n) for n, c in METHODS.values()]
        handles.append(plt.Line2D([], [], color="gray", ls=":", lw=2.5, label="Base"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(args.results_dir / f"sort_curves_mv{args.mv_ratio}{suffix}_{split}.{ext}", dpi=300,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {configs} -> {args.results_dir}/sort_curves_mv{args.mv_ratio}{suffix}_{split}.{{pdf,png}}")


if __name__ == "__main__":
    main()
