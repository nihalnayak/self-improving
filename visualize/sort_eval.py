"""Bar plot of sorting pass@n per list length for each evaluated method.

Reads `<results-dir>/<label>/summary.jsonl` written by `sgs.eval_sort` (one
subdir per method, e.g. sgs-mbs32, grpo, base), keeps the last row per
(label, split) so re-evals overwrite, and writes to <results-dir>:
    sort_eval.pdf / .png   x = list length (+ all), hue = method, one panel per split
    summary.csv            long table: label, checkpoint, split, length, pass

Usage:
    python -m visualize.sort_eval --results-dir files/results/sgs/eval_sorting_qwen3-1.7b_10ep
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

FONT_SIZE = 16
LABEL_ORDER = ["base", "grpo", "reinforce", "sgs-mbs32", "sgs-mbs256"]


def load(results_dir):
    rows = []
    for summary in sorted(results_dir.glob("*/summary.jsonl")):
        last = {}
        for line in summary.read_text().splitlines():
            r = json.loads(line)
            last[r["split"]] = r
        for split, r in last.items():
            for k, v in r.items():
                if not k.startswith(f"{split}/pass@"):
                    continue
                metric, length = k.split("/")[1:]  # pass@5, len_8 | all
                rows.append({"label": summary.parent.name, "checkpoint": r["checkpoint"],
                             "split": split, "metric": metric,
                             "length": length.removeprefix("len_"), "pass": v})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, required=True)
    args = ap.parse_args()

    df = load(args.results_dir)
    if df.empty:
        print(f"[plot] no summary.jsonl under {args.results_dir}")
        return
    df.to_csv(args.results_dir / "summary.csv", index=False)

    lengths = sorted({x for x in df["length"] if x != "all"}, key=int) + ["all"]
    labels = [x for x in LABEL_ORDER if x in set(df["label"])] + \
             sorted(set(df["label"]) - set(LABEL_ORDER))
    splits = [s for s in ("train", "val") if s in set(df["split"])] + \
             sorted(set(df["split"]) - {"train", "val"})
    metric = df["metric"].iloc[0]

    sns.set_theme(style="whitegrid", rc={"font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE,
                                         "axes.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE - 2})
    fig, axes = plt.subplots(1, len(splits), figsize=(8 * len(splits), 5.5), sharey=True, squeeze=False)
    for ax, split in zip(axes[0], splits):
        sns.barplot(data=df[df["split"] == split], x="length", y="pass", hue="label",
                    order=lengths, hue_order=labels, palette="Set2", ax=ax)
        ax.set_title(f"{split} targets")
        ax.set_xlabel("list length")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.get_legend().remove()
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=len(names), frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.results_dir / f"sort_eval.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {sorted(set(df['label']))} -> {args.results_dir}/sort_eval.{{pdf,png}}")


if __name__ == "__main__":
    main()
