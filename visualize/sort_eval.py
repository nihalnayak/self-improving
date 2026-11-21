"""Bar plot of sorting pass@n per list length for each evaluated method.

--task (sorting default, multiplication, lcm, preorder) only changes the axis / title
wording (list length / digits / nodes); summary.jsonl is the same for every task.

Reads `<results-dir>/<label>/summary.jsonl` written by `sgs.eval_sort` (one
subdir per method, e.g. sgs-mbs32, grpo, base), keeps the last row per
(label, split) so re-evals overwrite, and writes to <results-dir>:
    sort_eval.pdf / .png   x = list length (+ all), hue = method, one panel per split
    summary.csv            long table: label, checkpoint, split, length, pass
    sort_eval_val_by_config.pdf / .png
                           only when labels are `<method>-<train config>` (e.g.
                           grpo-L8-16-128, base-all): val only, one panel per
                           train config, hue = method, one shared legend

Usage:
    python -m visualize.sort_eval --results-dir files/results/sgs/eval_sorting_qwen3-1.7b_10ep
    python -m visualize.sort_eval --task preorder --results-dir files/results/sgs/eval_preorder_qwen3-4b_5ep
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from sgs.tasks import TASKS  # noqa: E402

FONT_SIZE = 16
# OPSD's teacher sees the gold sorted list: not a fair comparison, so not plotted.
EXCLUDE_PREFIXES = ("opsd-",)
LABEL_ORDER = ["base", "grpo", "reinforce", "sgs-mbs32", "sgs-mbs256"]
# by-config figure: label = <method>-<config>, config = L<len>-<len>... or all
CONFIG_RE = re.compile(r"^(?P<method>.+)-(?P<config>L\d+(?:-\d+)*|all)$")
RSOLVE_NAMES = {"learnability": "s(1−s)", "rzero": "R-Zero"}
METHOD_NAMES = {"base": "Base", "grpo": "GRPO", "grpo-sgs-synth": "GRPO (SGS Synth.)", "sgs": "SGS",
                "sgs-onpolicy-grpo-reset": "SGS-OnPolicy-GRPO",
                **{f"sgs-onpolicy-grpo-mv{r}-reset": f"SGS-OnPolicy-GRPO (MV≥{r})"
                   for r in ("0.25", "0.5", "0.75", "1.0")},
                **{f"sgs-onpolicy-grpo-mv{r}-{rs}-reset": f"SGS-OnPolicy-GRPO (MV≥{r}, {name})"
                   for r in ("0.5", "0.75") for rs, name in RSOLVE_NAMES.items()}}
METHOD_ORDER = ["Base", "GRPO", "GRPO (SGS Synth.)", "SGS", "SGS-OnPolicy-GRPO",
                *[f"SGS-OnPolicy-GRPO (MV≥{r})" for r in ("0.25", "0.5", "0.75", "1.0")],
                *[f"SGS-OnPolicy-GRPO (MV≥{r}, {name})"
                  for r in ("0.5", "0.75") for name in RSOLVE_NAMES.values()]]


def load(results_dir):
    rows = []
    for summary in sorted(results_dir.glob("*/summary.jsonl")):
        if summary.parent.name.startswith(EXCLUDE_PREFIXES):
            continue
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
    ap.add_argument("--task", choices=sorted(TASKS), default="sorting")
    args = ap.parse_args()
    level_one, level_many = TASKS[args.task].level_name
    xlabel = "List Length" if args.task == "sorting" else level_one

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
        ax.set_xlabel(xlabel.lower())
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

    # --- val only, one panel per train config, hue = method ------------------
    val = df[df["split"] == "val"].copy()
    parts = val["label"].str.extract(CONFIG_RE)
    val["method"] = parts["method"].map(lambda m: METHOD_NAMES.get(m, m))
    val["config"] = parts["config"]
    val = val.dropna(subset=["config"])
    if val.empty:
        return
    configs = sorted(set(val["config"]), key=lambda c: (c == "all", len(c.split("-")), c))
    methods = [m for m in METHOD_ORDER if m in set(val["method"])] + \
              sorted(set(val["method"]) - set(METHOD_ORDER))
    all_methods = METHOD_ORDER + sorted(set(val["method"]) - set(METHOD_ORDER))
    palette = dict(zip(all_methods, sns.color_palette("tab10", len(all_methods))))

    fig, axes = plt.subplots(1, len(configs), figsize=(7 * len(configs), 5.5), sharey=True,
                             squeeze=False)
    for ax, config in zip(axes[0], configs):
        sns.barplot(data=val[val["config"] == config], x="length", y="pass", hue="method",
                    order=lengths, hue_order=methods, palette=palette, ax=ax)
        ax.set_title(f"train: all {level_many.lower()}" if config == "all"
                     else f"train: {level_many.lower()} " + ", ".join(config[1:].split("-")))
        ax.set_xticks(range(len(lengths)), [x.capitalize() for x in lengths])
        ax.tick_params(axis="x", labelsize=FONT_SIZE)
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE + 2)
        ax.set_ylabel(f"val {metric}")
        ax.set_ylim(0, 1)
        ax.get_legend().remove()
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=min(len(names), 5), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.results_dir / f"sort_eval_val_by_config.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {methods} x {configs} -> "
          f"{args.results_dir}/sort_eval_val_by_config.{{pdf,png}}")


if __name__ == "__main__":
    main()
