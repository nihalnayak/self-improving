"""Difficulty and length distribution of SGS synthetic sorting tasks per iteration.

For every 5-iter SGS run under <results-dir> (written by
`analysis.synthetic_difficulty`), joins the run's `synthetic.jsonl` (iter,
synthetic list, unsolved original target) with the base-model
`pass5_samples.jsonl` (5 completions per distinct list) and writes to <results-dir>:
    sgs_synth.pdf / .png   one column per train config
                           row 1: base solve rate (mean correct of 5) per iter,
                                  synthetic vs original (as analysis.synthetic_difficulty)
                           row 2: % of synthetic lists per iter, stacked by list
                                  length (log-scale color, shared colorbar)
    sgs_synth.csv          config, source, iter, length, solve_rate (one row per list)

Usage:
    python -m visualize.sgs_synth --results-dir files/results/sgs/synthetic_difficulty
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from datasets import load_dataset  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

FONT_SIZE = 16
RUNS = Path("/n/netscratch/dam_lab/Lab/nnayak1/sgs/runs")
RUN_RE = re.compile(r"-sorting-(?P<config>L\d+(?:-\d+)*|all)-5iters-")


def load(results_dir):
    rows = []
    for run_dir in sorted(results_dir.glob("*-5iters-*")):
        config = RUN_RE.search(run_dir.name)["config"]
        rate = {}
        for line in (run_dir / "pass5_samples.jsonl").read_text().splitlines():
            r = json.loads(line)
            rate[tuple(r["list"])] = float(np.mean(r["correct"]))
        ds = load_dataset("json", data_files=str(RUNS / run_dir.name / "synthetic.jsonl"),
                          split="train")
        for source, col in (("synthetic", "synthetic"), ("original", "unsolved")):
            rows += [{"config": config, "source": source, "iter": it, "length": len(xs),
                      "solve_rate": rate[tuple(xs)]} for it, xs in zip(ds["iter"], ds[col])]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, required=True)
    args = ap.parse_args()

    df = load(args.results_dir)
    if df.empty:
        print(f"[plot] no 5-iter runs under {args.results_dir}")
        return
    df.to_csv(args.results_dir / "sgs_synth.csv", index=False)

    configs = sorted(set(df["config"]), key=lambda c: (c == "all", len(c.split("-")), c))
    iters = sorted(set(df["iter"]))
    synth = df[df["source"] == "synthetic"]
    norm = LogNorm(vmin=synth["length"].min(), vmax=synth["length"].max())
    cmap = plt.get_cmap("viridis")

    sns.set_theme(style="whitegrid", rc={"font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE,
                                         "axes.labelsize": FONT_SIZE})
    fig, axes = plt.subplots(2, len(configs), figsize=(6 * len(configs), 9), sharey="row",
                             squeeze=False)
    for j, config in enumerate(configs):
        ax = axes[0][j]
        sns.lineplot(data=df[df["config"] == config], x="iter", y="solve_rate", hue="source",
                     hue_order=["synthetic", "original"], marker="o", ax=ax,
                     palette={"synthetic": "tab:blue", "original": "tab:red"})
        ax.set_xticks(iters)
        ax.set_title("train: all lengths" if config == "all"
                     else "train: lengths " + ", ".join(config[1:].split("-")))
        ax.set_xlabel("")
        ax.set_ylabel("Base Solve Rate" if j == 0 else "")
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=FONT_SIZE - 2)
        ax.get_legend().remove()

        sub = synth[synth["config"] == config]
        ax = axes[1][j]
        for x, it in enumerate(iters):
            frac = sub[sub["iter"] == it]["length"].value_counts(normalize=True).sort_index() * 100
            ax.bar([x] * len(frac), frac.values, bottom=frac.cumsum().values - frac.values,
                   color=cmap(norm(frac.index.values)), width=0.8, linewidth=0)
        ax.set_xticks(range(len(iters)), iters)
        ax.set_xlabel("SGS Iteration")
        ax.set_ylabel("% of Synthetic Tasks" if j == 0 else "")
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=FONT_SIZE - 2)
        ax.grid(False, axis="x")

    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, [n.capitalize() for n in names], loc="upper center", ncol=len(names),
               frameon=False, bbox_to_anchor=(0.47, 1.05), fontsize=FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    pos = axes[1][-1].get_position()
    cax = fig.add_axes((0.95, pos.y0, 0.012, pos.height))
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("List Length")
    for ext in ("pdf", "png"):
        fig.savefig(args.results_dir / f"sgs_synth.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {configs} x iters {iters} -> {args.results_dir}/sgs_synth.{{pdf,png}}")


if __name__ == "__main__":
    main()
