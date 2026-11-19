"""Per-epoch pass@5 curves for GRPO vs SGS-OnPolicy-GRPO, one row per train config.

For each `grpo-<config>` / `sgs-onpolicy-grpo-reset-<config>` label in --results-dir
(the `sgs.eval_sort` dir; its `<label>/<run_name>/` subdir names the training run),
reads the evals the training runs already wrote:
    SGS:  <run>/eval_pass5.jsonl            per iter; eval/ = train targets, val/ = val
    GRPO: <run>/eval_samples/{step}.jsonl   verl val generations (5 per list, train + val
                                            targets); pass@5 = any correct, per length
Base model pass@5 (dotted line) is the final-eval row of `base-<config>/summary.jsonl`.

--metric pass1 plots pass@1 instead (mean correctness over all samples), from the
per-sample files: SGS <run>/eval_samples/iter_*.jsonl, GRPO <run>/eval_samples/{step}.jsonl,
base base-<config>/<ckpt>/{train,val}_samples.jsonl. Outputs get a `_pass1` suffix.

Writes to --results-dir:
    sort_curves_{train,val}.pdf / .png   rows = train config, cols = list length,
                                         x = epoch (GRPO) / iter (SGS), y = pass@5
    sort_curves.csv                      split, config, method, epoch, length, pass

Usage:
    python -m visualize.sort_curves --results-dir files/results/sgs/eval_sorting_qwen3-4b_5ep
    python -m visualize.sort_curves --results-dir files/results/sgs/eval_sorting_qwen3-4b_5ep --metric pass1
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from datasets import load_dataset  # noqa: E402

FONT_SIZE = 22
LENGTHS = ["8", "16", "32", "64", "128"]
VAL_DATASET = "files/data/sorting/sorting_val_100.jsonl"
# same colors as sort_eval.py's tab10 METHOD_ORDER palette
_TAB10 = sns.color_palette("tab10", 10)
METHODS = {"grpo": ("GRPO", _TAB10[1]), "sgs-onpolicy-grpo-reset": ("SGS-OnPolicy-GRPO", _TAB10[4])}
LIST_RE = re.compile(r"List: (\[[^\]]*\])")


def run_dir(results_dir, label, runs_root):
    (run,) = [p.name for p in (results_dir / label).iterdir() if p.is_dir()]
    return runs_root / run


def sgs_rows(run, metric="pass5"):
    if metric == "pass1":
        for path in sorted((run / "eval_samples").glob("iter_*.jsonl")):
            by = defaultdict(list)
            for line in path.read_text().splitlines():
                r = json.loads(line)
                by["train" if r["split"] == "eval" else "val", str(len(r["list"]))].append(r["correct"])
            for (split, length), cs in by.items():
                yield {"split": split, "epoch": int(path.stem.split("_")[1]), "length": length,
                       "pass": sum(cs) / len(cs)}
        return
    for line in (run / "eval_pass5.jsonl").read_text().splitlines():
        r = json.loads(line)
        for k, v in r.items():
            if k.startswith(("eval/pass@5/len_", "val/pass@5/len_")):
                split = "train" if k.startswith("eval/") else "val"
                yield {"split": split, "epoch": r["iter"], "length": k.split("len_")[1], "pass": v}


def grpo_rows(run, val_lists, metric="pass5"):
    eval_every = json.loads((run / "args.json").read_text())["eval_every"]
    for path in (run / "eval_samples").glob("*.jsonl"):
        if not path.stem.isdigit():
            continue
        correct = defaultdict(list)  # list -> correctness of each of its samples
        for line in path.read_text().splitlines():
            r = json.loads(line)
            xs = tuple(json.loads(LIST_RE.search(r["input"]).group(1)))
            correct[xs].append(r["acc"] > 0.5)
        by = defaultdict(list)
        for xs, c in correct.items():
            # pass@5: any sample correct; pass@1: every sample counts
            by["val" if xs in val_lists else "train", str(len(xs))] += c if metric == "pass1" else [any(c)]
        for (split, length), cs in by.items():
            yield {"split": split, "epoch": int(path.stem) / eval_every, "length": length,
                   "pass": sum(cs) / len(cs)}


def base_pass(results_dir, config, metric="pass5"):
    out = {}
    if metric == "pass1":
        for split in ("train", "val"):
            (path,) = (results_dir / f"base-{config}").glob(f"*/{split}_samples.jsonl")
            by = defaultdict(list)
            for line in path.read_text().splitlines():
                r = json.loads(line)
                by[str(len(r["list"]))] += r["correct"]
            for length, cs in by.items():
                out[split, length] = sum(cs) / len(cs)
        return out
    for line in (results_dir / f"base-{config}" / "summary.jsonl").read_text().splitlines():
        r = json.loads(line)
        for k, v in r.items():
            if "/pass@5/len_" in k:
                out[r["split"], k.split("len_")[1]] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, default=Path("/n/netscratch/dam_lab/Lab/nnayak1/sgs/runs"))
    ap.add_argument("--metric", choices=("pass5", "pass1"), default="pass5")
    args = ap.parse_args()
    suffix = "" if args.metric == "pass5" else f"_{args.metric}"
    ylabel = "Pass@5" if args.metric == "pass5" else "Pass@1"

    val_lists = {tuple(r["list"]) for r in load_dataset("json", data_files=VAL_DATASET, split="train")}
    labels = {p.name for p in args.results_dir.iterdir() if p.is_dir()}
    configs = [c for c in ("L8-128", "L8-16-128", "L8-16-32-128", "all")
               if all(f"{m}-{c}" in labels for m in METHODS)]

    rows = []
    for config in configs:
        for m, (name, _) in METHODS.items():
            run = run_dir(args.results_dir, f"{m}-{config}", args.runs_root)
            it = sgs_rows(run, args.metric) if m.startswith("sgs") else grpo_rows(run, val_lists, args.metric)
            rows += [{"config": config, "method": name, **r} for r in it]
    df = pd.DataFrame(rows).sort_values(["split", "config", "method", "length", "epoch"])
    df.to_csv(args.results_dir / f"sort_curves{suffix}.csv", index=False)

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
                ax.set_xlabel("Epoch / Iteration" if i == len(configs) - 1 else "")
                ax.set_ylabel(f"{split.capitalize()} {ylabel}" if j == 0 else "")
                ax.tick_params(labelbottom=True, labelleft=True)
            # row label left of the Pass@5 axis label: the lengths this row trained on
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
            fig.savefig(args.results_dir / f"sort_curves{suffix}_{split}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {configs} -> {args.results_dir}/sort_curves{suffix}_{split}.{{pdf,png}}")


if __name__ == "__main__":
    main()
