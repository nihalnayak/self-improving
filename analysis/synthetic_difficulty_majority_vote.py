"""Base pass@5 on synthetic lists per SGS iteration: gold verifier vs majority-vote thresholds.

One panel per training dataset (L8-16-32-128, all); one line per run (gold
verifier, MV 0.25/0.5/0.75/1.0). Reads `<run>/synthetic.jsonl` and scores every
distinct synthetic list with base-model pass@5 via
analysis.synthetic_difficulty.base_pass. Runs without synthetic.jsonl yet are
skipped.

Outputs (under --out-dir):
    pass5_mv.pdf / .png
    synthetic_difficulty_mv.csv   per-(dataset, method, iter) n, mean pass@5
    pass5_samples.jsonl           eval_sort rows per distinct list

Usage:
    python -m analysis.synthetic_difficulty_majority_vote [--model Qwen/Qwen3-4B]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from datasets import load_dataset  # noqa: E402

from analysis.synthetic_difficulty import base_pass  # noqa: E402

# (dataset, method) -> slurm job id of the sgs-onpolicy-grpo 5-iter optim-reset run
RUNS = {
    ("L8-16-32-128", "Gold verifier"): "60150236",
    ("L8-16-32-128", "MV 0.25"): "61157819_0",
    ("L8-16-32-128", "MV 0.5"): "61157819_1",
    ("L8-16-32-128", "MV 0.75"): "61157819_2",
    ("L8-16-32-128", "MV 1.0"): "61157819_3",
    ("all", "Gold verifier"): "60150258",
    ("all", "MV 0.25"): "61157819_4",
    ("all", "MV 0.5"): "61157819_5",
    ("all", "MV 0.75"): "61157819_6",
    ("all", "MV 1.0"): "61157819_7",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", type=Path, default=Path("/n/netscratch/dam_lab/Lab/nnayak1/sgs/runs"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("files/results/sgs/synthetic_difficulty/majority_vote"))
    ap.add_argument("--model", default="Qwen/Qwen3-4B", help="base model for pass@5")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for (dataset, method), job in RUNS.items():
        (run,) = args.runs_root.glob(f"*-j{job}")
        if not (run / "synthetic.jsonl").exists():
            print(f"[skip] {dataset} / {method}: no synthetic.jsonl in {run.name}")
            continue
        ds = load_dataset("json", data_files=str(run / "synthetic.jsonl"), split="train")
        print(f"[load] {dataset} / {method}: {len(ds)} rows, iters {sorted(set(ds['iter']))}")
        dfs.append(pd.DataFrame({"dataset": dataset, "method": method,
                                 "iter": ds["iter"], "list": ds["synthetic"]}))
    df = pd.concat(dfs)

    passed = base_pass(df["list"], args, args.out_dir)
    df["pass5"] = df["list"].map(lambda x: passed[tuple(x)])
    df.groupby(["dataset", "method", "iter"]).agg(count=("pass5", "size"), pass5=("pass5", "mean")).to_csv(
        args.out_dir / "synthetic_difficulty_mv.csv")

    sns.set_theme(style="whitegrid")
    datasets = list(dict.fromkeys(d for d, _ in RUNS))
    methods = list(dict.fromkeys(m for _, m in RUNS))
    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 5), sharey=True)
    for ax, dataset in zip(axes, datasets):
        sns.lineplot(data=df[df["dataset"] == dataset], x="iter", y="pass5", hue="method",
                     hue_order=methods, errorbar=None, marker="o", ax=ax)
        ax.set_title(dataset)
        ax.set_xlabel("Iteration")
        ax.set_xticks(sorted(df["iter"].unique()))
        ax.get_legend().remove()
    axes[0].set_ylabel(f"Base pass@5 ({args.model.split('/')[-1]})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"pass5_mv.{ext}", bbox_inches="tight")
    print(f"[plot] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
