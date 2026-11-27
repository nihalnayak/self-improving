"""Per-epoch pass@5 curves for GRPO vs SGS-OnPolicy-GRPO, one row per train config.

For each `grpo-<config>` / `sgs-onpolicy-grpo-reset-<config>` label in --results-dir
(the `sgs.eval_sort` dir; its `<label>/<run_name>/` subdir names the training run),
reads the evals the training runs already wrote:
    SGS:  <run>/eval_pass5.jsonl            per iter; eval/ = train targets, val/ = val
    GRPO: <run>/eval_samples/{step}.jsonl   verl val generations (5 per list, train + val
                                            targets); pass@5 = any correct, per length
Optional grpo-1.5x-<config> and grpo-2x-<config> labels add scaled-data GRPO curves.
--sgs-synth (off by default) adds grpo-sgs-synth-<config> (GRPO on SGS synthetic data), val
only: its train set is synthetic, with arbitrary lengths. Outputs get a `_sgs_synth` suffix.
Each GRPO run uses its own training dataset for train curves and its own steps per epoch.
Base model pass@5 (dotted line) is the final-eval row of `base-<config>/summary.jsonl`.
--steps: x = Solver optimizer step instead of epoch, for step-matched runs (SGS run with
--max-solver-steps --eval-every-steps); adds sgs-onpolicy-grpo-<config> (no optimizer reset) and
sgs-onpolicy-grpo-prefilter-<config> (sgs.sgs_onpolicy_grpo_prefilter).
Outputs get a `_steps` suffix.

--task (sorting default, multiplication, lcm, preorder) sets the level (list length /
operand digits / nodes) and the axis wording; labels are `<method>-<config>` with config
`L<level>-<level>...` or `all`. GRPO val generations are assigned to train / val and to a
level by matching their prompt against the run's own `args.json` dataset / val_dataset.

--metric pass1 plots pass@1 instead (mean correctness over all samples), from the
per-sample files: SGS <run>/eval_samples/iter_*.jsonl, GRPO <run>/eval_samples/{step}.jsonl,
base base-<config>/<ckpt>/{train,val}_samples.jsonl. Outputs get a `_pass1` suffix.

Writes to --results-dir:
    sort_curves_{train,val}.pdf / .png   rows = train config, cols = level (list length),
                                         x = epoch (GRPO) / iter (SGS), y = pass@5
    sort_curves.csv                      split, config, method, epoch, length, pass

Usage:
    python -m visualize.sort_curves --results-dir files/results/sgs/eval_sorting_qwen3-4b_5ep
    python -m visualize.sort_curves --results-dir files/results/sgs/eval_sorting_qwen3-4b_5ep --metric pass1
    python -m visualize.sort_curves --results-dir files/results/sgs/eval_sorting_qwen3-4b_5ep --sgs-synth
    python -m visualize.sort_curves --task lcm --results-dir files/results/sgs/eval_lcm_qwen3-4b_5ep
    python -m visualize.sort_curves --results-dir files/results/sgs/sgs_steps_match_sorting_qwen3-4b --steps
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

from sgs.tasks import TASKS  # noqa: E402

FONT_SIZE = 22
# same colors as sort_eval.py's tab10 METHOD_ORDER palette
_TAB10 = sns.color_palette("tab10", 10)
METHODS = {"grpo": ("GRPO", _TAB10[1]), "sgs-onpolicy-grpo-reset": ("SGS-OnPolicy-GRPO", _TAB10[4]),
           "grpo-1.5x": ("GRPO (1.5×)", _TAB10[2]), "grpo-2x": ("GRPO (2×)", _TAB10[3])}
SGS_SYNTH = {"grpo-sgs-synth": ("GRPO (SGS data)", _TAB10[5])}
STEPS = {"sgs-onpolicy-grpo": ("SGS-OnPolicy-GRPO (no reset)", _TAB10[6]),
         "sgs-onpolicy-grpo-prefilter": ("SGS-OnPolicy-GRPO (pre-filter)", _TAB10[7])}
CONFIG_RE = re.compile(r"^grpo-(?:1\.5x-|2x-)?(L\d+(?:-\d+)*|all)$")


def available_configs(labels):
    return sorted({m[1] for m in map(CONFIG_RE.match, labels) if m},
                  key=lambda c: (c == "all", len(c.split("-")), c))


def run_dir(results_dir, label, runs_root):
    (run,) = [p.name for p in (results_dir / label).iterdir() if p.is_dir()]
    return runs_root / run


def sgs_rows(run, task, metric="pass5", steps=False):
    """steps: x = Solver step from a --max-solver-steps run (step_*.jsonl, eval_pass5 "step")."""
    if metric == "pass1":
        for path in sorted((run / "eval_samples").glob(f"{'step' if steps else 'iter'}_*.jsonl")):
            by = defaultdict(list)
            for line in path.read_text().splitlines():
                r = json.loads(line)
                by["train" if r["split"] == "eval" else "val", str(task.level(r["list"]))].append(r["correct"])
            for (split, length), cs in by.items():
                yield {"split": split, "epoch": int(path.stem.split("_")[1]), "length": length,
                       "pass": sum(cs) / len(cs), "step": int(path.stem.split("_")[1]) if steps else None}
        return
    for line in (run / "eval_pass5.jsonl").read_text().splitlines():
        r = json.loads(line)
        if steps and "step" not in r:   # per-iteration eval, not a Solver-step eval
            continue
        for k, v in r.items():
            if k.startswith(("eval/pass@5/len_", "val/pass@5/len_")):
                split = "train" if k.startswith("eval/") else "val"
                yield {"split": split, "epoch": r["iter"], "length": k.split("len_")[1], "pass": v,
                       "step": r.get("step")}


def user_content(task, p):
    return task.solver_messages(p)[0]["content"]


def grpo_rows(run, task, metric="pass5"):
    run_args = json.loads((run / "args.json").read_text())
    targets = task.load(run_args["dataset"])
    steps_per_epoch = len(targets) // run_args["train_batch_size"]
    # Solver prompt -> (split, level); val wins if a problem is in both files
    split_of = {user_content(task, p): ("train", str(task.level(p))) for p in targets}
    split_of.update({user_content(task, p): ("val", str(task.level(p))) for p in task.load(run_args["val_dataset"])})
    for path in (run / "eval_samples").glob("*.jsonl"):
        if not path.stem.isdigit():
            continue
        correct = defaultdict(list)  # prompt -> correctness of each of its samples
        for line in path.read_text().splitlines():
            r = json.loads(line)
            # verl decodes the chat prompt as "user\n<content>\nassistant\n..."
            content = r["input"].removeprefix("user\n").split("\nassistant\n")[0]
            correct[content].append(r["acc"] > 0.5)
        by = defaultdict(list)
        for content, c in correct.items():
            if content not in split_of:  # verl left-truncates prompts longer than prompt_length
                continue
            # pass@5: any sample correct; pass@1: every sample counts
            by[split_of[content]] += c if metric == "pass1" else [any(c)]
        for (split, length), cs in by.items():
            yield {"split": split, "epoch": int(path.stem) / steps_per_epoch, "length": length,
                   "pass": sum(cs) / len(cs), "step": int(path.stem)}


def base_pass(results_dir, config, task, metric="pass5"):
    """{(split, level): pass} of base-<config>; {} until that eval has finished."""
    out = {}
    if not (results_dir / f"base-{config}" / "summary.jsonl").exists():
        return out
    if metric == "pass1":
        for split in ("train", "val"):
            paths = list((results_dir / f"base-{config}").glob(f"*/{split}_samples.jsonl"))
            if not paths:  # this split not evaluated yet
                continue
            (path,) = paths
            by = defaultdict(list)
            for line in path.read_text().splitlines():
                r = json.loads(line)
                by[str(task.level(r["list"]))] += r["correct"]
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
    ap.add_argument("--task", choices=sorted(TASKS), default="sorting")
    ap.add_argument("--sgs-synth", action="store_true", help="add GRPO-on-SGS-data val curves")
    ap.add_argument("--steps", action="store_true", help="x = Solver step (step-matched runs)")
    args = ap.parse_args()
    task = TASKS[args.task]
    level_one, level_many = task.level_name
    suffix = (("" if args.metric == "pass5" else f"_{args.metric}") + ("_sgs_synth" if args.sgs_synth else "")
              + ("_steps" if args.steps else ""))
    methods = {**METHODS, **(SGS_SYNTH if args.sgs_synth else {}), **(STEPS if args.steps else {})}
    x = "step" if args.steps else "epoch"
    ylabel = "Pass@5" if args.metric == "pass5" else "Pass@1"

    labels = {p.name for p in args.results_dir.iterdir() if p.is_dir()}
    configs = available_configs(labels)
    if not configs:
        print(f"[plot] no GRPO configs under {args.results_dir}")
        return

    rows = []
    for config in configs:
        for m, (name, _) in methods.items():
            if f"{m}-{config}" not in labels:
                continue
            run = run_dir(args.results_dir, f"{m}-{config}", args.runs_root)
            it = (sgs_rows(run, task, args.metric, args.steps) if m.startswith("sgs")
                  else grpo_rows(run, task, args.metric))
            rows += [{"config": config, "method": name, **r} for r in it
                     if m != "grpo-sgs-synth" or r["split"] == "val"]
    df = pd.DataFrame(rows).sort_values(["split", "config", "method", "length", x])
    df.to_csv(args.results_dir / f"sort_curves{suffix}.csv", index=False)

    sns.set_theme(style="whitegrid", rc={"font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE,
                                         "axes.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE,
                                         "xtick.labelsize": FONT_SIZE - 4, "ytick.labelsize": FONT_SIZE - 4})
    palette = {name: color for name, color in methods.values()}
    lengths = sorted(set(df["length"]), key=int)
    xticks = range(0, int(df[x].max()) + 1, 20 if args.steps else 1)
    for split in ("train", "val"):
        fig, axes = plt.subplots(len(configs), len(lengths), figsize=(4.5 * len(lengths), 4 * len(configs)),
                                 sharex=True, sharey=True, squeeze=False)
        titled = set()  # column title on the first non-empty panel (train skips lengths)
        for i, config in enumerate(configs):
            base = base_pass(args.results_dir, config, task, args.metric)
            for j, length in enumerate(lengths):
                ax = axes[i][j]
                d = df[(df["split"] == split) & (df["config"] == config) & (df["length"] == length)]
                if d.empty:  # length not in this train config
                    ax.axis("off")
                    continue
                sns.lineplot(data=d, x=x, y="pass", hue="method", palette=palette,
                             marker="o", markersize=10, linewidth=2.5, ax=ax, legend=False)
                if (split, length) in base:
                    ax.axhline(base[split, length], ls=":", color="gray", lw=2)
                ax.set_ylim(-0.03, 1.03)
                ax.set_xticks(xticks)
                if length not in titled:
                    ax.set_title(f"{level_one} {length}")
                    titled.add(length)
                xlabel = "Solver step" if args.steps else "Epoch / Iteration"
                ax.set_xlabel(xlabel if i == len(configs) - 1 else "")
                ax.set_ylabel(f"{split.capitalize()} {ylabel}" if j == 0 else "")
                ax.tick_params(labelbottom=True, labelleft=True)
            # row label left of the Pass@5 axis label: the lengths this row trained on
            train_lengths = lengths if config == "all" else config[1:].split("-")
            axes[i][0].annotate(f"Train {level_many}\n" + ", ".join(train_lengths), xy=(-0.42, 0.5),
                                xycoords="axes fraction", rotation=90, ha="right", va="center", multialignment="center",
                                fontsize=FONT_SIZE)
        handles = [plt.Line2D([], [], color=c, marker="o", markersize=10, lw=2.5, label=n)
                   for n, c in methods.values() if n in set(df["method"])]
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
