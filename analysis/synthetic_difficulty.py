"""Line plot of mean synthetic vs original-task difficulty (list length) per SGS iteration.

--task (sorting default, multiplication, lcm, preorder) sets the prompt / verifier for
the base-model pass@5 and the difficulty level (list length / operand digits / nodes).

Reads `<run>/synthetic.jsonl` (one row per parsed synthetic problem:
iter, unsolved, synthetic) and plots mean len(synthetic) and mean len(unsolved) (the original task the
synthetic problem was generated for, in red) vs iter. Also runs base-model
(--model) pass@5 on every distinct list via sgs.eval_sort (same prompt,
verifier and sampling) and plots mean pass@5 vs iter the same way.

Outputs (under --out-dir):
    synthetic_difficulty.pdf / .png
    synthetic_difficulty.csv   per-(source, iter) n, mean length, mean pass@5
    pass5.pdf / .png
    pass5_samples.jsonl        eval_sort rows (completions, correctness) per distinct list

Usage:
    python -m analysis.synthetic_difficulty \
        --run /n/netscratch/dam_lab/Lab/nnayak1/sgs/runs/<sgs run> [--model Qwen/Qwen3-4B]
    python -m analysis.synthetic_difficulty --task lcm --run <lcm sgs run> \
        --out-dir files/results/sgs/synthetic_difficulty_lcm/<run name>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM  # noqa: E402

from sgs import eval_sort  # noqa: E402
from sgs.tasks import TASKS  # noqa: E402


def key(problem):
    """Hashable problem; nested lists (preorder) -> nested tuples, 1.0 == 1 as before."""
    return tuple(map(key, problem)) if isinstance(problem, list) else problem


def base_pass(lists, args, out_dir):
    """{key(problem): pass@n} from eval_sort.evaluate on each distinct problem."""
    ev = eval_sort.parse_args(["--checkpoints", args.model, "--task", args.task,
                               "--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    uniq = {}
    for x in lists:
        uniq.setdefault(key(x), x)
    uniq = list(uniq.values())
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=ev.max_model_len,
              gpu_memory_utilization=ev.gpu_memory_utilization,
              enable_prefix_caching=True, disable_log_stats=True, seed=ev.seed)
    m, rows = eval_sort.evaluate(llm, tok, uniq, ev)
    print(f"[eval] {args.model} on {len(uniq)} distinct lists: pass@{ev.n}={m[f'pass@{ev.n}/all']:.3f}")
    with open(out_dir / "pass5_samples.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return {key(r["list"]): float(any(r["correct"])) for r in rows}


def lineplot(df, y, ylabel, path):
    """Mean `y` vs iter per source (± 95% CI), original in red; writes .pdf and .png."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df, x="iter", y=y, hue="source", marker="o", ax=ax,
                 palette={"synthetic": "tab:blue", "original": "tab:red"})
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.legend(title=None)
    ax.set_xticks(sorted(df["iter"].unique()))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True, help="sgs out_dir")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-4B", help="base model for pass@5")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--task", choices=sorted(TASKS), default="sorting")
    args = ap.parse_args()
    task = TASKS[args.task]

    out_dir = args.out_dir or Path("files/results/sgs/synthetic_difficulty") / args.run.name
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("json", data_files=str(args.run / "synthetic.jsonl"), split="train")
    df = pd.concat([
        pd.DataFrame({"iter": ds["iter"], "source": "synthetic", "list": ds["synthetic"]}),
        pd.DataFrame({"iter": ds["iter"], "source": "original", "list": ds["unsolved"]}),
    ])
    df["length"] = df["list"].map(task.level)
    passed = base_pass(df["list"], args, out_dir)
    df["pass5"] = df["list"].map(lambda x: passed[key(x)])
    df.groupby(["source", "iter"]).agg(count=("length", "size"), length=("length", "mean"),
                                       pass5=("pass5", "mean")).to_csv(
        out_dir / "synthetic_difficulty.csv")

    sns.set_theme(style="whitegrid")
    lineplot(df, "length", "Mean list length" if args.task == "sorting"
             else f"Mean {task.level_name[0].lower()}", out_dir / "synthetic_difficulty")
    lineplot(df, "pass5", f"Base pass@5 ({args.model.split('/')[-1]})", out_dir / "pass5")
    print(f"[plot] wrote {out_dir}")


if __name__ == "__main__":
    main()
