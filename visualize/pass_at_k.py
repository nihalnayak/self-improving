#!/usr/bin/env python3
"""Plot pass@k curves across baselines for a given model.

Walks `<results_root>/<model_short>/<run_name>/<baseline>.json` emitted by
`baselines.*` scripts and plots one line per (baseline_tag, config) with
x = k, y = accuracy. Writes `<results_root>/<model_short>/pass_at_k.pdf`
(and `.png` with --save-png).

Explicit two-level loop: outer over baselines (PS, PQ), inner over each
baseline's configs (one per SOAR run for PS; one per SOAR run × PQ-LR for PQ).
Stable colors per baseline tag so PS/PQ keep the same hue when new runs land.

Sorting mode (--sort-results-dir): reads `<dir>/<label>/**/<split>_samples.jsonl`
from `sgs.eval_sort` (n samples per list with per-sample `correct`), computes
the unbiased pass@k for k = 1..n per list length, and writes
`<dir>/pass_at_k_<split>.pdf` (and `.png` with --save-png): one panel per list
length plus "all", one line per label.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


FONT_SIZE = 22
TITLE_FONT_SIZE = 20
LEGEND_FONT_SIZE = 16

_SET2 = sns.color_palette("Set2", 8)
MARKER_CYCLE = ("o", "s", "D", "^", "v", "P", "X", "*")

# Order the baselines appear in the legend (and get pinned colors).
BASELINE_ORDER = ["PS", "PQ", "HO"]
TAG_FULL_NAMES = {"PS": "Promoted Student", "PQ": "Promotion Question", "HO": "Hard Only"}
TAG_COLORS = {"PS": _SET2[0], "PQ": _SET2[1], "HO": _SET2[2]}


def load_records(results_root: Path, model_short: str) -> list[dict[str, Any]]:
    model_dir = results_root / model_short
    if not model_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(model_dir.glob("*/*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"[plot] skip (bad JSON): {path}", flush=True)
    return records


def _apply_plot_theme() -> None:
    rc = {
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "font.size": FONT_SIZE,
    }
    sns.set_theme(style="whitegrid", rc=rc)
    plt.rcParams.update(rc)


def _make_label(tag: str, rec: dict[str, Any]) -> str:
    run_name = rec.get("run_name", "?")
    if tag == "PQ":
        lr = rec.get("lr")
        if lr is not None:
            soar_short = run_name
            suffix = f"-pqlr{lr}"
            if soar_short.endswith(suffix):
                soar_short = soar_short[: -len(suffix)]
            return f"PQ · lr={lr} · {soar_short}"
        return f"PQ · {run_name}"
    if tag == "PS":
        return f"PS · {run_name}"
    if tag == "HO":
        lr = rec.get("lr")
        seed = rec.get("seed")
        return f"HO · lr={lr} · seed={seed}"
    return f"{tag} · {run_name}"


def _parse_pass_at_k(rec: dict[str, Any]) -> tuple[list[int], list[float]]:
    pk = rec.get("test_pass_at_k") or {}
    pairs: list[tuple[int, float]] = []
    for k_str, v in pk.items():
        try:
            pairs.append((int(k_str), float(v)))
        except (TypeError, ValueError):
            continue
    pairs.sort()
    if not pairs:
        return [], []
    ks, ys = zip(*pairs)
    return list(ks), list(ys)


def build_plot(
    records: list[dict[str, Any]],
    *,
    model_short: str,
    output_path: Path,
    save_png: bool,
) -> None:
    if not records:
        print(f"[plot] no records for {model_short}; skipping", flush=True)
        return

    _apply_plot_theme()
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    legend_handles: list[Line2D] = []
    all_ks: set[int] = set()
    n_lines_drawn = 0

    # Collect baselines in BASELINE_ORDER first, then any unknown tags afterward.
    seen_tags = {r.get("baseline_tag") for r in records if r.get("baseline_tag")}
    ordered_tags = [t for t in BASELINE_ORDER if t in seen_tags] + sorted(
        seen_tags - set(BASELINE_ORDER)
    )

    for tag_i, tag in enumerate(ordered_tags):
        configs = sorted(
            [r for r in records if r.get("baseline_tag") == tag],
            key=lambda r: r.get("run_name", ""),
        )
        color = TAG_COLORS.get(tag, _SET2[tag_i % len(_SET2)])
        for i, rec in enumerate(configs):
            ks, ys = _parse_pass_at_k(rec)
            if not ks:
                print(
                    f"[plot] skip (no test_pass_at_k): {rec.get('run_name', '?')}",
                    flush=True,
                )
                continue
            marker = MARKER_CYCLE[i % len(MARKER_CYCLE)]
            label = _make_label(tag, rec)
            ax.plot(ks, ys, color=color, marker=marker,
                    linewidth=5.0, markersize=15.0, label=label)
            legend_handles.append(
                Line2D(
                    [0], [0], color=color, marker=marker,
                    markersize=12, linewidth=5.0, label=label,
                )
            )
            all_ks.update(ks)
            n_lines_drawn += 1

    if n_lines_drawn == 0:
        print(f"[plot] no usable records for {model_short}; skipping", flush=True)
        plt.close(fig)
        return

    ax.set_xscale("log", base=2)
    xticks = sorted(all_ks)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(k) for k in xticks])
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title(f"pass@k — {model_short}")

    ncol = min(2, len(legend_handles))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=ncol,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.subplots_adjust(bottom=0.30)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if save_png:
        png_path = output_path.with_suffix(".png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"[plot] wrote {png_path}", flush=True)
    plt.close(fig)
    print(f"[plot] wrote {output_path}", flush=True)


def load_sort_pass_at_k(results_dir: Path) -> pd.DataFrame:
    """Long table: label, split, length (str, incl. "all"), k, pass (unbiased pass@k)."""
    rows = []
    for label_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for split in ("train", "val"):
            files = sorted(label_dir.rglob(f"{split}_samples.jsonl"))
            if not files:
                continue
            df = pd.read_json(files[-1], lines=True)  # last checkpoint if several
            n = len(df["correct"].iloc[0])
            df["length"] = df["list"].map(len)
            for k in range(1, n + 1):
                df["pk"] = df["correct"].map(lambda oks: 1 - comb(n - sum(oks), k) / comb(n, k))
                for length, g in [*df.groupby("length"), ("all", df)]:
                    rows.append({"label": label_dir.name, "split": split,
                                 "length": str(length), "k": k, "pass": g["pk"].mean()})
    return pd.DataFrame(rows)


def build_sort_plots(df: pd.DataFrame, *, results_dir: Path, save_png: bool) -> None:
    _apply_plot_theme()
    labels = sorted(df["label"].unique())
    style = {lab: (_SET2[i % len(_SET2)], MARKER_CYCLE[i % len(MARKER_CYCLE)])
             for i, lab in enumerate(labels)}
    for split, sdf in df.groupby("split"):
        lengths = sorted({x for x in sdf["length"] if x != "all"}, key=int) + ["all"]
        fig, axes = plt.subplots(1, len(lengths), figsize=(5.5 * len(lengths), 5.5),
                                 sharey=True, squeeze=False)
        for ax, length in zip(axes[0], lengths):
            for lab, g in sdf[sdf["length"] == length].groupby("label"):
                color, marker = style[lab]
                ax.plot(g["k"], g["pass"], color=color, marker=marker,
                        linewidth=3.0, markersize=10.0)
            ax.set_title(f"len {length}" if length != "all" else "all")
            ax.set_xticks(sorted(sdf["k"].unique()))
            ax.set_xlabel("k")
        axes[0][0].set_ylabel("pass@k")
        handles = [Line2D([0], [0], color=c, marker=m, markersize=10, linewidth=3.0, label=lab)
                   for lab, (c, m) in style.items() if lab in set(sdf["label"])]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                   ncol=min(4, len(handles)), frameon=False, fontsize=LEGEND_FONT_SIZE)
        fig.suptitle(f"pass@k — {results_dir.name} ({split})", fontsize=TITLE_FONT_SIZE, y=1.04)
        out = results_dir / f"pass_at_k_{split}.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        if save_png:
            fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {out}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path,
                        default=Path("files/results/baselines"))
    parser.add_argument("--sort-results-dir", type=Path, default=None,
                        help="sorting mode: sgs.eval_sort results dir (one subdir per label)")
    parser.add_argument("--model-short", type=str, default=None,
                        help="e.g. llama-3.2-3b-instruct")
    parser.add_argument("--out", type=Path, default=None,
                        help="output path; default <results-root>/<model-short>/pass_at_k.pdf")
    parser.add_argument("--save-png", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sort_results_dir is not None:
        df = load_sort_pass_at_k(args.sort_results_dir)
        if df.empty:
            print(f"[plot] no *_samples.jsonl under {args.sort_results_dir}", flush=True)
            return
        build_sort_plots(df, results_dir=args.sort_results_dir, save_png=args.save_png)
        return
    if args.model_short is None:
        raise SystemExit("--model-short is required without --sort-results-dir")
    records = load_records(args.results_root, args.model_short)
    out_path = args.out or (args.results_root / args.model_short / "pass_at_k.pdf")
    build_plot(records, model_short=args.model_short, output_path=out_path,
               save_png=args.save_png)


if __name__ == "__main__":
    main()
