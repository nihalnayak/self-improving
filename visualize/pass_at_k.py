#!/usr/bin/env python3
"""Plot pass@k curves across baselines for a given model.

Walks `<results_root>/<model_short>/<run_name>/<baseline>.json` emitted by
`baselines.*` scripts and plots one line per (baseline_tag, run_name) with
x = k, y = accuracy. Writes `<results_root>/<model_short>/pass_at_k.pdf`
(and `.png` with --save-png).

Styled after /n/holylabs/dam_lab/Users/nnayak1/synthetic-personas/analysis/samples_vs_vendi_v2.py.
"""

from __future__ import annotations

import argparse
import json
from itertools import cycle
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


def _build_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        tag = rec.get("baseline_tag")
        run_name = rec.get("run_name")
        pk = rec.get("test_pass_at_k") or {}
        if not tag or not run_name or not pk:
            print(f"[plot] skip record missing fields: {rec.get('run_name', '?')}",
                  flush=True)
            continue
        label = f"{tag} ({run_name})"
        for k_str, acc in pk.items():
            try:
                k = int(k_str)
                v = float(acc)
            except (TypeError, ValueError):
                continue
            rows.append({
                "baseline_tag": tag,
                "run_name": run_name,
                "label": label,
                "K": k,
                "accuracy": v,
            })
    return pd.DataFrame(rows)


def _assign_styles(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, str]]:
    """Stable color per baseline_tag; distinct marker per (tag, run_name)."""
    tags = sorted(df["baseline_tag"].unique())
    tag_color = {tag: _SET2[i % len(_SET2)] for i, tag in enumerate(tags)}

    label_color: dict[str, Any] = {}
    label_marker: dict[str, str] = {}
    for tag in tags:
        labels_for_tag = sorted(df.loc[df["baseline_tag"] == tag, "label"].unique())
        marker_iter = cycle(MARKER_CYCLE)
        for label in labels_for_tag:
            label_color[label] = tag_color[tag]
            label_marker[label] = next(marker_iter)
    return label_color, label_marker


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

    df = _build_dataframe(records)
    if df.empty:
        print(f"[plot] no usable rows for {model_short}; skipping", flush=True)
        return

    _apply_plot_theme()
    label_color, label_marker = _assign_styles(df)
    label_order = sorted(df["label"].unique())

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    sns.lineplot(
        data=df.sort_values("K"),
        x="K",
        y="accuracy",
        hue="label",
        hue_order=label_order,
        style="label",
        style_order=label_order,
        markers=label_marker,
        palette=label_color,
        linewidth=5.0,
        markersize=15.0,
        legend=False,
        ax=ax,
    )

    ax.set_xscale("log", base=2)
    xticks = sorted({int(k) for k in df["K"].tolist()})
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(k) for k in xticks])
    ax.set_xlabel("k")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("pass@k")
    ax.set_title(f"pass@k — {model_short}")

    handles = [
        Line2D(
            [0],
            [0],
            color=label_color[label],
            marker=label_marker[label],
            markersize=12,
            linewidth=5.0,
            label=label,
        )
        for label in label_order
    ]
    ncol = min(2, len(handles))
    fig.legend(
        handles=handles,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path,
                        default=Path("files/results/baselines"))
    parser.add_argument("--model-short", type=str, required=True,
                        help="e.g. llama-3.2-3b-instruct")
    parser.add_argument("--out", type=Path, default=None,
                        help="output path; default <results-root>/<model-short>/pass_at_k.pdf")
    parser.add_argument("--save-png", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.results_root, args.model_short)
    out_path = args.out or (args.results_root / args.model_short / "pass_at_k.pdf")
    build_plot(records, model_short=args.model_short, output_path=out_path,
               save_png=args.save_png)


if __name__ == "__main__":
    main()
