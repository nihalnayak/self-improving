"""Print stats for every fail@K dataset under a root directory tree.

Walks `--root` for `meta.json` files and reports, per dataset:
    - base_model, pass@K
    - n_source, n_fail, n_train, n_test, fail-rate
    - runtime, hist-based summary (fraction solved at least once, median non-zero)
    - subject and level breakdown of the fail set (from train.jsonl + test.jsonl)
Followed by a side-by-side comparison table across all datasets.

Usage:
    python -m misc.stats_fail128
    python -m misc.stats_fail128 --root /n/netscratch/dam_lab/Lab/nnayak1/soar/data/math_fail
    python -m misc.stats_fail128 --path dir1 --path dir2
    python -m misc.stats_fail128 --csv summary.csv
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import List, Dict

from soar.data import load_fail128_split


DEFAULT_ROOT = "/n/netscratch/dam_lab/Lab/nnayak1/soar/data/math_fail"


def discover_datasets(root: Path, extra: List[Path]) -> List[Path]:
    dirs: List[Path] = []
    if root.exists():
        dirs.extend(sorted({p.parent for p in root.rglob("meta.json")}))
    for p in extra:
        p = Path(p)
        if (p / "meta.json").exists() and p not in dirs:
            dirs.append(p)
    return dirs


def _hist_summary(meta: dict) -> Dict[str, float]:
    hist = {int(k): int(v) for k, v in meta.get("pass_at_n_histogram", {}).items()}
    n_source = int(meta.get("n_source", sum(hist.values()) or 1))
    zero = hist.get(0, 0)
    non_zero = [k for k, v in hist.items() for _ in range(v) if k > 0]
    return {
        "frac_solved_once": 1.0 - zero / n_source if n_source else 0.0,
        "median_nonzero_success": float(median(non_zero)) if non_zero else 0.0,
        "max_success": float(max(hist)) if hist else 0.0,
    }


def _bucket_counts(items, key) -> List[tuple]:
    c = Counter(key(x) or "(unknown)" for x in items)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))


def _fmt_bucket(label: str, counts: List[tuple]) -> str:
    total = sum(v for _, v in counts) or 1
    lines = [f"  {label}:"]
    for name, n in counts:
        lines.append(f"    {name:<28} {n:>5}  ({100 * n / total:5.1f}%)")
    return "\n".join(lines)


def analyze(dataset_dir: Path) -> dict:
    with open(dataset_dir / "meta.json") as f:
        meta = json.load(f)
    train, test = load_fail128_split(str(dataset_dir))
    kept = train + test
    hist = _hist_summary(meta)
    return {
        "dir": dataset_dir,
        "meta": meta,
        "n_kept_loaded": len(kept),
        "subject_counts": _bucket_counts(kept, lambda p: p.subject),
        "level_counts": _bucket_counts(kept, lambda p: p.level),
        **hist,
    }


def print_detail(r: dict) -> None:
    m = r["meta"]
    print("")
    print("=" * 78)
    print(f" {m['base_model']}  pass@{m['n_samples']}")
    print(f" dir: {r['dir']}")
    print("=" * 78)
    n_source = m.get("n_source", 0)
    n_fail = m.get("n_fail", 0)
    fail_rate = n_fail / n_source if n_source else 0.0
    runtime_min = m.get("runtime_sec", 0.0) / 60.0
    print(f"  n_source  = {n_source}")
    print(f"  n_fail    = {n_fail}  ({100 * fail_rate:.1f}% of source)")
    print(f"  n_train   = {m.get('n_train', 0)}")
    print(f"  n_test    = {m.get('n_test', 0)}")
    print(f"  loaded    = {r['n_kept_loaded']} (train+test rows actually on disk)")
    print(f"  runtime   = {runtime_min:.1f} min")
    print(f"  solved>=1 = {100 * r['frac_solved_once']:.1f}% of source "
          f"(median non-zero success count = {r['median_nonzero_success']:.0f}, "
          f"max = {r['max_success']:.0f})")
    print(_fmt_bucket("subject", r["subject_counts"]))
    print(_fmt_bucket("level", r["level_counts"]))


def print_table(results: List[dict], csv_path: Path = None) -> None:
    rows = []
    for r in results:
        m = r["meta"]
        n_source = m.get("n_source", 0)
        n_fail = m.get("n_fail", 0)
        rows.append({
            "model": m["base_model"],
            "K": m["n_samples"],
            "n_fail": n_fail,
            "fail_rate": f"{(n_fail / n_source if n_source else 0.0):.3f}",
            "n_train": m.get("n_train", 0),
            "n_test": m.get("n_test", 0),
            "runtime_min": f"{m.get('runtime_sec', 0.0) / 60.0:.1f}",
        })
    rows.sort(key=lambda r: (r["model"], int(r["K"])))

    cols = ["model", "K", "n_fail", "fail_rate", "n_train", "n_test", "runtime_min"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols} if rows else {c: len(c) for c in cols}

    print("")
    print("=" * 78)
    print(" summary")
    print("=" * 78)
    header = "  ".join(f"{c:<{widths[c]}}" for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        print("  ".join(f"{str(r[c]):<{widths[c]}}" for c in cols))

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[write] {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="Walk this tree for meta.json files.")
    ap.add_argument("--path", action="append", default=[],
                    help="Also include this explicit dataset dir (repeatable).")
    ap.add_argument("--csv", default=None, help="Write summary table to this CSV.")
    args = ap.parse_args()

    dirs = discover_datasets(Path(args.root), [Path(p) for p in args.path])
    if not dirs:
        print(f"no datasets with meta.json found under {args.root} "
              f"(and no --path overrides provided)")
        return

    results = [analyze(d) for d in dirs]
    results.sort(key=lambda r: (r["meta"]["base_model"], int(r["meta"]["n_samples"])))
    for r in results:
        print_detail(r)
    print_table(results, Path(args.csv) if args.csv else None)


if __name__ == "__main__":
    main()
