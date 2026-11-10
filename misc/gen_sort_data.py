"""Generate sorting training data for SGS trainers.

Writes a JSONL corpus of unsorted integer lists (plus their sorted
counterparts) consumable by `sgs/sgs_verl.py` and `sgs/sgs_simple.py`
via `--dataset <path>`. The trainers read only `list` and compute
`sorted(list)` internally (see `load_targets` in sgs_verl.py); the
`sorted` field here is a redundant sanity record.

Defaults produce 1000 samples balanced across lengths {8, 16, 32, 64,
128} (200 each) with integers in [0, 10000] (matching
`sgs_verl.Cfg.conj_lo/conj_hi`).

Usage:
    python misc/gen_sort_data.py
    python misc/gen_sort_data.py --seed 1 --out files/data/sorting/sorting_seed1.jsonl
    python misc/gen_sort_data.py --n 1000 --lengths 8,16,128 --out files/data/sorting/sorting_1000_L8-16-128.jsonl
"""

import argparse
import json
import random
from pathlib import Path


def gen_sample(length: int, rng: random.Random, lo: int, hi: int) -> dict:
    while True:
        xs = [rng.randint(lo, hi) for _ in range(length)]
        if xs != sorted(xs):
            return {"list": xs, "sorted": sorted(xs)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path,
                   default=Path("files/data/sorting/sorting_1000.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lengths", type=str, default="8,16,32,64,128",
                   help="comma-separated list lengths")
    p.add_argument("--per-length", type=int, default=200,
                   help="number of samples per length")
    p.add_argument("--n", type=int, default=None,
                   help="total samples split evenly across lengths "
                        "(remainder to the last lengths); overrides --per-length")
    p.add_argument("--lo", type=int, default=0)
    p.add_argument("--hi", type=int, default=10000)
    args = p.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    k = len(lengths)
    if args.n is None:
        counts = [args.per_length] * k
    else:
        counts = [args.n // k + (i >= k - args.n % k) for i in range(k)]
    rng = random.Random(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w") as f:
        for L, c in zip(lengths, counts):
            for _ in range(c):
                row = gen_sample(L, rng, args.lo, args.hi)
                f.write(json.dumps(row) + "\n")
                n += 1

    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
