"""Generate greatest common divisor (GCD) training data for SGS trainers.

Writes a JSONL corpus of non-negative integer pairs and their greatest common divisors.
Each record has this form:

    {"number1": <integer>, "number2": <integer>, "gcd": <integer>}

Defaults produce 1000 samples balanced across operand digit counts
{8, 16, 32, 64, 128} (200 each). Both operands have the same digit
count, with no leading zeros. GCDs are stored as JSON integers.
Use --exclude-coprime to resample pairs with GCD 1 while keeping sample counts.

Usage:
    python misc/gen_gcd_data.py
    python misc/gen_gcd_data.py --exclude-coprime
    python misc/gen_gcd_data.py --seed 1 --out files/data/gcd/gcd_seed1.jsonl
    python misc/gen_gcd_data.py --n 1000 --lengths 2,4,6,8,10 --out files/data/gcd/gcd_1000_L2-4-6-8-10.jsonl
    python misc/gen_gcd_data.py --exclude-coprime --n 1000 --lengths 2,4,6,8,10 --out files/data/gcd_excludecoprime/gcd_1000_L2-4-6-8-10.jsonl
"""

import argparse
import json
import math
import random
from pathlib import Path


def gen_sample(length: int, rng: random.Random) -> dict:
    lo, hi = 10 ** (length - 1), 10 ** length - 1
    number1 = rng.randint(lo, hi)
    number2 = rng.randint(lo, hi)
    return {"number1": number1, "number2": number2,
            "gcd": math.gcd(number1, number2)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path,
                   default=Path("files/data/gcd/gcd_1000.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-coprime", action="store_true",
                   help="resample pairs with GCD 1 to keep only GCDs greater than 1")
    p.add_argument("--lengths", type=str, default="8,16,32,64,128",
                   help="comma-separated operand digit counts")
    p.add_argument("--per-length", type=int, default=200,
                   help="number of samples per digit count")
    p.add_argument("--n", type=int, default=None,
                   help="total samples split evenly across lengths "
                        "(remainder to the last lengths); overrides --per-length")
    args = p.parse_args()

    try:
        lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    except ValueError:
        p.error("--lengths must contain positive digit counts")
    if not lengths or any(x < 1 for x in lengths):
        p.error("--lengths must contain positive digit counts")
    if args.per_length < 0 or (args.n is not None and args.n < 0):
        p.error("sample counts must be non-negative")
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
                sample = gen_sample(L, rng)
                while args.exclude_coprime and sample["gcd"] == 1:
                    sample = gen_sample(L, rng)
                f.write(json.dumps(sample) + "\n")
                n += 1

    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
