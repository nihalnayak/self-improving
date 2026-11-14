"""Generate multiplication training data for SGS trainers.

Writes a JSONL corpus of non-negative integer pairs and their products.
Each record has this form:

    {"number1": <integer>, "number2": <integer>, "product": <integer>}

Defaults produce 1000 samples balanced across operand digit counts
{8, 16, 32, 64, 128} (200 each). Both operands have the same digit
count, with no leading zeros. Products are stored as JSON integers.

Usage:
    python misc/gen_multiplication_data.py
    python misc/gen_multiplication_data.py --seed 1 --out files/data/multiplication/multiplication_seed1.jsonl
    python misc/gen_multiplication_data.py --n 1000 --lengths 8,16,128
"""

import argparse
import json
import random
from pathlib import Path


def gen_sample(length: int, rng: random.Random) -> dict:
    lo, hi = 10 ** (length - 1), 10 ** length - 1
    number1 = rng.randint(lo, hi)
    number2 = rng.randint(lo, hi)
    return {"number1": number1, "number2": number2,
            "product": number1 * number2}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path,
                   default=Path("files/data/multiplication/multiplication_1000.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lengths", type=str, default="8,16,32,64,128",
                   help="comma-separated operand digit counts")
    p.add_argument("--per-length", type=int, default=200,
                   help="number of samples per digit count")
    p.add_argument("--n", type=int, default=None,
                   help="total samples split evenly across lengths "
                        "(remainder to the last lengths); overrides --per-length")
    args = p.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
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
                f.write(json.dumps(gen_sample(L, rng)) + "\n")
                n += 1

    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()



