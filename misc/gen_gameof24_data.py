"""Generate Game of 24 training data from the Tree of Thoughts CSV.

Writes JSONL records with four input numbers, one solution expression
as a string, and the source adjusted median time (AMT) in seconds:

    {"numbers": <list>, "answer": <string>, "amt": <float>}

Defaults sample 1000 puzzles uniformly without replacement, using seed 0.
Solutions use each number exactly once with +, -, *, / and parentheses.
Exact rational arithmetic allows fractional intermediate results.
Use --min-amt and --max-amt to select a difficulty range (inclusive).
The CSV is downloaded at runtime unless --source supplies a local copy.

Usage:
    python misc/gen_gameof24_data.py
    python misc/gen_gameof24_data.py --seed 1 --out files/data/gameof24/gameof24_seed1.jsonl
    python misc/gen_gameof24_data.py --n 100 --min-amt 10
"""

import argparse
import csv
import io
import json
import random
from fractions import Fraction
from pathlib import Path
from urllib.request import urlopen


SOURCE_URL = "https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/master/src/tot/data/24/24.csv"


def solve(numbers: list) -> str | None:
    def search(items):
        if len(items) == 1:
            return items[0][1] if items[0][0] == 24 else None
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, ae = items[i]
                b, be = items[j]
                rest = [item for k, item in enumerate(items) if k not in (i, j)]
                candidates = [(a + b, f"({ae} + {be})"),
                              (a - b, f"({ae} - {be})"),
                              (b - a, f"({be} - {ae})"),
                              (a * b, f"({ae} * {be})")]
                if b:
                    candidates.append((a / b, f"({ae} / {be})"))
                if a:
                    candidates.append((b / a, f"({be} / {ae})"))
                for value, expression in candidates:
                    answer = search(rest + [(value, expression)])
                    if answer is not None:
                        return answer
        return None

    return search([(Fraction(n), str(n)) for n in numbers])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path,
                   default=Path("files/data/gameof24/gameof24_1000.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=1000, help="number of distinct puzzles")
    p.add_argument("--source", type=Path, help="local Tree of Thoughts CSV")
    p.add_argument("--min-amt", type=float, default=0)
    p.add_argument("--max-amt", type=float, default=float("inf"))
    args = p.parse_args()
    if args.n < 0:
        p.error("--n must be non-negative")
    if args.source is None:
        with urlopen(SOURCE_URL, timeout=30) as response:
            source = response.read().decode("utf-8")
    else:
        source = args.source.read_text()
    puzzles = []
    for row in csv.DictReader(io.StringIO(source)):
        amt = float(row["AMT (s)"])
        if args.min_amt <= amt <= args.max_amt:
            puzzles.append(([int(x) for x in row["Puzzles"].split()], amt))
    if args.n > len(puzzles):
        p.error(f"requested {args.n} puzzles, but only {len(puzzles)} match the AMT range")
    rng = random.Random(args.seed)
    rows = []
    for numbers, amt in rng.sample(puzzles, args.n):
        answer = solve(numbers)
        if answer is None:
            raise ValueError(f"source puzzle has no solution: {numbers}")
        rows.append({"numbers": numbers, "answer": answer, "amt": amt})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
