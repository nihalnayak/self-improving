"""Generate preorder traversal training data for SGS trainers.

Writes traversals of random binary trees with unique labels 0 to length - 1.
Labels are shuffled, then each subtree gets a random left-subtree size.
Each record has this form:

    {"inorder": <list>, "postorder": <list>, "preorder": <list>}

Defaults produce 1000 samples balanced across node counts
{8, 16, 32, 64, 128} (200 each).

Usage:
    python misc/gen_preordertraversal_data.py
    python misc/gen_preordertraversal_data.py --seed 1 --out files/data/preordertraversal/preordertraversal_seed1.jsonl
    python misc/gen_preordertraversal_data.py --n 1000 --lengths 8,16,128
"""

import argparse
import json
import random
from pathlib import Path


def gen_sample(length: int, rng: random.Random) -> dict:
    labels = list(range(length))
    rng.shuffle(labels)

    def traverse(values):
        if not values:
            return [], [], []
        root = values[0]
        split = rng.randrange(len(values))
        li, lp, lr = traverse(values[1:split + 1])
        ri, rp, rr = traverse(values[split + 1:])
        return li + [root] + ri, lp + rp + [root], [root] + lr + rr

    inorder, postorder, preorder = traverse(labels)
    return {"inorder": inorder, "postorder": postorder, "preorder": preorder}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path,
                   default=Path("files/data/preordertraversal/preordertraversal_1000.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lengths", type=str, default="8,16,32,64,128",
                   help="comma-separated node counts")
    p.add_argument("--per-length", type=int, default=200,
                   help="number of samples per node count")
    p.add_argument("--n", type=int, default=None,
                   help="total samples split evenly across lengths "
                        "(remainder to the last lengths); overrides --per-length")
    args = p.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    if not lengths or any(x < 1 for x in lengths):
        p.error("--lengths must contain positive node counts")
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
