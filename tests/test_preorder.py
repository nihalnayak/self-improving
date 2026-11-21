"""Preorder traversal task pipeline: data -> Solver prompt -> output parsing -> Conjecturer
-> Guide -> verl.

A problem is [inorder, postorder]; the answer is the preorder of the unique binary tree
with those traversals (recomputed by sgs.tasks.preorder_from, never read from jsonl).

    $PYTHON tests/test_preorder.py
"""

import random
import tempfile

from _trace import benchmark_defs, read_jsonl, row, run, show_prompt

from misc.gen_preordertraversal_data import gen_sample
from sgs.tasks import PREORDER_SOLVER_PROMPT, TASKS, Preorder, parse_preorder, preorder_from

FILES = ["preorder_val_100", "preorder_1000", "preorder_1000_L8-128", "preorder_1000_L8-16-128",
         "preorder_1000_L8-16-32-128"]
#       2
#      / \
#     0   1
#      \
#       3
P = [[0, 3, 2, 1], [3, 0, 1, 2]]
GOLD = [2, 0, 3, 1]


def test_generation():
    rng = random.Random(0)
    for L in (3, 8):
        s = gen_sample(L, rng)
        print(f"  nodes={L}: inorder={s['inorder']} postorder={s['postorder']} -> preorder={s['preorder']}")
        assert sorted(s["inorder"]) == sorted(s["postorder"]) == sorted(s["preorder"]) == list(range(L))
        assert preorder_from(s["inorder"], s["postorder"]) == s["preorder"]


def test_reconstruction():
    print(f"  example tree: inorder={P[0]} postorder={P[1]} -> {preorder_from(*P)}")
    assert preorder_from(*P) == GOLD
    for label, ino, post in [
        ("impossible split", [0, 1, 2], [2, 0, 1]),
        ("repeated label", [0, 0, 1], [0, 1, 0]),
        ("different label sets", [0, 1, 2], [0, 1, 3]),
        ("different lengths", [0, 1, 2], [0, 1]),
    ]:
        got = preorder_from(ino, post)
        print(f"  {label:<22} inorder={ino} postorder={post} -> {got}")
        assert got is None
    assert preorder_from([7], [7]) == [7] and preorder_from([], []) == []
    # random trees, including degenerate chains, up to 300 nodes (recursion depth)
    rng = random.Random(1)
    for L in list(range(1, 40)) + [128, 300]:
        s = gen_sample(L, rng)
        assert preorder_from(s["inorder"], s["postorder"]) == s["preorder"]
    chain = list(range(300))                     # every node only a left child
    assert preorder_from(chain, chain) == chain[::-1]


def test_data_files():
    val = {tuple(r["inorder"] + r["postorder"]) for r in read_jsonl(f"data/preorder/{FILES[0]}.jsonl")}
    for name in FILES:
        path = f"data/preorder/{name}.jsonl"
        rows, targets = read_jsonl(path), Preorder.load(path)
        levels = sorted({len(r["inorder"]) for r in rows})
        print(f"  {name:<28} rows={len(rows):>4} nodes={levels}")
        # gold is recomputed from the traversals and must equal the generator's preorder
        for r, p in zip(rows, targets):
            assert p == [r["inorder"], r["postorder"]] and Preorder.answer(p) == r["preorder"]
        assert levels == ([8, 128] if "L8-128" in name else [8, 16, 128] if "L8-16-128" in name
                          else [8, 16, 32, 128] if "L8-16-32-128" in name else [8, 16, 32, 64, 128])
        if name != FILES[0]:
            assert not {tuple(r["inorder"] + r["postorder"]) for r in rows} & val, "train tree in val"
    assert Preorder.level(targets[-1]) == 128


def test_solver_prompt():
    show_prompt("Solver", Preorder.solver_messages(P))
    (bench_prompt,) = benchmark_defs("misc/preorder_benchmark.py", ["PROMPT_TEMPLATE"])
    assert PREORDER_SOLVER_PROMPT == bench_prompt, "Solver prompt must match misc/preorder_benchmark.py"


OUTPUTS = [  # (label, model output, expected parsed value)
    ("correct boxed", "\\boxed{[2, 0, 3, 1]}", GOLD),
    ("no brackets in box", "\\boxed{2, 0, 3, 1}", GOLD),
    ("think trace, then boxed", "<think>root is 2 ... \\boxed{[1]}</think>\n\\boxed{[2, 0, 3, 1]}", GOLD),
    ("wrong order", "\\boxed{[2, 3, 0, 1]}", [2, 3, 0, 1]),
    ("postorder copied", "\\boxed{[3, 0, 1, 2]}", [3, 0, 1, 2]),
    ("missing a node", "\\boxed{[2, 0, 3]}", [2, 0, 3]),
    ("no box (no fallback)", "The preorder is [2, 0, 3, 1].", None),
    ("two boxes: first wins", "\\boxed{[2, 0, 3, 1]} \\boxed{[0]}", GOLD),
    ("non-integer token", "\\boxed{[2, 0, a, 1]}", None),
    ("empty box", "\\boxed{[]}", []),
    ("truncated mid-box", "\\boxed{[2, 0, 3", None),
]


def test_solver_parsing():
    (bench_parse,) = benchmark_defs("misc/preorder_benchmark.py", ["parse_preorder"])
    print(f"  problem {P}, gold {Preorder.answer(P)}")
    for label, out, want in OUTPUTS:
        got, ok = parse_preorder(out), Preorder.verify(out, P)
        row(label, out, got, ok)
        assert got == want == bench_parse(out), "parser must match misc/preorder_benchmark.parse_preorder"
        assert ok == (want == GOLD) == Preorder.check(out, GOLD)


STMT = ("A binary tree has unique node labels. Given its inorder and postorder traversals, "
        "compute its preorder traversal.\n")
CONJ_OUTPUTS = [  # (label, conjecturer output, expected problem; "none-none" = unparsable)
    ("valid", f"<problem>\n{STMT}Inorder: [0, 3, 2, 1]\nPostorder: [3, 0, 1, 2]\n</problem>\n"
              "<solution>\n[2, 0, 3, 1]\n</solution>", P),
    ("wrong solution ignored", "<problem>\nInorder: [0, 3, 2, 1]\nPostorder: [3, 0, 1, 2]\n</problem>\n"
                               "<solution>\n[1, 2, 3, 0]\n</solution>", P),
    ("lowercase keys", "<problem>\ninorder: [1, 0, 2]\npostorder: [1, 2, 0]\n</problem>", [[1, 0, 2], [1, 2, 0]]),
    ("inconsistent traversals", "<problem>\nInorder: [0, 1, 2]\nPostorder: [2, 0, 1]\n</problem>", None),
    ("repeated label", "<problem>\nInorder: [0, 0, 1]\nPostorder: [0, 1, 0]\n</problem>", None),
    ("single node", "<problem>\nInorder: [5]\nPostorder: [5]\n</problem>", None),
    ("non-integer label", "<problem>\nInorder: [a, b]\nPostorder: [b, a]\n</problem>", "none-none"),
    ("missing postorder", "<problem>\nInorder: [0, 1, 2]\n</problem>", "none-none"),
    ("no block", "Inorder: [0, 1]\nPostorder: [1, 0]", "none-none"),
    ("two blocks: last wins", "<problem>Inorder: [0, 1]\nPostorder: [0, 1]</problem>\n"
                              "<problem>Inorder: [1, 0, 2]\nPostorder: [1, 2, 0]</problem>",
     [[1, 0, 2], [1, 2, 0]]),
]


def test_conjecturer():
    target = Preorder.load("data/preorder/preorder_val_100.jsonl")[0]
    show_prompt("Conjecturer", Preorder.conj_messages(target, 0, 10000))
    for label, out, want in CONJ_OUTPUTS:
        statement, problem = Preorder.conjecture(out)
        gold = Preorder.answer(problem) if problem else None
        row(label, out, {"problem": problem, "gold": gold}, problem is not None)
        if want == "none-none":
            assert (statement, problem) == (None, None)
        else:
            assert problem == want and statement is not None
    # round trip: a generated tree written in the Conjecturer's format parses back to its gold
    rng = random.Random(2)
    for L in (3, 5, 8, 16):
        s = gen_sample(L, rng)
        text = (f"<problem>\n{STMT}Inorder: {s['inorder']}\nPostorder: {s['postorder']}\n</problem>\n"
                f"<solution>\n{s['preorder']}\n</solution>")
        _, problem = Preorder.conjecture(text)
        assert Preorder.answer(problem) == s["preorder"]


def test_guide():
    from sgs.sgs_simple import guide_score, parse_guide
    target = Preorder.load("data/preorder/preorder_val_100.jsonl")[0]
    statement, _ = Preorder.conjecture(CONJ_OUTPUTS[0][1])
    show_prompt("Guide", Preorder.guide_messages(target, statement))
    for label, out, want in [
        ("all tags", "...\n<relevance>3</relevance>\n<redundancy>1</redundancy>\n<complexity>0</complexity>", (3, 1, 0)),
        ("last tag wins", "<relevance>1</relevance> no, <relevance>4</relevance><redundancy>0</redundancy>"
                          "<complexity>3</complexity>", (4, 0, 3)),
        ("missing tag", "<relevance>4</relevance>", None),
    ]:
        scores = parse_guide(out)
        row(label, out, {"scores": scores, "R_guide": guide_score(scores)}, scores is not None)
        assert scores == want


def test_verl_path():
    import pandas as pd
    from sgs.solver_simple import compute_score, write_parquet
    targets = Preorder.load("data/preorder/preorder_val_100.jsonl")
    with tempfile.TemporaryDirectory() as d:
        write_parquet(targets, f"{d}/val.parquet", TASKS["preorder"], source_prefix="val_")
        df = pd.read_parquet(f"{d}/val.parquet")
    r = df.iloc[0]
    gt = r.reward_model["ground_truth"]
    print(f"  data_source={r.data_source} ground_truth={list(map(int, gt))}")
    assert r.data_source == "val_preorder_len_8" and list(gt) == Preorder.answer(targets[0])
    assert r.prompt[0]["content"] == Preorder.solver_messages(targets[0])[0]["content"]
    for src, out, want in [("preorder_len_4", "\\boxed{[2, 0, 3, 1]}", 1.0),
                           ("val_preorder_len_4", "\\boxed{[3, 0, 1, 2]}", 0.0)]:
        score = compute_score(src, out, GOLD)
        print(f"  compute_score({src!r}, {out!r}) = {score}")
        assert score == want
    # ground truth as read back from parquet (numpy array) still scores
    assert compute_score(r.data_source, f"\\boxed{{{list(map(int, gt))}}}", gt) == 1.0


if __name__ == "__main__":
    run(globals())
