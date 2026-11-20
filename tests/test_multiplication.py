"""Multiplication task pipeline: data -> Solver prompt -> output parsing -> Conjecturer
-> Guide -> verl.

    $PYTHON tests/test_multiplication.py
"""

import random
import tempfile

from _trace import benchmark_defs, read_jsonl, row, run, show_prompt

from misc.gen_multiplication_data import gen_sample
from sgs.tasks import MULT_SOLVER_PROMPT, TASKS, Multiplication, parse_product

FILES = ["multiplication_val_100", "multiplication_1000", "multiplication_1000_L2-10",
         "multiplication_1000_L2-4-10", "multiplication_1000_L2-4-6-10"]
P = [12, 34]   # 408


def pairs(rows):
    return {tuple(sorted((r["number1"], r["number2"]))) for r in rows}


def test_generation():
    s = gen_sample(4, random.Random(0))
    print(f"  digits=4: {s['number1']} x {s['number2']} = {s['product']}")
    assert s["product"] == s["number1"] * s["number2"] and len(str(s["number1"])) == 4


def test_data_files():
    val = pairs(read_jsonl(f"data/multiplication/{FILES[0]}.jsonl"))
    for name in FILES:
        rows = read_jsonl(f"data/multiplication/{name}.jsonl")
        print(f"  {name:<34} rows={len(rows):>4} digits={sorted({len(str(r['number1'])) for r in rows})}")
        assert all(r["product"] == r["number1"] * r["number2"] for r in rows)
        assert len(pairs(rows)) == len(rows)
        if name != FILES[0]:
            assert not pairs(rows) & val
    big = max(Multiplication.load("data/multiplication/multiplication_1000.jsonl"))
    print(f"  largest target {big} -> answer {Multiplication.answer(big)!r}")
    assert int(Multiplication.answer(big)) == big[0] * big[1]


def test_solver_prompt():
    show_prompt("Solver", Multiplication.solver_messages(P))
    (bench_prompt,) = benchmark_defs("misc/multiplication_benchmark.py", ["PROMPT_TEMPLATE"])
    assert MULT_SOLVER_PROMPT == bench_prompt


OUTPUTS = [
    ("correct boxed", "\\boxed{408}", 408),
    ("think trace, then boxed", "<think>\\boxed{400}</think>\\boxed{408}", 408),
    ("no box: last integer", "12 x 34 = 408", 408),
    ("wrong", "\\boxed{418}", 418),
    ("commas inside box", "\\boxed{4,08}", None),
    ("empty", "", None),
]


def test_solver_parsing():
    (bench_parse,) = benchmark_defs("misc/multiplication_benchmark.py", ["parse_product"])
    print(f"  problem {P}, gold {Multiplication.answer(P)!r}")
    for label, out, want in OUTPUTS:
        got, ok = parse_product(out), Multiplication.verify(out, P)
        row(label, out, got, ok)
        assert got == want == bench_parse(out)
        assert ok == (want == 408) == Multiplication.check(out, Multiplication.answer(P))


CONJ_OUTPUTS = [
    ("valid", "<problem>\nMultiply the following two integers.\nnumber1: 23\nnumber2: 47\n</problem>\n"
              "<solution>\n1081\n</solution>", [23, 47]),
    ("one-digit operand", "<problem>\nnumber1: 3\nnumber2: 47\n</problem>", None),
    ("missing operand", "<problem>\nnumber1: 23\n</problem>", "none-none"),
]


def test_conjecturer():
    show_prompt("Conjecturer", Multiplication.conj_messages([4821, 7350], 0, 10000))
    for label, out, want in CONJ_OUTPUTS:
        statement, problem = Multiplication.conjecture(out)
        row(label, out, {"problem": problem}, problem is not None)
        if want == "none-none":
            assert (statement, problem) == (None, None)
        else:
            assert problem == want


def test_guide():
    from sgs.sgs_simple import parse_guide
    statement, _ = Multiplication.conjecture(CONJ_OUTPUTS[0][1])
    show_prompt("Guide", Multiplication.guide_messages([4821, 7350], statement))
    out = "<relevance>5</relevance>\n<redundancy>0</redundancy>\n<complexity>0</complexity>"
    row("all tags", out, parse_guide(out), True)
    assert parse_guide(out) == (5, 0, 0)


def test_verl_path():
    import pandas as pd
    from sgs.solver_simple import compute_score, write_parquet
    with tempfile.TemporaryDirectory() as d:
        write_parquet([P], f"{d}/t.parquet", TASKS["multiplication"], source_prefix="val_")
        r = pd.read_parquet(f"{d}/t.parquet").iloc[0]
    print(f"  data_source={r.data_source} ground_truth={r.reward_model['ground_truth']!r}")
    assert r.data_source == "val_mult_len_2"
    assert compute_score(r.data_source, "\\boxed{408}", r.reward_model["ground_truth"]) == 1.0
    assert compute_score("mult_len_2", "\\boxed{409}", "408") == 0.0


if __name__ == "__main__":
    run(globals())
