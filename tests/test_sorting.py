"""Sorting task pipeline: data -> Solver prompt -> output parsing -> Conjecturer -> Guide -> verl.

    $PYTHON tests/test_sorting.py
"""

import random
import tempfile

from _trace import benchmark_defs, read_jsonl, row, run, show_prompt

from misc.gen_sort_data import gen_sample
from sgs.tasks import SOLVER_PROMPT, TASKS, Sorting, parse_sorted

FILES = ["sorting_val_100", "sorting_1000", "sorting_1000_L8-128", "sorting_1000_L8-16-128",
         "sorting_1000_L8-16-32-128"]
P = [5, 3, 9, 1]


def test_generation():
    rng = random.Random(0)
    s = gen_sample(8, rng, 0, 10000)
    print(f"  length=8: list={s['list']} -> sorted={s['sorted']}")
    assert s["list"] != s["sorted"] == sorted(s["list"])


def test_data_files():
    val = {tuple(r["list"]) for r in read_jsonl(f"data/sorting/{FILES[0]}.jsonl")}
    for name in FILES:
        rows = read_jsonl(f"data/sorting/{name}.jsonl")
        print(f"  {name:<28} rows={len(rows):>4} lengths={sorted({len(r['list']) for r in rows})}")
        for r in rows:
            assert r["sorted"] == sorted(r["list"]) != r["list"]
        if name != FILES[0]:
            assert not {tuple(r["list"]) for r in rows} & val
    assert Sorting.load("data/sorting/sorting_val_100.jsonl")[0] == read_jsonl(
        "data/sorting/sorting_val_100.jsonl")[0]["list"]


def test_solver_prompt():
    show_prompt("Solver", Sorting.solver_messages(P))
    (bench_prompt,) = benchmark_defs("misc/sort_benchmark.py", ["PROMPT_TEMPLATE"])
    assert SOLVER_PROMPT == bench_prompt


OUTPUTS = [  # (label, model output, expected parsed value)
    ("correct boxed", "\\boxed{[1, 3, 5, 9]}", [1, 3, 5, 9]),
    ("no box: last [...] fallback", "first [5, 3] then [1, 3, 5, 9]", [1, 3, 5, 9]),
    ("box inside think wins (no </think> strip)", "<think>\\boxed{[9]}</think>\\boxed{[1, 3, 5, 9]}", [9]),
    ("wrong order", "\\boxed{[1, 5, 3, 9]}", [1, 5, 3, 9]),
    ("dropped element", "\\boxed{[1, 3, 5]}", [1, 3, 5]),
    ("nothing parsable", "I cannot sort this.", None),
]


def test_solver_parsing():
    (bench_parse,) = benchmark_defs("misc/sort_benchmark.py", ["parse_sorted"])
    print(f"  problem {P}, gold {Sorting.answer(P)}")
    for label, out, want in OUTPUTS:
        got, ok = parse_sorted(out), Sorting.verify(out, P)
        row(label, out, got, ok)
        assert got == want == bench_parse(out)
        assert ok == (want == [1, 3, 5, 9]) == Sorting.check(out, Sorting.answer(P))


CONJ_OUTPUTS = [
    ("valid", "<problem>\nSort the numbers in the list in ascending order.\nList: [4, 2, 7]\n</problem>\n"
              "<solution>\n[2, 4, 7]\n</solution>", [4, 2, 7]),
    ("single number", "<problem>\nList: [4]\n</problem>", None),
    ("no list", "<problem>\nSort some numbers.\n</problem>", "none-none"),
    ("no block", "List: [4, 2, 7]", "none-none"),
]


def test_conjecturer():
    show_prompt("Conjecturer", Sorting.conj_messages(P, 0, 10000))
    for label, out, want in CONJ_OUTPUTS:
        statement, problem = Sorting.conjecture(out)
        row(label, out, {"problem": problem}, problem is not None)
        if want == "none-none":
            assert (statement, problem) == (None, None)
        else:
            assert problem == want


def test_guide():
    from sgs.sgs_simple import parse_guide
    statement, _ = Sorting.conjecture(CONJ_OUTPUTS[0][1])
    show_prompt("Guide", Sorting.guide_messages(P, statement))
    out = "<relevance>4</relevance>\n<redundancy>0</redundancy>\n<complexity>0</complexity>"
    row("all tags", out, parse_guide(out), True)
    assert parse_guide(out) == (4, 0, 0)


def test_verl_path():
    import pandas as pd
    from sgs.solver_simple import compute_score, write_parquet
    with tempfile.TemporaryDirectory() as d:
        write_parquet([P], f"{d}/t.parquet", TASKS["sorting"])
        r = pd.read_parquet(f"{d}/t.parquet").iloc[0]
    print(f"  data_source={r.data_source} ground_truth={list(map(int, r.reward_model['ground_truth']))}")
    assert r.data_source == "sort_len_4"
    assert compute_score(r.data_source, "\\boxed{[1, 3, 5, 9]}", r.reward_model["ground_truth"]) == 1.0
    assert compute_score("val_sort_len_4", "\\boxed{[9, 5, 3, 1]}", [1, 3, 5, 9]) == 0.0


if __name__ == "__main__":
    run(globals())
