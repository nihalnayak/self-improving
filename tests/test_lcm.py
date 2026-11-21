"""LCM task pipeline: data -> Solver prompt -> output parsing -> Conjecturer -> Guide -> verl.

    $PYTHON tests/test_lcm.py
"""

import filecmp
import glob
import math
import random
import tempfile

from _trace import benchmark_defs, read_jsonl, row, run, show_prompt

from misc.gen_lcm_data import gen_sample
from sgs.tasks import LCM, LCM_SOLVER_PROMPT, TASKS, Multiplication, parse_product

FILES = ["lcm_val_100", "lcm_1000", "lcm_1000_L2-32", "lcm_1000_L2-4-32", "lcm_1000_L2-4-8-32"]
WO = [f.replace("lcm_", "lcm_wo_coprime_", 1) for f in FILES]
P = [12, 18]                                  # gcd 6 -> lcm 36 (product would be 216)
TARGET = [4820, 7350]                         # a level-4 target


def pairs(name):
    return {tuple(sorted((r["number1"], r["number2"]))) for r in read_jsonl(f"data/lcm/{name}.jsonl")}


def test_generation():
    rng = random.Random(0)
    for L in (2, 4, 8, 16, 32):
        s = gen_sample(L, rng)
        print(f"  digits={L:>2}: lcm({s['number1']}, {s['number2']}) = {s['lcm']}")
        assert len(str(s["number1"])) == len(str(s["number2"])) == L
        assert s["lcm"] == math.lcm(s["number1"], s["number2"])


def test_data_files():
    for files, wo in ((FILES, False), (WO, True)):
        val = pairs(files[0])
        for name in files:
            rows = read_jsonl(f"data/lcm/{name}.jsonl")
            levels = sorted({len(str(r["number1"])) for r in rows})
            coprime = sum(math.gcd(r["number1"], r["number2"]) == 1 for r in rows)
            print(f"  {name:<32} rows={len(rows):>4} digits={levels} coprime={coprime}")
            for r in rows:
                assert len(str(r["number1"])) == len(str(r["number2"]))
                assert r["lcm"] == math.lcm(r["number1"], r["number2"])
            assert len(pairs(name)) == len(rows), "duplicate (unordered) pair"
            assert levels == ([2, 32] if "L2-32" in name else [2, 4, 32] if "L2-4-32" in name
                              else [2, 4, 8, 32] if "L2-4-8-32" in name else [2, 4, 8, 16, 32])
            if wo:
                assert coprime == 0
            if name != files[0]:
                assert not pairs(name) & val, "train pair appears in val"
    # load() must keep every operand exact: pyarrow (load_dataset) rounds 16/32-digit ones to float64
    assert not issubclass(LCM, Multiplication), "LCM must not inherit Multiplication.load"
    for path in sorted(glob.glob("data/lcm/*.jsonl")):
        rows = read_jsonl(path)
        assert LCM.load(path) == [[r["number1"], r["number2"]] for r in rows], f"inexact load: {path}"
    print("  LCM.load is exact on every data/lcm/*.jsonl")
    # lcm_wo_coprime: its own task, inheriting everything from LCM, on copies of the data/lcm files
    wo_task = TASKS["lcm_wo_coprime"]
    assert issubclass(wo_task, LCM) and wo_task.source == "lcm_wo_coprime"
    copies = sorted(glob.glob("data/lcm_wo_coprime/*.jsonl"))
    assert [p.split("/")[-1] for p in copies] == [f"{n}.jsonl" for n in sorted(WO)], copies
    for path in copies:
        assert filecmp.cmp(path, f"data/lcm/{path.split('/')[-1]}", shallow=False), f"copy differs: {path}"
        assert wo_task.load(path) == [[r["number1"], r["number2"]] for r in read_jsonl(path)]
    print(f"  data/lcm_wo_coprime/: {len(copies)} files identical to data/lcm/, exact load")
    # answers are recomputed from the operands (str for int64 safety)
    targets = LCM.load("data/lcm/lcm_1000.jsonl")
    assert targets[0] == [read_jsonl("data/lcm/lcm_1000.jsonl")[0][k] for k in ("number1", "number2")]
    big = max(targets, key=lambda p: p[0])
    print(f"  largest target {big} -> answer {LCM.answer(big)!r} (level {LCM.level(big)})")
    assert isinstance(LCM.answer(big), str) and int(LCM.answer(big)) == math.lcm(*big)


def test_solver_prompt():
    show_prompt("Solver", LCM.solver_messages(P))
    (bench_prompt,) = benchmark_defs("misc/lcm_benchmark.py", ["PROMPT_TEMPLATE"])
    assert LCM_SOLVER_PROMPT == bench_prompt, "Solver prompt must match misc/lcm_benchmark.py"


OUTPUTS = [  # (label, model output, expected parsed value)
    ("correct boxed", "\\boxed{36}", 36),
    ("reasoning then boxed", "gcd(12, 18) = 6, so lcm = 12*18/6 = 36.\n\\boxed{36}", 36),
    ("think trace, then boxed", "<think>maybe \\boxed{216}?</think>\n\\boxed{36}", 36),
    ("product instead of lcm", "\\boxed{216}", 216),
    ("no box: last integer", "The LCM of 12 and 18 is 36", 36),
    ("two boxes: first wins", "\\boxed{36} or \\boxed{72}", 36),
    ("thousands separators", "\\boxed{1,234}", 1234),   # benchmark parse_lcm: None
    ("bad comma grouping", "\\boxed{1,2}", None),
    ("non-integer in box", "\\boxed{36.0}", None),
    ("empty", "", None),
    ("truncated mid-box", "so the answer is \\boxed{3", 3),
]


# labels where parse_product intentionally differs from misc/lcm_benchmark.parse_lcm
DIVERGES = {"thousands separators"}


def test_solver_parsing():
    (parse_lcm,) = benchmark_defs("misc/lcm_benchmark.py", ["parse_lcm"])
    gold = LCM.answer(P)
    print(f"  problem {P}, gold {gold!r}")
    for label, out, want in OUTPUTS:
        got, ok = parse_product(out), LCM.verify(out, P)
        row(label, out, got, ok)
        assert got == want
        assert (got == parse_lcm(out)) != (label in DIVERGES), "only DIVERGES may differ from parse_lcm"
        assert ok == (want == 36) == LCM.check(out, gold)


CONJ_OUTPUTS = [  # (label, conjecturer output, expected problem)
    ("valid", "<problem>\nCompute the least common multiple (LCM) of the following two integers.\n"
              "number1: 84\nnumber2: 126\n</problem>\n<solution>\n252\n</solution>", [84, 126]),
    ("wrong solution ignored", "<problem>\nnumber1: 84\nnumber2: 126\n</problem>\n<solution>\n1\n"
                               "</solution>", [84, 126]),
    ("one-digit operand", "<problem>\nnumber1: 8\nnumber2: 126\n</problem>", None),
    ("leading zero", "<problem>\nnumber1: 084\nnumber2: 126\n</problem>", "none-none"),
    ("missing number2", "<problem>\nnumber1: 84\n</problem>", "none-none"),
    ("no block", "number1: 84\nnumber2: 126", "none-none"),
    ("two blocks: last wins", "<problem>number1: 11\nnumber2: 13</problem>\n"
                              "<problem>number1: 36\nnumber2: 48</problem>", [36, 48]),
]


def test_conjecturer():
    show_prompt("Conjecturer", LCM.conj_messages(TARGET, 0, 10000))
    for label, out, want in CONJ_OUTPUTS:
        statement, problem = LCM.conjecture(out)
        gold = LCM.answer(problem) if problem else None
        row(label, out, {"problem": problem, "gold": gold}, problem is not None)
        if want == "none-none":
            assert (statement, problem) == (None, None)
        else:
            assert problem == want and statement is not None


def test_guide():
    from sgs.sgs_simple import guide_score, parse_guide
    statement, _ = LCM.conjecture(CONJ_OUTPUTS[0][1])
    show_prompt("Guide", LCM.guide_messages(TARGET, statement))
    for label, out, want in [
        ("all tags", "...\n<relevance>4</relevance>\n<redundancy>0</redundancy>\n<complexity>1</complexity>", (4, 0, 1)),
        ("missing tag", "<relevance>4</relevance>\n<redundancy>0</redundancy>", None),
        ("out of range", "<relevance>7</relevance><redundancy>0</redundancy><complexity>1</complexity>", None),
    ]:
        scores = parse_guide(out)
        row(label, out, {"scores": scores, "R_guide": guide_score(scores)}, scores is not None)
        assert scores == want


def test_verl_path():
    import pandas as pd
    from sgs.solver_simple import compute_score, write_parquet
    targets = LCM.load("data/lcm/lcm_val_100.jsonl")
    with tempfile.TemporaryDirectory() as d:
        write_parquet(targets, f"{d}/val.parquet", TASKS["lcm"], source_prefix="val_")
        df = pd.read_parquet(f"{d}/val.parquet")
    r = df.iloc[-1]
    print(f"  data_source={r.data_source} ground_truth={r.reward_model['ground_truth']!r}")
    assert r.data_source == "val_lcm_len_32" and int(r.reward_model["ground_truth"]) == math.lcm(*targets[-1])
    for src, out, want in [("lcm_len_2", "\\boxed{36}", 1.0), ("val_lcm_len_2", "\\boxed{216}", 0.0)]:
        score = compute_score(src, out, LCM.answer(P))
        print(f"  compute_score({src!r}, {out!r}) = {score}")
        assert score == want
    assert compute_score(r.data_source, f"\\boxed{{{r.reward_model['ground_truth']}}}",
                         r.reward_model["ground_truth"]) == 1.0
    # lcm_wo_coprime: own data_source stem, routed back to its task by compute_score
    wo = TASKS["lcm_wo_coprime"]
    wo_targets = wo.load("data/lcm_wo_coprime/lcm_wo_coprime_val_100.jsonl")
    with tempfile.TemporaryDirectory() as d:
        write_parquet(wo_targets, f"{d}/val.parquet", wo, source_prefix="val_")
        w = pd.read_parquet(f"{d}/val.parquet").iloc[-1]
    print(f"  data_source={w.data_source} ground_truth={w.reward_model['ground_truth']!r}")
    assert w.data_source == "val_lcm_wo_coprime_len_32"
    gt = w.reward_model["ground_truth"]
    assert compute_score(w.data_source, f"\\boxed{{{gt}}}", gt) == 1.0
    assert compute_score(w.data_source, f"\\boxed{{{int(gt) + 1}}}", gt) == 0.0


if __name__ == "__main__":
    run(globals())
