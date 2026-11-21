"""Shared helpers for the per-task pipeline tests (tests/test_<task>.py).

Each test prints a trace of what flows through the pipeline (input -> prompt ->
model output -> parsed -> gold -> ok) before asserting, so running a test file
directly doubles as a readable walkthrough of the task:

    $PYTHON tests/test_lcm.py
"""

import ast
import json
import re
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def banner(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show_prompt(name, messages):
    """Print a one-turn chat exactly as the model sees it (before the chat template)."""
    assert len(messages) == 1 and messages[0]["role"] == "user"
    print(f"--- {name} prompt (role=user) ---\n{messages[0]['content']}\n--- end {name} prompt ---")


def short(x, n=120):
    s = x if isinstance(x, str) else json.dumps(x)
    return s if len(s) <= n else s[:n] + f"... [{len(s)} chars]"


def row(label, output, parsed, ok):
    print(f"  [{label}]\n    output = {short(output)!r}\n    parsed = {short(parsed)}\n    ok     = {ok}")


def read_jsonl(path):
    return [json.loads(line) for line in open(REPO / path)]


def benchmark_defs(path, names):
    """Top-level assignments/functions `names` from a misc/*_benchmark.py, without
    importing it (those modules import vllm/torch at the top). Used to check that
    sgs/tasks.py copies their prompt and parser verbatim."""
    tree = ast.parse((REPO / path).read_text())
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names
                                                   for t in n.targets))
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "").startswith("_")
                                                   and getattr(t, "id", "").endswith("_RE")
                                                   for t in n.targets))]
    ns = {"re": re}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return [ns[n] for n in names]


def run(module_globals):
    """Run every test_* function of a test file in order; exit 1 on any failure."""
    tests = [(k, v) for k, v in module_globals.items() if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in tests:
        banner(name)
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            traceback.print_exc()
            print(f"FAIL {name}")
            failed.append(name)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)
