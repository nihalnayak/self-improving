"""Answer extraction and scoring for open-ended MATH problems.

Reward scheme follows SOAR paper §B.3:
    120.0 if boxed(y) ∧ verify(boxed, gold)
    20.0  if boxed(y) ∧ ¬verify ∧ gold-as-substring-of-boxed
    10.0  if boxed(y) ∧ ¬verify ∧ gold-not-in-boxed
    0.0   otherwise
"""

from typing import Optional

try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    _HAS_MATH_VERIFY = True
except Exception:
    _HAS_MATH_VERIFY = False


def extract_boxed(text: str) -> Optional[str]:
    """Return the contents of the last `\\boxed{...}` in `text`, or None.

    Handles nested braces by scanning for balance after the `\\boxed{`.
    """
    if not text:
        return None
    key = r"\boxed{"
    last_start = text.rfind(key)
    if last_start == -1:
        return None
    i = last_start + len(key)
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
            out.append(c)
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(c)
        else:
            out.append(c)
        i += 1
    if depth != 0:
        return None
    return "".join(out).strip()


def _wrap_latex(expr: str) -> str:
    """math_verify.parse expects LaTeX-style strings; wrap in $...$ if needed."""
    expr = expr.strip()
    if not expr:
        return expr
    if expr.startswith("$") and expr.endswith("$"):
        return expr
    return f"${expr}$"


def is_equiv(pred: str, gold: str) -> bool:
    """Symbolic equivalence via math_verify. Falls back to string compare if the package
    isn't installed (keeps unit tests running in minimal envs).
    """
    if pred is None or gold is None:
        return False
    if pred.strip() == gold.strip():
        return True
    if not _HAS_MATH_VERIFY:
        return False
    try:
        p = _mv_parse(_wrap_latex(pred))
        g = _mv_parse(_wrap_latex(gold))
        return bool(_mv_verify(g, p))
    except Exception:
        return False


def success_math_answer(generated: str, gold: str) -> bool:
    """Binary 0/1 success: is there a boxed answer that math-verifies against gold?

    Used for fail@128 filtering and baseline accuracy evaluation.
    """
    boxed = extract_boxed(generated)
    if boxed is None:
        return False
    return is_equiv(boxed, gold)


def score_math_answer(generated: str, gold: str) -> float:
    """Paper's RLVR reward (§B.3). 120 / 20 / 10 / 0 tiers."""
    boxed = extract_boxed(generated)
    if boxed is None:
        return 0.0
    if is_equiv(boxed, gold):
        return 120.0
    if gold.strip() and gold.strip() in boxed:
        return 20.0
    return 10.0


if __name__ == "__main__":
    # Minimal sanity checks (don't require math_verify to be installed).
    assert extract_boxed(r"foo \boxed{42} bar") == "42"
    assert extract_boxed(r"a \boxed{\frac{1}{2}} b") == r"\frac{1}{2}"
    assert extract_boxed("no boxed here") is None
    # Nested braces
    assert extract_boxed(r"\boxed{a_{1}+b}") == "a_{1}+b"
    # Last-occurrence wins
    assert extract_boxed(r"\boxed{1} and later \boxed{2}") == "2"

    # String-equality fast path (works without math_verify)
    assert is_equiv("42", "42")
    assert not is_equiv("42", "43")
    assert success_math_answer(r"The answer is \boxed{42}", "42")
    assert not success_math_answer("no boxed", "42")

    # Reward tiers
    assert score_math_answer(r"\boxed{42}", "42") == 120.0
    assert score_math_answer(r"\boxed{foo42bar}", "42") == 20.0
    assert score_math_answer(r"\boxed{foo}", "42") == 10.0
    assert score_math_answer("no boxed", "42") == 0.0
    print("verify.py: all sanity checks passed")
