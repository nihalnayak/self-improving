"""Teacher prompt + parser.

Exact zero-shot prompt from SOAR paper page 20, Appendix B.1. Parses the enforced
`<question>...</question><answer>\\boxed{...}</answer>` format (§B.2 filters).
"""

import re
from typing import List, Optional

from soar.data import MathProblem
from soar.verify import extract_boxed, is_equiv


# Verbatim teacher prompt from paper page 20.
TEACHER_PROMPT = (
    "You are generating a new math problem for a math assistant.\n"
    "\n"
    "Allowed topics: Algebra, Counting and Probability, Geometry, Intermediate Algebra, "
    "Number Theory, Prealgebra, or Precalculus.\n"
    "\n"
    "\n"
    "Output rules (follow EXACTLY):\n"
    "    - Provide the final formatted problem in this structure: "
    "<question>[full math question]</question><answer>\\boxed{[answer]}</answer>\n"
    "    - Any explanations, steps, or reasoning about the problem goes OUTSIDE the "
    "<question> and <answer> tags.\n"
    "\n"
    "    Constraints:\n"
    "        - The problem must be original, challenging, and require at least 2--3 steps of reasoning.\n"
    "        - Output exactly ONE problem. You MUST follow the specified format EXACTLY.\n"
    "Begin now:"
)


def build_teacher_prompt(exemplars_pool: Optional[List[MathProblem]] = None, k: int = 0,
                         rng=None) -> str:
    """Return the teacher's user turn. Paper's prompt is zero-shot; `exemplars_pool`/`k`
    are kept for compatibility with the training loop signature but ignored by default.
    """
    return TEACHER_PROMPT


# ----- Parsing (paper §B.2 filters) -----

_QA_RE = re.compile(
    r"<question>\s*(?P<q>.*?)\s*</question>\s*<answer>\s*(?P<a>.*?)\s*</answer>",
    re.DOTALL | re.IGNORECASE,
)


def parse_teacher_output(text: str) -> Optional[MathProblem]:
    """Extract one MathProblem from a teacher completion. Returns None if any §B.2 filter
    fails:
        - must contain opening/closing <question></question><answer></answer> tags
        - answer block must contain \\boxed{...}
        - boxed content must be parseable (we approximate with a non-empty + math_verify
          self-equiv check; a malformed expression typically fails is_equiv even against
          itself after wrapping).
    """
    if not text:
        return None
    m = _QA_RE.search(text)
    if not m:
        return None
    q = m.group("q").strip()
    a_block = m.group("a").strip()
    boxed = extract_boxed(a_block)
    if not q or not boxed:
        return None
    # Self-equiv check: a well-formed expression must verify against itself.
    # (If math_verify isn't installed, is_equiv falls back to string equality -> always True
    # here, so this only filters when the package is present.)
    if not is_equiv(boxed, boxed):
        return None
    return MathProblem(qid="teacher", problem=q, answer=boxed, solution="", level="", subject="")


if __name__ == "__main__":
    good = "foo <question>What is 1+1?</question><answer>\\boxed{2}</answer> bar"
    p = parse_teacher_output(good)
    assert p is not None and p.problem == "What is 1+1?" and p.answer == "2", p

    assert parse_teacher_output("<question>Q</question>") is None
    assert parse_teacher_output("<question></question><answer>\\boxed{2}</answer>") is None
    assert parse_teacher_output("<question>Q</question><answer>no boxed</answer>") is None
    print("teacher.py: parsing sanity checks passed")
