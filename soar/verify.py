"""Extract an answer letter from a model completion and score it."""

import re
from typing import Optional

_LETTERS = "ABCDEFGHIJ"

# Patterns tried in order; first hit wins. The (?!\w) prevents matching the "i" of "is".
_PATTERNS = [
    re.compile(r"\\boxed\{\s*\(?([A-Ja-j])\)?\s*\}"),
    re.compile(r"[Aa]nswer\s*(?:is|:)\s*\(?([A-Ja-j])(?!\w)"),
    re.compile(r"\*\*\s*([A-Ja-j])\s*\*\*"),
    re.compile(r"\b([A-Ja-j])\b(?![\s\S]*\b[A-Ja-j]\b)"),   # last standalone letter anywhere
]


def extract_letter(text: str) -> Optional[str]:
    """Return 'A'..'J' or None if nothing parseable found."""
    if not text:
        return None
    for pat in _PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    return None


def score_answer(generated_text: str, gold_letter: str) -> float:
    pred = extract_letter(generated_text)
    return 1.0 if pred == gold_letter.upper() else 0.0
