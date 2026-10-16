"""Teacher prompt (few-shot) + parser for teacher-generated problems."""

import random
import re
from typing import List, Optional

from soar.data import MCQ

_LETTERS = "ABCDEFGHIJ"

_INSTRUCTION = (
    "You are an expert problem-writer for a graduate-level MATH benchmark. "
    "Generate ONE new, original, challenging multiple-choice math problem similar to the examples below. "
    "Use EXACTLY this format with no extra commentary:\n\n"
    "Question: <one-paragraph question statement>\n\n"
    "Options:\n"
    "A. <option>\n"
    "B. <option>\n"
    "C. <option>\n"
    "D. <option>\n"
    "E. <option>\n"
    "F. <option>\n"
    "G. <option>\n"
    "H. <option>\n"
    "I. <option>\n"
    "J. <option>\n\n"
    "Answer: <single letter A-J>"
)


def _format_exemplar(q: MCQ) -> str:
    opt_lines = "\n".join(f"{_LETTERS[i]}. {o}" for i, o in enumerate(q.options))
    return (
        f"Question: {q.question}\n\n"
        f"Options:\n{opt_lines}\n\n"
        f"Answer: {q.answer}"
    )


def build_teacher_prompt(exemplars_pool: List[MCQ], k: int, rng: random.Random) -> str:
    """Build the teacher's user turn: instruction + k random exemplars + 'now generate'."""
    picks = rng.sample(exemplars_pool, k=min(k, len(exemplars_pool)))
    examples = "\n\n---\n\n".join(_format_exemplar(q) for q in picks)
    return (
        f"{_INSTRUCTION}\n\n"
        f"--- Examples ---\n\n{examples}\n\n"
        f"--- Now generate a new problem in the same format. ---"
    )


# ---- parsing ----

_Q_RE = re.compile(r"Question:\s*(.*?)\s*\n\s*\n\s*Options:", re.DOTALL)
_OPT_RE = re.compile(r"^\s*([A-J])[\.\)]\s*(.+?)\s*$", re.MULTILINE)
_A_RE = re.compile(r"Answer:\s*\(?([A-J])\)?", re.IGNORECASE)


def parse_teacher_output(text: str) -> Optional[MCQ]:
    """Parse a teacher-generated completion into an MCQ. Return None if any part is malformed."""
    if not text:
        return None
    qm = _Q_RE.search(text)
    am = _A_RE.search(text)
    if not qm or not am:
        return None
    question = qm.group(1).strip()
    answer = am.group(1).upper()

    # Options block is between "Options:" and "Answer:" (or end)
    tail = text[qm.end():]
    ans_pos = _A_RE.search(tail)
    opt_block = tail[:ans_pos.start()] if ans_pos else tail
    opts = _OPT_RE.findall(opt_block)
    # Collect in letter order; require at least 4 options and the answer letter present
    opt_map = {letter: content for letter, content in opts}
    ordered = []
    for L in _LETTERS:
        if L in opt_map:
            ordered.append(opt_map[L])
        else:
            break
    if len(ordered) < 4 or answer not in opt_map:
        return None

    try:
        answer_index = _LETTERS.index(answer)
    except ValueError:
        return None
    if answer_index >= len(ordered):
        return None

    return MCQ(qid=-1, question=question, options=ordered, answer=answer, answer_index=answer_index)
