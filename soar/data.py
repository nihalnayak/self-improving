"""MathProblem dataclass + loader for fail@128 subsets produced by misc.build_fail128."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple


@dataclass
class MathProblem:
    """One open-ended math problem with a gold answer string."""
    qid: str              # unique id (sha1 of problem text, or original dataset id)
    problem: str          # the question text
    answer: str           # gold final answer (what should be inside \boxed{})
    solution: str = ""    # optional full solution (from the source dataset)
    level: str = ""       # optional difficulty label (e.g. "Level 5")
    subject: str = ""     # optional topic (e.g. "Algebra")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "MathProblem":
        return cls(
            qid=str(d["qid"]),
            problem=d["problem"],
            answer=d["answer"],
            solution=d.get("solution", ""),
            level=d.get("level", ""),
            subject=d.get("subject", ""),
        )


def _read_jsonl(path: Path) -> List[MathProblem]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(MathProblem.from_dict(json.loads(line)))
    return out


def load_fail128_split(dir: str) -> Tuple[List[MathProblem], List[MathProblem]]:
    """Return (train, test) MathProblem lists from a fail128 output directory."""
    d = Path(dir)
    train = _read_jsonl(d / "train.jsonl")
    test = _read_jsonl(d / "test.jsonl")
    return train, test


def format_math_prompt(problem: MathProblem) -> str:
    """Return the user-turn question string. The surrounding student prompt template is
    applied by `soar.models.student_chat_prompt`.
    """
    return problem.problem
