"""Load MMLU-Pro, filter one subject, split into Q_eval (teacher reward) + Q_train (teacher exemplars)."""

from dataclasses import dataclass
from typing import List

from datasets import load_dataset


@dataclass
class MCQ:
    """One multiple-choice question. `options` is a list of ~10 strings. `answer` is a letter A..J."""
    qid: int
    question: str
    options: List[str]
    answer: str              # letter 'A'..'J'
    answer_index: int        # 0..len(options)-1


def _row_to_mcq(row) -> MCQ:
    return MCQ(
        qid=int(row["question_id"]),
        question=row["question"],
        options=list(row["options"]),
        answer=row["answer"],
        answer_index=int(row["answer_index"]),
    )


def load_mmlu_pro_split(subject: str, n_eval: int, seed: int) -> tuple[List[MCQ], List[MCQ]]:
    """Return (Q_eval, Q_train_exemplars). Q_eval is the fixed set used to compute teacher reward.
    Q_train is just a pool to draw few-shot exemplars from when prompting the teacher.
    """
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    ds = ds.filter(lambda x: x["category"] == subject)
    ds = ds.shuffle(seed=seed)
    rows = [_row_to_mcq(r) for r in ds]
    q_eval = rows[:n_eval]
    q_train = rows[n_eval:]
    return q_eval, q_train


def format_mcq_prompt(q: MCQ) -> str:
    """Format a multiple-choice question for the student to answer."""
    letters = "ABCDEFGHIJ"
    opt_lines = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(q.options))
    return (
        f"Question: {q.question}\n\n"
        f"Options:\n{opt_lines}\n\n"
        f"Think briefly, then answer with the final letter on a line like: Answer: X"
    )
