"""Task definitions for the SGS trainers: prompts, verifiers and data loading.

A problem is a plain JSON list: the integer list for sorting, [number1, number2]
for multiplication and LCM, [inorder, postorder] for preorder traversal. Trainers pick a task with `--task` and go through
`TASKS[name]`, which provides:

    load(path)                   jsonl -> problems
    solver_messages(p)           Solver chat
    answer(p)                    gold answer (stored as verl ground_truth)
    check(text, gold)            bool: Solver completion matches gold
    verify(text, p)              check(text, answer(p))
    level(p)                     difficulty bucket for metrics (list length / digits / nodes)
    conj_messages(p, lo, hi)     Conjecturer chat
    conjecture(text)             (statement, problem) or (None, None) if invalid
    guide_messages(p, statement) Guide chat
    source                       verl data_source stem ("sort" / "mult" / "lcm" / "preorder")

Sorting prompts and parsers are unchanged from before this file existed (moved from
sgs/common.py and sgs/sgs_simple.py, which re-export them).
"""

import json
import math
import re


def user_msg(text):
    """One-turn chat. verl applies the chat template to this, never us."""
    return [{"role": "user", "content": text}]


# =============================================================================
# Sorting
# =============================================================================

# Solver: verbatim from misc/sort_benchmark.py.
SOLVER_PROMPT = (
    "Sort the numbers in the list in ascending order.\n\n"
    "List: {list_str}\n\n"
    "Return only the sorted list on a single line, enclosed in \\boxed{{}}, "
    "e.g. \\boxed{{[1, 2, 3]}}."
)


def solver_messages(xs):
    return user_msg(SOLVER_PROMPT.format(list_str=str(xs)))


_BOXED_RE = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)
_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _nums(s):
    nums = _NUM_RE.findall(s)
    return [float(n) for n in nums] if nums else None


def parse_sorted(text):
    """Solver answer: first \\boxed{...}, else last [...] (misc/sort_benchmark.py)."""
    m = _BOXED_RE.search(text)
    if m:
        return _nums(m.group(1))
    brackets = _BRACKET_RE.findall(text)
    return _nums(brackets[-1]) if brackets else None


# Conjecturer: verbatim from misc/sort_proposer.py. `example_list` is the
# unsolved target x, which is how the Conjecturer is conditioned on x.
CONJ_PROMPT = (
    "I am learning to sort lists of integers in [{lo}, {hi}] in "
    "ascending order. This is the task I ultimately want to solve. "
    "Here is one example sorting problem for reference:\n\n"
    "List: {example_list}\n\n"
    "To make progress, I want to first practice on a simpler version: "
    "a shorter and easier sorting problem. "
    "Learning to solve this simpler task will be a "
    "stepping stone toward solving the more challenging "
    "task.\n\n"
    "Generate one such simpler sorting problem and its solution. "
    "The input list must NOT already be sorted and must NOT be an "
    "obvious canonical example. The solution must contain exactly "
    "the same integers, including duplicates, in ascending order.\n\n"
    "Return exactly the following two blocks, replacing the placeholder "
    "lists with actual integers:\n\n"
    "<problem>\n"
    "Sort the numbers in the list in ascending order.\n"
    "List: [a, b, c, ...]\n"
    "</problem>\n"
    "<solution>\n"
    "[sorted integers]\n"
    "</solution>\n\n"
    "Write each list on a single line using comma-separated integers. "
    "Do not include reasoning, explanations, Markdown fences, LaTeX, "
    "or any text outside these two blocks."
)

# Guide: paper §E.3 (relevance 0-5, redundancy 0/1, complexity 0-4), with the
# Lean "lemma / premises / conclusion" wording mapped onto sorting problems.
GUIDE_PROMPT = (
    "You are a sorting expert. Here is a seed sorting problem:\n\n"
    "<seed>\n{seed}\n</seed>\n\n"
    "Here is a simpler sorting problem that is supposed to be useful for "
    "solving the above problem. It can be useful in that either it is a "
    "smaller version of the seed problem that can be directly used as a "
    "stepping stone, or it is a related problem that plausibly requires "
    "similar skills to solve.\n\n"
    "<problem>\n{conjecture}\n</problem>\n\n"
    "Please rate the relevance of the problem to the seed problem on a scale "
    "of 0 to 5, where 0 is \"not at all related\" and 5 is \"very useful for "
    "solving the seed problem.\" If the problem is trivial, you should give "
    "it a low score.\n\n"
    "Here is a rubric for how to score the problem:\n"
    "- 0: The problem is not at all related to the seed problem and is "
    "trivial to solve (e.g. the list has two or fewer numbers or is already "
    "sorted). OR the problem is identical to the seed problem (the same "
    "numbers, possibly reordered).\n"
    "- 1: The problem is not at all related to the seed problem (e.g. it is "
    "not a problem of sorting a list of integers in ascending order).\n"
    "- 2: The problem is related to the seed problem in that it is a sorting "
    "problem, but it is not directly useful for solving the seed problem "
    "(e.g. it is not simpler than the seed problem, or its numbers look "
    "nothing like the seed's).\n"
    "- 3: The problem is related to the seed problem and may be useful for "
    "solving the seed problem.\n"
    "- 4: The problem is directly useful for solving the seed problem. That "
    "is, if the problem was solved, the seed problem would be easier to solve "
    "(e.g. a shorter list whose numbers have similar size, duplicates and "
    "ordering patterns).\n"
    "- 5: The problem is very useful for solving the seed problem, and "
    "solving it will dramatically reduce the difficulty of the seed "
    "problem.\n\n"
    "Next decide how redundant the problem statement is. Rate this as 0 or 1:\n"
    "- 0: There is nothing redundant in the problem statement.\n"
    "- 1: There are redundant parts. That is, instructions, constraints or "
    "text that are not needed to state the sorting task.\n\n"
    "Next decide if the problem statement is overly complex. Rate this on a "
    "score of 0 to 4:\n"
    "- 0: The problem is minimally complex. A single instruction to sort a "
    "single list that is maximally clear.\n"
    "- 1: The problem has low complexity. Multiple related parts, but they "
    "form a cohesive sorting task.\n"
    "- 2: The problem has moderate complexity. Contains a few extra "
    "conditions or alternatives that are all on-theme.\n"
    "- 3: The problem has high complexity. Contains multiple unrelated tasks "
    "or conditions, or nested instructions that obscure the main sorting "
    "task.\n"
    "- 4: The problem has very high complexity. Many largely unrelated tasks "
    "or deeply nested instructions that are hard to parse.\n\n"
    "End your response with the three scores in exactly this format:\n"
    "<relevance>N</relevance>\n"
    "<redundancy>N</redundancy>\n"
    "<complexity>N</complexity>"
)

# How a target is shown to the Guide (same statement format the Conjecturer emits).
SEED_STATEMENT = "Sort the numbers in the list in ascending order.\nList: {list_str}"


def conj_messages(target, lo, hi):
    return user_msg(CONJ_PROMPT.format(lo=lo, hi=hi, example_list=str(target)))


def guide_messages(target, conj_statement):
    seed = SEED_STATEMENT.format(list_str=str(target))
    return user_msg(GUIDE_PROMPT.format(seed=seed, conjecture=conj_statement))


_PROBLEM_RE = re.compile(r"<problem>(.*?)</problem>", re.DOTALL | re.IGNORECASE)
def parse_conjecture(text):
    """(statement, list) from the last non-empty <problem> block (misc/sort_proposer.py).

    Returns (None, None) if there is no block or no bracketed list inside it.
    """
    blocks = [b.strip() for b in _PROBLEM_RE.findall(text) if b.strip()]
    if not blocks:
        return None, None
    brackets = _BRACKET_RE.findall(blocks[-1])
    xs = _nums(brackets[-1]) if brackets else None
    return (blocks[-1], xs) if xs else (None, None)


class Sorting:
    source = "sort"

    @staticmethod
    def load(path):
        """Target lists from a jsonl with a `list` field per row."""
        from datasets import load_dataset
        return [list(row["list"]) for row in load_dataset("json", data_files=path, split="train")]

    solver_messages = staticmethod(solver_messages)
    answer = staticmethod(sorted)

    @staticmethod
    def check(text, gold):
        return parse_sorted(text) == [int(x) for x in gold]

    @staticmethod
    def verify(text, p):
        return parse_sorted(text) == sorted(p)

    level = staticmethod(len)
    conj_messages = staticmethod(conj_messages)

    @staticmethod
    def conjecture(text):
        statement, xs = parse_conjecture(text)
        return (statement, xs) if xs is not None and len(xs) >= 2 else (statement, None)

    guide_messages = staticmethod(guide_messages)


# =============================================================================
# Multiplication
# =============================================================================

# Solver prompt + parser: verbatim from misc/multiplication_benchmark.py (copied, not
# imported: that module imports vllm/torch at the top).
MULT_SOLVER_PROMPT = (
    "Multiply the following two integers.\n\n"
    "number1: {number1}\n"
    "number2: {number2}\n\n"
    "Return only the product on a single line, enclosed in \\boxed{{}}, "
    "e.g. \\boxed{{1234}}."
)

_MULT_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_STANDALONE_INTEGER_RE = re.compile(r"(?<![\w.+-])[0-9]+(?!\w|\.[0-9])")
_INTEGER_RE = re.compile(r"[0-9]+")
_THOUSANDS_RE = re.compile(r"[0-9]{1,3}(?:,[0-9]{3})+")


def parse_product(text):
    """Parse the multiplicative product from the model output.

    Prefer the first \\boxed{...}; fall back to the last complete integer.
    Returns None if nothing parses to an integer. Unlike misc/{multiplication,lcm}_benchmark.py,
    well-formed thousands separators are accepted (\\boxed{5,312,856} -> 5312856); other commas
    (\\boxed{1,2}, \\boxed{5,31,2856}) still fail.
    """

    # Only consider text after the thinking trace
    text = text.rsplit("</think>", 1)[-1].strip()

    # Get the first boxed answer
    m = _MULT_BOXED_RE.search(text)
    if m:
        inside = m.group(1).strip()
    else:
        integers = _STANDALONE_INTEGER_RE.findall(text)
        if not integers:
            return None
        inside = integers[-1]

    if _THOUSANDS_RE.fullmatch(inside):
        inside = inside.replace(",", "")
    if not _INTEGER_RE.fullmatch(inside):
        return None
    try:
        return int(inside)
    except ValueError:
        return None


# Conjecturer: sorting's CONJ_PROMPT with the domain mapped onto multiplication.
MULT_CONJ_PROMPT = (
    "I am learning to multiply two positive integers that have the same "
    "number of digits. This is the task I ultimately want to solve. "
    "Here is one example multiplication problem for reference:\n\n"
    "number1: {number1}\n"
    "number2: {number2}\n\n"
    "To make progress, I want to first practice on a simpler version: "
    "a smaller and easier multiplication problem. "
    "Learning to solve this simpler task will be a "
    "stepping stone toward solving the more challenging "
    "task.\n\n"
    "Generate one such simpler multiplication problem and its solution. "
    "Each integer must have at least two digits and must NOT be an "
    "obvious canonical example (such as a power of 10 or a number made of "
    "one repeated digit). The solution must be the exact product of the "
    "two integers.\n\n"
    "Return exactly the following two blocks, replacing the placeholders "
    "with actual integers:\n\n"
    "<problem>\n"
    "Multiply the following two integers.\n"
    "number1: a\n"
    "number2: b\n"
    "</problem>\n"
    "<solution>\n"
    "product\n"
    "</solution>\n\n"
    "Write each integer using digits only, with no commas, spaces or other "
    "separators. Do not include reasoning, explanations, Markdown fences, "
    "LaTeX, or any text outside these two blocks."
)

# Guide: sorting's GUIDE_PROMPT (paper §E.3) with the domain mapped onto multiplication.
MULT_GUIDE_PROMPT = (
    "You are a multiplication expert. Here is a seed multiplication problem:\n\n"
    "<seed>\n{seed}\n</seed>\n\n"
    "Here is a simpler multiplication problem that is supposed to be useful for "
    "solving the above problem. It can be useful in that either it is a "
    "smaller version of the seed problem that can be directly used as a "
    "stepping stone, or it is a related problem that plausibly requires "
    "similar skills to solve.\n\n"
    "<problem>\n{conjecture}\n</problem>\n\n"
    "Please rate the relevance of the problem to the seed problem on a scale "
    "of 0 to 5, where 0 is \"not at all related\" and 5 is \"very useful for "
    "solving the seed problem.\" If the problem is trivial, you should give "
    "it a low score.\n\n"
    "Here is a rubric for how to score the problem:\n"
    "- 0: The problem is not at all related to the seed problem and is "
    "trivial to solve (e.g. an integer has one digit, or is 0, 1 or a power "
    "of 10). OR the problem is identical to the seed problem (the same two "
    "integers, possibly swapped).\n"
    "- 1: The problem is not at all related to the seed problem (e.g. it is "
    "not a problem of multiplying two integers).\n"
    "- 2: The problem is related to the seed problem in that it is a "
    "multiplication problem, but it is not directly useful for solving the "
    "seed problem (e.g. it is not simpler than the seed problem, or its "
    "integers look nothing like the seed's).\n"
    "- 3: The problem is related to the seed problem and may be useful for "
    "solving the seed problem.\n"
    "- 4: The problem is directly useful for solving the seed problem. That "
    "is, if the problem was solved, the seed problem would be easier to solve "
    "(e.g. integers with fewer digits but similar digit patterns and "
    "carries).\n"
    "- 5: The problem is very useful for solving the seed problem, and "
    "solving it will dramatically reduce the difficulty of the seed "
    "problem.\n\n"
    "Next decide how redundant the problem statement is. Rate this as 0 or 1:\n"
    "- 0: There is nothing redundant in the problem statement.\n"
    "- 1: There are redundant parts. That is, instructions, constraints or "
    "text that are not needed to state the multiplication task.\n\n"
    "Next decide if the problem statement is overly complex. Rate this on a "
    "score of 0 to 4:\n"
    "- 0: The problem is minimally complex. A single instruction to multiply "
    "two integers that is maximally clear.\n"
    "- 1: The problem has low complexity. Multiple related parts, but they "
    "form a cohesive multiplication task.\n"
    "- 2: The problem has moderate complexity. Contains a few extra "
    "conditions or alternatives that are all on-theme.\n"
    "- 3: The problem has high complexity. Contains multiple unrelated tasks "
    "or conditions, or nested instructions that obscure the main "
    "multiplication task.\n"
    "- 4: The problem has very high complexity. Many largely unrelated tasks "
    "or deeply nested instructions that are hard to parse.\n\n"
    "End your response with the three scores in exactly this format:\n"
    "<relevance>N</relevance>\n"
    "<redundancy>N</redundancy>\n"
    "<complexity>N</complexity>"
)

# How a target is shown to the Guide (same statement format the Conjecturer emits).
MULT_SEED_STATEMENT = "Multiply the following two integers.\nnumber1: {number1}\nnumber2: {number2}"

# Operands without leading zeros; a leading "0" fails the match -> invalid.
_OPERAND_RE = {key: re.compile(rf"{key}\s*:\s*([1-9][0-9]*)\b", re.IGNORECASE)
               for key in ("number1", "number2")}


class Multiplication:
    source = "mult"

    @staticmethod
    def load(path):
        """[number1, number2] per row. The jsonl `product` column is ignored: pyarrow
        reads products above 2**63 as float64, so it is recomputed from the operands.
        Exact only while operands stay below ~15 digits (pyarrow float64 above 2**63, and
        a column with one such value is all float64); for larger operands read the jsonl
        with the json module as LCM.load does."""
        from datasets import load_dataset
        ds = load_dataset("json", data_files=path, split="train")
        return [[int(a), int(b)] for a, b in zip(ds["number1"], ds["number2"])]

    @staticmethod
    def solver_messages(p):
        return user_msg(MULT_SOLVER_PROMPT.format(number1=p[0], number2=p[1]))

    @staticmethod
    def answer(p):
        # str: 10-digit x 10-digit products overflow int64 in the verl parquet.
        return str(p[0] * p[1])

    @staticmethod
    def check(text, gold):
        return parse_product(text) == int(gold)

    @staticmethod
    def verify(text, p):
        return parse_product(text) == p[0] * p[1]

    @staticmethod
    def level(p):
        return len(str(p[0]))

    @staticmethod
    def conj_messages(p, lo, hi):
        """`lo`/`hi` are sorting's integer range; unused here."""
        return user_msg(MULT_CONJ_PROMPT.format(number1=p[0], number2=p[1]))

    @staticmethod
    def conjecture(text):
        """(statement, [a, b]) from the last non-empty <problem> block. The problem is
        None unless both operands parse and have at least two digits."""
        blocks = [b.strip() for b in _PROBLEM_RE.findall(text) if b.strip()]
        if not blocks:
            return None, None
        ms = [_OPERAND_RE[key].search(blocks[-1]) for key in ("number1", "number2")]
        if not all(ms):
            return None, None
        p = [int(m.group(1)) for m in ms]
        return blocks[-1], (p if min(p) >= 10 else None)

    @staticmethod
    def guide_messages(p, conj_statement):
        seed = MULT_SEED_STATEMENT.format(number1=p[0], number2=p[1])
        return user_msg(MULT_GUIDE_PROMPT.format(seed=seed, conjecture=conj_statement))


# =============================================================================
# Least common multiple
# =============================================================================

# Solver prompt: verbatim from misc/lcm_benchmark.py. Its parse_lcm is character-
# identical to parse_product above, so the verifier reuses parse_product.
LCM_SOLVER_PROMPT = (
    "Compute the least common multiple (LCM) of the following two integers.\n\n"
    "number1: {number1}\n"
    "number2: {number2}\n\n"
    "Return only the LCM on a single line, enclosed in \\boxed{{}}, "
    "e.g. \\boxed{{1234}}."
)

# Conjecturer: MULT_CONJ_PROMPT with the domain mapped onto LCM.
LCM_CONJ_PROMPT = (
    "I am learning to compute the least common multiple (LCM) of two positive "
    "integers that have the same number of digits. This is the task I "
    "ultimately want to solve. Here is one example LCM problem for reference:\n\n"
    "number1: {number1}\n"
    "number2: {number2}\n\n"
    "To make progress, I want to first practice on a simpler version: "
    "a smaller and easier LCM problem. "
    "Learning to solve this simpler task will be a "
    "stepping stone toward solving the more challenging "
    "task.\n\n"
    "Generate one such simpler LCM problem and its solution. "
    "Each integer must have at least two digits and must NOT be an "
    "obvious canonical example (such as a power of 10, a number made of "
    "one repeated digit, or one integer that divides the other). The "
    "solution must be the exact least common multiple of the two integers.\n\n"
    "Return exactly the following two blocks, replacing the placeholders "
    "with actual integers:\n\n"
    "<problem>\n"
    "Compute the least common multiple (LCM) of the following two integers.\n"
    "number1: a\n"
    "number2: b\n"
    "</problem>\n"
    "<solution>\n"
    "lcm\n"
    "</solution>\n\n"
    "Write each integer using digits only, with no commas, spaces or other "
    "separators. Do not include reasoning, explanations, Markdown fences, "
    "LaTeX, or any text outside these two blocks."
)

# Guide: MULT_GUIDE_PROMPT (paper §E.3) with the domain mapped onto LCM.
LCM_GUIDE_PROMPT = (
    "You are an LCM expert. Here is a seed least common multiple (LCM) problem:\n\n"
    "<seed>\n{seed}\n</seed>\n\n"
    "Here is a simpler LCM problem that is supposed to be useful for "
    "solving the above problem. It can be useful in that either it is a "
    "smaller version of the seed problem that can be directly used as a "
    "stepping stone, or it is a related problem that plausibly requires "
    "similar skills to solve.\n\n"
    "<problem>\n{conjecture}\n</problem>\n\n"
    "Please rate the relevance of the problem to the seed problem on a scale "
    "of 0 to 5, where 0 is \"not at all related\" and 5 is \"very useful for "
    "solving the seed problem.\" If the problem is trivial, you should give "
    "it a low score.\n\n"
    "Here is a rubric for how to score the problem:\n"
    "- 0: The problem is not at all related to the seed problem and is "
    "trivial to solve (e.g. an integer has one digit, is 0, 1 or a power "
    "of 10, or one integer divides the other). OR the problem is identical to "
    "the seed problem (the same two integers, possibly swapped).\n"
    "- 1: The problem is not at all related to the seed problem (e.g. it is "
    "not a problem of computing the LCM of two integers).\n"
    "- 2: The problem is related to the seed problem in that it is an "
    "LCM problem, but it is not directly useful for solving the "
    "seed problem (e.g. it is not simpler than the seed problem, or its "
    "integers look nothing like the seed's).\n"
    "- 3: The problem is related to the seed problem and may be useful for "
    "solving the seed problem.\n"
    "- 4: The problem is directly useful for solving the seed problem. That "
    "is, if the problem was solved, the seed problem would be easier to solve "
    "(e.g. integers with fewer digits but a similar common-factor "
    "structure).\n"
    "- 5: The problem is very useful for solving the seed problem, and "
    "solving it will dramatically reduce the difficulty of the seed "
    "problem.\n\n"
    "Next decide how redundant the problem statement is. Rate this as 0 or 1:\n"
    "- 0: There is nothing redundant in the problem statement.\n"
    "- 1: There are redundant parts. That is, instructions, constraints or "
    "text that are not needed to state the LCM task.\n\n"
    "Next decide if the problem statement is overly complex. Rate this on a "
    "score of 0 to 4:\n"
    "- 0: The problem is minimally complex. A single instruction to compute "
    "the LCM of two integers that is maximally clear.\n"
    "- 1: The problem has low complexity. Multiple related parts, but they "
    "form a cohesive LCM task.\n"
    "- 2: The problem has moderate complexity. Contains a few extra "
    "conditions or alternatives that are all on-theme.\n"
    "- 3: The problem has high complexity. Contains multiple unrelated tasks "
    "or conditions, or nested instructions that obscure the main "
    "LCM task.\n"
    "- 4: The problem has very high complexity. Many largely unrelated tasks "
    "or deeply nested instructions that are hard to parse.\n\n"
    "End your response with the three scores in exactly this format:\n"
    "<relevance>N</relevance>\n"
    "<redundancy>N</redundancy>\n"
    "<complexity>N</complexity>"
)

LCM_SEED_STATEMENT = ("Compute the least common multiple (LCM) of the following two integers.\n"
                      "number1: {number1}\nnumber2: {number2}")


class LCM:
    """[number1, number2] per problem, like Multiplication, but standalone: its loader
    must keep operands of any size exact (see load)."""
    source = "lcm"

    @staticmethod
    def load(path):
        """[number1, number2] per row, read with the json module, NOT load_dataset:
        pyarrow parses integers above 2**63 (and casts any column with one) as float64,
        which rounds 16- and 32-digit operands (e.g. ...27155284046640608 ->
        ...25849943428956160). The `lcm` column must match the recomputed LCM."""
        problems = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                a, b = row["number1"], row["number2"]
                assert type(a) is int and type(b) is int, f"non-integer operands in {path}: {row}"
                assert row.get("lcm", math.lcm(a, b)) == math.lcm(a, b), f"bad lcm in {path}: {row}"
                problems.append([a, b])
        return problems

    @staticmethod
    def solver_messages(p):
        return user_msg(LCM_SOLVER_PROMPT.format(number1=p[0], number2=p[1]))

    @staticmethod
    def answer(p):
        # str: LCMs of large operands overflow int64 in the verl parquet.
        return str(math.lcm(p[0], p[1]))

    @staticmethod
    def check(text, gold):
        return parse_product(text) == int(gold)

    @staticmethod
    def verify(text, p):
        return parse_product(text) == math.lcm(p[0], p[1])

    @staticmethod
    def level(p):
        return len(str(p[0]))

    @staticmethod
    def conj_messages(p, lo, hi):
        """`lo`/`hi` are sorting's integer range; unused here."""
        return user_msg(LCM_CONJ_PROMPT.format(number1=p[0], number2=p[1]))

    @staticmethod
    def conjecture(text):
        """(statement, [a, b]) from the last non-empty <problem> block. The problem is
        None unless both operands parse and have at least two digits."""
        blocks = [b.strip() for b in _PROBLEM_RE.findall(text) if b.strip()]
        if not blocks:
            return None, None
        ms = [_OPERAND_RE[key].search(blocks[-1]) for key in ("number1", "number2")]
        if not all(ms):
            return None, None
        p = [int(m.group(1)) for m in ms]
        return blocks[-1], (p if min(p) >= 10 else None)

    @staticmethod
    def guide_messages(p, conj_statement):
        seed = LCM_SEED_STATEMENT.format(number1=p[0], number2=p[1])
        return user_msg(LCM_GUIDE_PROMPT.format(seed=seed, conjecture=conj_statement))


# =============================================================================
# Preorder traversal
# =============================================================================

# Solver prompt + parser: verbatim from misc/preorder_benchmark.py (copied, not
# imported: that module imports vllm/torch at the top).
PREORDER_SOLVER_PROMPT = (
    "A binary tree has unique node labels. Given its inorder and postorder "
    "traversals, compute its preorder traversal.\n\n"
    "Inorder: {inorder}\n"
    "Postorder: {postorder}\n\n"
    "Return only the preorder traversal on a single line, enclosed in \\boxed{{}}, "
    "e.g. \\boxed{{[2, 0, 1]}}."
)


def parse_preorder(text):
    """Parse the first \\boxed{[...]} after the thinking trace into a list of ints, else None."""
    text = text.rsplit("</think>", 1)[-1]
    m = _MULT_BOXED_RE.search(text)
    if not m:
        return None
    inside = m.group(1).strip().strip("[]").strip()
    if not inside:
        return []
    try:
        return [int(x) for x in inside.split(",")]
    except ValueError:
        return None


def preorder_from(inorder, postorder):
    """Preorder of the binary tree with these traversals, or None if no such tree
    exists (repeated labels, different label sets or lengths, impossible split)."""
    if len(inorder) != len(postorder) or len(set(inorder)) != len(inorder) \
            or set(inorder) != set(postorder):
        return None
    pos = {x: i for i, x in enumerate(inorder)}
    out = []

    def build(in_lo, post_lo, n):
        # subtree = inorder[in_lo:in_lo+n] = postorder[post_lo:post_lo+n] as label sets
        if n == 0:
            return True
        root = postorder[post_lo + n - 1]
        i = pos[root]
        if not in_lo <= i < in_lo + n:
            return False
        out.append(root)
        n_left = i - in_lo
        return (build(in_lo, post_lo, n_left)
                and build(i + 1, post_lo + n_left, n - n_left - 1))

    return out if build(0, 0, len(inorder)) else None


# Conjecturer: sorting's CONJ_PROMPT with the domain mapped onto preorder traversal.
PREORDER_CONJ_PROMPT = (
    "I am learning to compute the preorder traversal of a binary tree with "
    "unique node labels from its inorder and postorder traversals. This is "
    "the task I ultimately want to solve. Here is one example problem for "
    "reference:\n\n"
    "Inorder: {inorder}\n"
    "Postorder: {postorder}\n\n"
    "To make progress, I want to first practice on a simpler version: "
    "a smaller and easier preorder traversal problem. "
    "Learning to solve this simpler task will be a "
    "stepping stone toward solving the more challenging "
    "task.\n\n"
    "Generate one such simpler problem and its solution. "
    "The tree must have at least three nodes, its labels must be distinct "
    "non-negative integers, and it must NOT be an obvious canonical example "
    "(such as a tree where every node has only a left child or only a right "
    "child). The inorder and postorder traversals must both come from the "
    "same binary tree, and the solution must be that tree's preorder "
    "traversal.\n\n"
    "Return exactly the following two blocks, replacing the placeholder "
    "lists with actual integers:\n\n"
    "<problem>\n"
    "A binary tree has unique node labels. Given its inorder and postorder "
    "traversals, compute its preorder traversal.\n"
    "Inorder: [a, b, c, ...]\n"
    "Postorder: [a, b, c, ...]\n"
    "</problem>\n"
    "<solution>\n"
    "[preorder traversal]\n"
    "</solution>\n\n"
    "Write each list on a single line using comma-separated integers. "
    "Do not include reasoning, explanations, Markdown fences, LaTeX, "
    "or any text outside these two blocks."
)

# Guide: sorting's GUIDE_PROMPT (paper §E.3) with the domain mapped onto preorder traversal.
PREORDER_GUIDE_PROMPT = (
    "You are a binary tree expert. Here is a seed tree traversal problem:\n\n"
    "<seed>\n{seed}\n</seed>\n\n"
    "Here is a simpler tree traversal problem that is supposed to be useful for "
    "solving the above problem. It can be useful in that either it is a "
    "smaller version of the seed problem that can be directly used as a "
    "stepping stone, or it is a related problem that plausibly requires "
    "similar skills to solve.\n\n"
    "<problem>\n{conjecture}\n</problem>\n\n"
    "Please rate the relevance of the problem to the seed problem on a scale "
    "of 0 to 5, where 0 is \"not at all related\" and 5 is \"very useful for "
    "solving the seed problem.\" If the problem is trivial, you should give "
    "it a low score.\n\n"
    "Here is a rubric for how to score the problem:\n"
    "- 0: The problem is not at all related to the seed problem and is "
    "trivial to solve (e.g. the tree has two or fewer nodes, or every node "
    "has only a left child or only a right child). OR the problem is identical "
    "to the seed problem (the same traversals).\n"
    "- 1: The problem is not at all related to the seed problem (e.g. it is "
    "not a problem of computing a preorder traversal from inorder and "
    "postorder traversals).\n"
    "- 2: The problem is related to the seed problem in that it is a tree "
    "traversal problem, but it is not directly useful for solving the seed "
    "problem (e.g. it is not simpler than the seed problem, or its tree "
    "looks nothing like the seed's).\n"
    "- 3: The problem is related to the seed problem and may be useful for "
    "solving the seed problem.\n"
    "- 4: The problem is directly useful for solving the seed problem. That "
    "is, if the problem was solved, the seed problem would be easier to solve "
    "(e.g. a tree with fewer nodes but a similar shape and branching).\n"
    "- 5: The problem is very useful for solving the seed problem, and "
    "solving it will dramatically reduce the difficulty of the seed "
    "problem.\n\n"
    "Next decide how redundant the problem statement is. Rate this as 0 or 1:\n"
    "- 0: There is nothing redundant in the problem statement.\n"
    "- 1: There are redundant parts. That is, instructions, constraints or "
    "text that are not needed to state the traversal task.\n\n"
    "Next decide if the problem statement is overly complex. Rate this on a "
    "score of 0 to 4:\n"
    "- 0: The problem is minimally complex. A single instruction to compute "
    "one preorder traversal from two given traversals that is maximally clear.\n"
    "- 1: The problem has low complexity. Multiple related parts, but they "
    "form a cohesive traversal task.\n"
    "- 2: The problem has moderate complexity. Contains a few extra "
    "conditions or alternatives that are all on-theme.\n"
    "- 3: The problem has high complexity. Contains multiple unrelated tasks "
    "or conditions, or nested instructions that obscure the main "
    "traversal task.\n"
    "- 4: The problem has very high complexity. Many largely unrelated tasks "
    "or deeply nested instructions that are hard to parse.\n\n"
    "End your response with the three scores in exactly this format:\n"
    "<relevance>N</relevance>\n"
    "<redundancy>N</redundancy>\n"
    "<complexity>N</complexity>"
)

PREORDER_SEED_STATEMENT = (
    "A binary tree has unique node labels. Given its inorder and postorder "
    "traversals, compute its preorder traversal.\n"
    "Inorder: {inorder}\nPostorder: {postorder}"
)

_TRAVERSAL_RE = {key: re.compile(rf"{key}\s*:\s*\[([^\[\]]*)\]", re.IGNORECASE)
                 for key in ("inorder", "postorder")}


def _int_list(s):
    try:
        return [int(x) for x in s.split(",")] if s.strip() else []
    except ValueError:
        return None


class Preorder:
    source = "preorder"

    @staticmethod
    def load(path):
        """[inorder, postorder] per row. The jsonl `preorder` column is ignored
        (recomputed by answer)."""
        from datasets import load_dataset
        ds = load_dataset("json", data_files=path, split="train")
        return [[list(i), list(p)] for i, p in zip(ds["inorder"], ds["postorder"])]

    @staticmethod
    def solver_messages(p):
        return user_msg(PREORDER_SOLVER_PROMPT.format(inorder=p[0], postorder=p[1]))

    @staticmethod
    def answer(p):
        return preorder_from(p[0], p[1])

    @staticmethod
    def check(text, gold):
        return parse_preorder(text) == [int(x) for x in gold]

    @staticmethod
    def verify(text, p):
        return parse_preorder(text) == preorder_from(p[0], p[1])

    @staticmethod
    def level(p):
        return len(p[0])

    @staticmethod
    def conj_messages(p, lo, hi):
        """`lo`/`hi` are sorting's integer range; unused here."""
        return user_msg(PREORDER_CONJ_PROMPT.format(inorder=p[0], postorder=p[1]))

    @staticmethod
    def conjecture(text):
        """(statement, [inorder, postorder]) from the last non-empty <problem> block.
        The problem is None unless both lists parse, the tree has at least two
        nodes and the two traversals come from one binary tree."""
        blocks = [b.strip() for b in _PROBLEM_RE.findall(text) if b.strip()]
        if not blocks:
            return None, None
        ms = [_TRAVERSAL_RE[key].search(blocks[-1]) for key in ("inorder", "postorder")]
        if not all(ms):
            return None, None
        inorder, postorder = (_int_list(m.group(1)) for m in ms)
        if inorder is None or postorder is None:
            return None, None
        ok = len(inorder) >= 2 and preorder_from(inorder, postorder) is not None
        return blocks[-1], ([inorder, postorder] if ok else None)

    @staticmethod
    def guide_messages(p, conj_statement):
        seed = PREORDER_SEED_STATEMENT.format(inorder=p[0], postorder=p[1])
        return user_msg(PREORDER_GUIDE_PROMPT.format(seed=seed, conjecture=conj_statement))


TASKS = {"sorting": Sorting, "multiplication": Multiplication, "lcm": LCM, "preorder": Preorder}
TASK_BY_SOURCE = {t.source: t for t in TASKS.values()}
