"""Answer extraction and scoring for open-ended MATH problems.

Grader is a verbatim port of the lm-evaluation-harness MATH task helpers
(`lm_eval/tasks/hendrycks_math/utils.py`): `last_boxed_only_string`,
`remove_boxed`, `strip_string`, `fix_fracs`, `fix_a_slash_b`,
`remove_right_units`, `fix_sqrt`, `is_equiv`. This is the canonical grader
behind the Hendrycks MATH baseline the SOAR paper benchmarks against.
Pure string normalization + exact match — no sympy, no timeouts.

Reward scheme follows SOAR paper §B.3:
    120.0 if boxed(y) ∧ is_equiv(boxed, gold)
    20.0  if boxed(y) ∧ ¬is_equiv ∧ gold-as-substring-of-boxed
    10.0  if boxed(y) ∧ ¬is_equiv ∧ gold-not-in-boxed
    0.0   otherwise
"""

from typing import Optional


# ---- lm-eval-harness MATH helpers (verbatim) ---------------------------------

def _last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def _remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def _remove_right_units(string: str) -> str:
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# ---- Public API --------------------------------------------------------------

def extract_boxed(text: str) -> Optional[str]:
    """Return the inside of the last `\\boxed{...}` (or `\\boxed ANSWER` / `\\fbox{...}`)
    in `text`, or None if no boxed marker is present / parseable.

    Thin wrapper over the harness's `last_boxed_only_string` + `remove_boxed`;
    swallows the harness's internal asserts so a single malformed completion
    doesn't crash a batch.
    """
    if not text:
        return None
    boxed_span = _last_boxed_only_string(text)
    if boxed_span is None:
        return None
    try:
        return _remove_boxed(boxed_span).strip()
    except AssertionError:
        return None


def is_equiv(str1: Optional[str], str2: Optional[str], verbose: bool = False) -> bool:
    """lm-eval-harness MATH equivalence: normalize both sides with `_strip_string`
    and exact-match. Deterministic, no sympy, no timeouts."""
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def success_math_answer(generated: str, gold: str) -> bool:
    """Binary 0/1 success: is there a boxed answer that is_equiv against gold?

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
    # extract_boxed — brace form
    assert extract_boxed(r"foo \boxed{42} bar") == "42"
    assert extract_boxed(r"a \boxed{\frac{1}{2}} b") == r"\frac{1}{2}"
    assert extract_boxed("no boxed here") is None
    assert extract_boxed(r"\boxed{a_{1}+b}") == "a_{1}+b"
    # Last-occurrence wins
    assert extract_boxed(r"\boxed{1} and later \boxed{2}") == "2"
    # Space form (harness-only): `\boxed ANSWER` with no braces, terminated by `$`
    assert extract_boxed(r"The answer is $\boxed 42$ done.") == "42"
    # \fbox is a parse-only fallback in the harness (_last_boxed_only_string finds
    # it, but _remove_boxed can only unwrap \boxed) — we return None rather than
    # crash on the harness's assert.
    assert extract_boxed(r"ans \fbox{7}") is None

    # is_equiv — exact / mismatch
    assert is_equiv("42", "42")
    assert not is_equiv("42", "43")
    # Harness-specific normalizations
    assert is_equiv(r"\dfrac{1}{2}", r"\frac{1}{2}")           # dfrac -> frac
    assert is_equiv(r"\tfrac{1}{2}", r"\frac{1}{2}")           # tfrac -> frac
    assert is_equiv(r"\left(\frac{1}{2}\right)", r"(\frac{1}{2})")  # \left \right stripped
    assert is_equiv("0.5", r"\frac{1}{2}")                      # manual 0.5 rewrite
    assert is_equiv("1/2", r"\frac{1}{2}")                      # fix_a_slash_b
    assert is_equiv(r"5\text{ cm}", "5")                        # right-unit stripping
    assert is_equiv(r"\sqrt3", r"\sqrt{3}")                     # fix_sqrt
    assert is_equiv(r"90^\circ", "90")                          # degree stripping

    # success_math_answer
    assert success_math_answer(r"The answer is \boxed{42}", "42")
    assert not success_math_answer("no boxed", "42")
    assert success_math_answer(r"\boxed{\dfrac{1}{2}}", r"\frac{1}{2}")

    # Reward tiers (§B.3)
    assert score_math_answer(r"\boxed{42}", "42") == 120.0
    assert score_math_answer(r"\boxed{foo42bar}", "42") == 20.0
    assert score_math_answer(r"\boxed{foo}", "42") == 10.0
    assert score_math_answer("no boxed", "42") == 0.0
    print("verify.py: all sanity checks passed")
