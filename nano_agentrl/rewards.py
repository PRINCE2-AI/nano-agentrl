"""Outcome rewards: exact match on <answer> (Search-R1 Sec. 3.3), GSM8K numeric correctness,
and an optional small format reward (off by default).

Reward rule (both tasks), mirroring Search-R1's compute_score_em:
    correct answer                          -> 1.0
    wrong answer, but well-formatted output -> format_reward_weight  (only if enabled)
    otherwise                               -> 0.0
"""

from __future__ import annotations

import re
from fractions import Fraction

from nano_agentrl.eval.em import exact_match

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
TAGS = ("think", "search", "information", "answer")


# ---------------------------------------------------------------- parsing


def extract_answer(text: str) -> str | None:
    """Content of the LAST <answer>...</answer> block, stripped, or None.

    We pass only the response (not the prompt, whose example "<answer> Beijing </answer>"
    is why Search-R1 skips the first match), so the last match is the model's answer.
    """
    matches = ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else None


def parse_number(text: str) -> float | None:
    """'$1,234.50' -> 1234.5, '3/4' -> 0.75, '-7.' -> -7.0, 'abc' -> None."""
    cleaned = text.strip().replace(",", "").replace("$", "").rstrip(".").rstrip("%")
    try:
        return float(Fraction(cleaned))
    except (ValueError, ZeroDivisionError):
        return None


def extract_gsm8k_gold(answer_field: str) -> str:
    """GSM8K's 'answer' field ends with '#### 72'. Returns '72'."""
    return answer_field.split("####")[-1].strip().replace(",", "")


def extract_numeric_answer(text: str) -> float | None:
    """Final number of a GSM8K completion.

    Order: inside the last <answer> block; else after the last '####'; else the last
    number anywhere in the text.
    """
    region = extract_answer(text)
    if region is None and "####" in text:
        region = text.split("####")[-1]
    if region is None:
        region = text
    numbers = NUMBER_RE.findall(region)
    return parse_number(numbers[-1]) if numbers else None


# ---------------------------------------------------------------- format


def has_valid_format(text: str, task: str) -> bool:
    """Loose structural check of a response.

    Both tasks: exactly one <answer>...</answer>, and every tag in TAGS opens as often as it
    closes. Search task: the response must also end with </answer>.
    """
    if len(ANSWER_RE.findall(text)) != 1:
        return False
    for tag in TAGS:
        if text.count(f"<{tag}>") != text.count(f"</{tag}>"):
            return False
    if task == "search" and not text.rstrip().endswith("</answer>"):
        return False
    return True


# ---------------------------------------------------------------- rewards


def gsm8k_correct(text: str, gold: str) -> bool:
    pred = extract_numeric_answer(text)
    target = parse_number(gold)
    return pred is not None and target is not None and abs(pred - target) < 1e-6


def em_correct(text: str, golds: str | list[str]) -> bool:
    return exact_match(extract_answer(text), golds) == 1.0


def compute_reward(
    text: str,
    golds: str | list[str],
    task: str,
    use_format_reward: bool = False,
    format_reward_weight: float = 0.1,
) -> float:
    """Outcome reward for one response. task: "gsm8k" (golds = '72') or "search" (EM)."""
    if task == "gsm8k":
        gold = golds if isinstance(golds, str) else golds[0]
        correct = gsm8k_correct(text, gold)
    elif task == "search":
        correct = em_correct(text, golds)
    else:
        raise ValueError(f"unknown task {task!r}")
    if correct:
        return 1.0
    if use_format_reward and has_valid_format(text, task):
        return format_reward_weight
    return 0.0
