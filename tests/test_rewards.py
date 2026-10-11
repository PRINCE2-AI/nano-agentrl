"""Tests for rewards.py and eval/em.py on tricky strings."""

from __future__ import annotations

import pytest

from nano_agentrl.eval.em import exact_match, normalize_answer
from nano_agentrl.rewards import (
    compute_reward,
    extract_answer,
    extract_gsm8k_gold,
    extract_numeric_answer,
    has_valid_format,
    parse_number,
)

# ---------------------------------------------------------------- EM


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The Eiffel Tower!", "eiffel tower"),
        ("  an   Apple  ", "apple"),
        ("Theatre", "theatre"),  # 'the' inside a word is kept
        ("U.S.A.", "usa"),
    ],
)
def test_normalize_answer(text, expected):
    assert normalize_answer(text) == expected


def test_exact_match_multiple_golds_and_none():
    assert exact_match("the Beatles", ["Rolling Stones", "Beatles"]) == 1.0
    assert exact_match("Beatles band", ["Beatles"]) == 0.0
    assert exact_match(None, ["x"]) == 0.0
    assert exact_match("Paris", "paris") == 1.0


# ---------------------------------------------------------------- answer extraction


def test_extract_answer_takes_last_block():
    text = "<think>maybe <answer> A </answer>?</think> <answer> B </answer>"
    assert extract_answer(text) == "B"
    assert extract_answer("no answer here") is None
    assert extract_answer("<answer>\nmulti\nline\n</answer>") == "multi\nline"


@pytest.mark.parametrize(
    "s, value",
    [("1,234", 1234.0), ("$5.50", 5.5), ("3/4", 0.75), ("-7.", -7.0), ("12%", 12.0), ("x", None)],
)
def test_parse_number(s, value):
    assert parse_number(s) == value


def test_extract_gsm8k_gold():
    assert extract_gsm8k_gold("Natalia sold 48/2 = 24 clips.\n#### 1,072") == "1072"


@pytest.mark.parametrize(
    "text, value",
    [
        ("<think>2+2=4, then 4*3=12</think><answer> 12 </answer>", 12.0),
        ("<answer>The answer is $1,200.</answer>", 1200.0),
        ("so we get 5 apples\n#### 7", 7.0),
        ("first 3 then 9 finally 10.5", 10.5),
        ("<answer>none</answer>", None),
        ("no numbers", None),
    ],
)
def test_extract_numeric_answer(text, value):
    assert extract_numeric_answer(text) == value


# ---------------------------------------------------------------- format + reward


def test_has_valid_format_search():
    good = (
        "<think>x</think><search>q</search>\n\n<information>d</information>\n\n<answer>y</answer>"
    )
    assert has_valid_format(good, "search")
    assert not has_valid_format("<think>x</think>", "search")  # no answer
    assert not has_valid_format("<think>x<answer>y</answer>", "search")  # unbalanced think
    assert not has_valid_format("<answer>a</answer><answer>b</answer>", "search")  # two answers
    assert not has_valid_format("<answer>a</answer> trailing words", "search")


def test_compute_reward_gsm8k():
    assert compute_reward("<answer>18</answer>", "18", "gsm8k") == 1.0
    assert compute_reward("<answer>18.0</answer>", "18", "gsm8k") == 1.0
    assert compute_reward("<answer>17</answer>", "18", "gsm8k") == 0.0


def test_compute_reward_search_em():
    golds = ["Barack Obama", "Obama"]
    assert compute_reward("<think>..</think><answer> obama </answer>", golds, "search") == 1.0
    assert compute_reward("<answer>Michelle Obama</answer>", golds, "search") == 0.0
    assert compute_reward("Obama", golds, "search") == 0.0  # no <answer> tag


def test_format_reward_only_when_enabled_and_wrong():
    wrong = "<think>t</think><answer>Paris</answer>"
    assert compute_reward(wrong, ["London"], "search") == 0.0
    r = compute_reward(
        wrong, ["London"], "search", use_format_reward=True, format_reward_weight=0.1
    )
    assert r == pytest.approx(0.1)
    right = "<answer>London</answer>"
    assert compute_reward(right, ["London"], "search", use_format_reward=True) == 1.0
    assert compute_reward("London", ["London"], "search", use_format_reward=True) == 0.0


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        compute_reward("x", "y", "chess")
