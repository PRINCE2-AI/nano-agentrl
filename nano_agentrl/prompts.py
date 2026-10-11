"""Instruction prompt templates and the exact environment strings of Search-R1.

The Search-R1 strings below are copied verbatim from the Search-R1 repository
(https://github.com/PeterGriffinJin/Search-R1, Apache-2.0):
  - SEARCH_R1_TEMPLATE: scripts/data_process/nq_search.py, make_prefix(template_type="base")
  - format_information / format_docs: search_r1/llm_agent/generation.py
    (the observation wrapper and _passages2string)
  - INVALID_ACTION_MESSAGE: search_r1/llm_agent/generation.py, execute_predictions
Paper: Jin et al., 2025, "Search-R1", arXiv:2503.09516, Sec. 3 and Table 1.
"""

from __future__ import annotations

from typing import Any

SEARCH_R1_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new "
    "information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: {question}\n"
)

INVALID_ACTION_MESSAGE = (
    "\nMy previous action is invalid. If I want to search, I should put the query between "
    "<search> and </search>. If I want to give the final answer, I should put the answer "
    "between <answer> and </answer>. Let me try again.\n"
)

# Our own template for the single-turn GSM8K milestone (not from Search-R1).
GSM8K_TEMPLATE = (
    "Solve the math problem. Think step by step inside <think> and </think>, then give only "
    "the final number inside <answer> and </answer>. For example, <answer> 42 </answer>. "
    "Question: {question}\n"
)


def search_r1_prompt(question: str) -> str:
    return SEARCH_R1_TEMPLATE.format(question=question.strip())


def gsm8k_prompt(question: str) -> str:
    return GSM8K_TEMPLATE.format(question=question.strip())


def format_docs(docs: list[dict[str, Any]]) -> str:
    """Search-R1 `_passages2string`: 'Doc 1(Title: t) text\\n' per retrieved doc.

    docs: list of {"title": str, "text": str} (our retriever's format).
    """
    return "".join(f"Doc {i + 1}(Title: {d['title']}) {d['text']}\n" for i, d in enumerate(docs))


def format_information(docs: list[dict[str, Any]]) -> str:
    """The observation appended after a <search> call, exactly as in Search-R1."""
    return f"\n\n<information>{format_docs(docs).strip()}</information>\n\n"


def build_chat_prompt(tokenizer: Any, user_text: str) -> str:
    """Wrap `user_text` in the tokenizer's chat template (instruct models) if it has one."""
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_text
