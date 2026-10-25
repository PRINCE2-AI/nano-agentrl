"""Retrieval recall@k: question vs agent query, against the tiny in-memory corpus."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from nano_agentrl.retriever.client import TINY_CORPUS, InMemoryRetriever

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "retrieval_recall", ROOT / "scripts/retrieval_recall.py"
)
retrieval_recall = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrieval_recall)

EXAMPLES = [
    {"question": "Who wrote Hamlet?", "golds": ["William Shakespeare"]},
    {"question": "What is the capital of France?", "golds": ["Paris"]},
]


def test_parse_queries_keeps_order_and_skips_empty():
    text = "<search>hamlet author</search><information>...</information><search>  </search>"
    assert retrieval_recall.parse_queries(text) == ["hamlet author"]
    assert retrieval_recall.parse_queries("no search here") == []


def test_gold_match_is_word_bounded_and_normalized():
    assert retrieval_recall._has_gold("Paris is the capital of France.", ["paris"])
    assert not retrieval_recall._has_gold("Parisian suburbs", ["paris"])
    assert retrieval_recall._has_gold("Written by William Shakespeare.", ["The Shakespeare"])


def test_recall_at_k_counts_questions_not_docs():
    retriever = InMemoryRetriever(TINY_CORPUS)
    golds = [["William Shakespeare"], ["Paris"]]
    queries = ["who wrote hamlet", "capital of france"]
    assert retrieval_recall.recall_at_k(retriever, queries, golds, 3) == 1.0
    # A gold the corpus cannot contain is a miss, and 1 of 2 questions hit.
    golds = [["William Shakespeare"], ["Tokyo"]]
    queries = ["who wrote hamlet", "capital of japan"]
    assert retrieval_recall.recall_at_k(retriever, queries, golds, 3) == 0.5


def test_compare_separates_question_from_agent_query():
    retriever = InMemoryRetriever(TINY_CORPUS)
    records = [
        {
            "qid": 0,
            "sample": 0,
            "question": "Who wrote Hamlet?",
            "text": "<search>hamlet author</search><answer>William Shakespeare</answer>",
            "correct": True,
        },
        {
            "qid": 1,
            "sample": 0,
            "question": "What is the capital of France?",
            "text": "<answer>Paris</answer>",  # answered without searching
            "correct": True,
        },
        {"qid": 0, "sample": 1, "question": "Who wrote Hamlet?", "text": "<search>zzz</search>"},
    ]
    out = retrieval_recall.compare(retriever, EXAMPLES, records, topk=3)
    assert out["questions"] == 2
    assert out["searched"] == 1  # sample > 0 is ignored; question 2 never searched
    assert out["coverage"] == 0.5
    assert out["recall_question"] == 1.0
    assert out["recall_first_query"] == 1.0


def test_any_query_is_the_ceiling_when_a_later_search_finds_the_gold():
    retriever = InMemoryRetriever(TINY_CORPUS)
    records = [
        {
            "sample": 0,
            "question": "Who wrote Hamlet?",
            "text": "<search>mount everest</search><search>hamlet author</search>",
        }
    ]
    out = retrieval_recall.compare(retriever, EXAMPLES, records, topk=1)
    assert out["recall_first_query"] == 0.0
    assert out["recall_any_query"] == 1.0


def test_compare_handles_questions_missing_from_the_split():
    retriever = InMemoryRetriever(TINY_CORPUS)
    records = [{"qid": 9, "sample": 0, "question": "Unknown question?", "text": ""}]
    out = retrieval_recall.compare(retriever, EXAMPLES, records, topk=3)
    assert out["questions"] == 0
    assert out["coverage"] == 0.0
