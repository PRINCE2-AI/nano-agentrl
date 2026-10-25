"""Measure retrieval recall@3: does BM25 find the gold passage — for the question, and for
the query the agent writes?

The reliability analysis says ~70% of failures never retrieve the gold passage, and pass@8
(21-28%) roughly equals one-shot RAG's gold-in-answer. Neither tells you whether the policy
writes worse queries than the raw question, or whether the retriever is the ceiling.

    # question baseline only (needs the retriever server + dataset, no GPU)
    python scripts/retrieval_recall.py --split nq_dev_500 --max_examples 100

    # question vs agent, using the records a reliability sweep already saved
    python scripts/retrieval_recall.py --split nq_dev_500 \
        --records runs/search_0.5b_t4/reliability/90/nq_dev_500.jsonl

A question counts as retrieved when a gold answer appears (SQuAD-normalized, word-bounded)
in the title or text of one of the top-k documents. "first query" is the policy's actual
decision; "any query" is the ceiling if the policy kept searching.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_agentrl.eval.em import normalize_answer  # noqa: E402
from nano_agentrl.retriever.client import RetrieverClient  # noqa: E402

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.S)


def parse_queries(text: str) -> list[str]:
    """Every non-empty `<search>` query, in order."""
    return [q.strip() for q in SEARCH_RE.findall(text or "") if q.strip()]


def _has_gold(text: str, golds: list[str]) -> bool:
    norm = normalize_answer(text)
    for gold in map(normalize_answer, golds):
        if gold and re.search(rf"(?<!\w){re.escape(gold)}(?!\w)", norm):
            return True
    return False


def retrieved(docs: list[dict], golds: list[str]) -> bool:
    return any(_has_gold(f"{d.get('title', '')} {d.get('text', '')}", golds) for d in docs)


def recall_at_k(retriever, queries: list[str], golds_list: list[list[str]], topk: int) -> float:
    """Fraction of questions whose gold appears in the top-k docs of `queries[i]`."""
    if not queries:
        return 0.0
    docs_list = retriever.search(queries, topk)
    hits = [retrieved(docs, golds) for docs, golds in zip(docs_list, golds_list, strict=True)]
    return sum(hits) / len(hits)


def golds_by_question(examples: list[dict]) -> dict[str, list[str]]:
    return {normalize_answer(e["question"]): e["golds"] for e in examples}


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compare(retriever, examples: list[dict], records: list[dict], topk: int) -> dict:
    """Question recall@k vs the agent's first and any query, over the questions in `records`."""
    golds = golds_by_question(examples)
    first = [r for r in records if r.get("sample", 0) == 0]
    # A record whose question is not in the split is skipped.
    usable = [r for r in first if golds.get(normalize_answer(r["question"]))]

    questions = [r["question"] for r in usable]
    golds_list = [golds[normalize_answer(q)] for q in questions]
    queries = [parse_queries(r.get("text", "")) for r in usable]

    base = recall_at_k(retriever, questions, golds_list, topk)
    first_q = [q[0] if q else "" for q in queries]
    searched = [i for i, q in enumerate(first_q) if q]
    first_recall = (
        recall_at_k(
            retriever, [first_q[i] for i in searched], [golds_list[i] for i in searched], topk
        )
        if searched
        else 0.0
    )
    any_recall = 0.0
    if searched:
        flat = [q for i in searched for q in queries[i]]
        docs = retriever.search(flat, topk)
        hits, start = [], 0
        for i in searched:
            n = len(queries[i])
            hits.append(any(retrieved(d, golds_list[i]) for d in docs[start : start + n]))
            start += n
        any_recall = sum(hits) / len(searched)

    return {
        "questions": len(usable),
        "searched": len(searched),
        "coverage": len(searched) / max(len(usable), 1),
        "recall_question": base,
        "recall_first_query": first_recall,
        "recall_any_query": any_recall,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="retrieval recall@k: question vs agent query")
    parser.add_argument("--split", default="nq_dev_500")
    parser.add_argument(
        "--max_examples", type=int, default=0, help="0 = every question in the split"
    )
    parser.add_argument(
        "--records", nargs="*", default=[], help="eval/sweep jsonl with trajectories"
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--retriever_url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default=None, help="optional path for a json summary")
    args = parser.parse_args(argv)

    from nano_agentrl.data import load_split  # torch-free beyond this point is not worth it

    examples = load_split(args.split)
    if args.max_examples:
        examples = examples[: args.max_examples]
    retriever = RetrieverClient(args.retriever_url)

    golds_list = [e["golds"] for e in examples]
    questions = [e["question"] for e in examples]
    summary: dict = {
        "split": args.split,
        "questions": len(examples),
        "topk": args.topk,
        f"recall_question@{args.topk}": recall_at_k(retriever, questions, golds_list, args.topk),
    }

    for path in args.records:
        records = load_records(Path(path))
        key = f"{Path(path).parent.name}/{Path(path).stem}"
        summary[key] = compare(retriever, examples, records, args.topk)

    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
