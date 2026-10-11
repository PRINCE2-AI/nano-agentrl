"""LLM-as-judge labelling of failed trajectories into a fixed failure taxonomy.

Every failed trajectory (correct == False) gets exactly one label and a one-sentence rationale:

    bad_query             searched, but the queries were off-target, so the retrieved docs
                          never contained the answer
    ignored_relevant_doc  a retrieved doc contained (or clearly implied) the gold answer, but
                          the model answered something else
    hallucinated_answer   answered confidently with no support: no search, or an answer that
                          appears nowhere in the retrieved docs
    over_searching        ran out of turns, or kept repeating near-identical searches,
                          without committing to an answer
    format_error          no valid <answer>...</answer> (malformed or missing tags)
    other                 none of the above (e.g. gold label wrong, answer correct but
                          phrased differently)

Judges:
    RuleJudge    deterministic heuristics; free, used in tests and as a baseline
    ClaudeJudge  Claude (claude-opus-5-5) with a JSON-schema output, effort "low"
Both go through `label_failures`, which caches every judgement in a jsonl keyed by a hash
of (judge name, prompt version, question, golds, trajectory text), so re-running a sweep
never pays twice for the same trajectory.

CLI:
    # 1. pick 50 failures for Prince to hand-label (fills label = "" to edit)
    python -m nano_agentrl.eval.failure_taxonomy export runs/eval/x/nq_dev_500_search.jsonl \
        --n 50 --out docs/hand_labels.jsonl
    # 2. judge them (needs ANTHROPIC_API_KEY or `ant auth login`)
    python -m nano_agentrl.eval.failure_taxonomy judge docs/hand_labels.jsonl --judge claude \
        --out runs/judge/hand_labels_claude.jsonl
    # 3. agreement with the hand labels
    python -m nano_agentrl.eval.failure_taxonomy agreement runs/judge/hand_labels_claude.jsonl \
        --gold docs/hand_labels.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from nano_agentrl.eval.em import normalize_answer

LABELS = (
    "bad_query",
    "ignored_relevant_doc",
    "hallucinated_answer",
    "over_searching",
    "format_error",
    "other",
)
PROMPT_VERSION = "v1"
INFO_RE = re.compile(r"<information>(.*?)</information>", re.DOTALL)


# ---------------------------------------------------------------- features


def retrieved_text(text: str) -> str:
    return "\n".join(INFO_RE.findall(text))


def gold_in_retrieved(record: dict) -> bool:
    """Does any normalized gold answer appear in the normalized retrieved docs?"""
    docs = normalize_answer(retrieved_text(record.get("text", "")))
    return any(normalize_answer(g) and normalize_answer(g) in docs for g in record["golds"])


def repeated_queries(record: dict) -> int:
    """How many searches repeated an earlier query (after normalization)."""
    queries = re.findall(r"<search>(.*?)</search>", record.get("text", ""), re.DOTALL)
    seen, repeats = set(), 0
    for q in (normalize_answer(q) for q in queries):
        repeats += q in seen
        seen.add(q)
    return repeats


# ---------------------------------------------------------------- judges


class Judge(Protocol):
    name: str

    def judge(self, record: dict) -> dict[str, str]: ...


class RuleJudge:
    """Heuristic labels from the trajectory alone. Order matters: first match wins."""

    name = "rule"

    def __init__(self, max_turns: int = 4) -> None:
        self.max_turns = max_turns

    def judge(self, record: dict) -> dict[str, str]:
        if record.get("prediction") is None:
            if record.get("num_searches", 0) >= self.max_turns or repeated_queries(record) >= 2:
                return {"label": "over_searching", "rationale": "search budget used, no answer"}
            return {"label": "format_error", "rationale": "no <answer> block"}
        if repeated_queries(record) >= 2:
            return {"label": "over_searching", "rationale": "repeated near-identical searches"}
        if gold_in_retrieved(record):
            return {"label": "ignored_relevant_doc", "rationale": "gold appears in retrieved docs"}
        if record.get("num_searches", 0) == 0:
            return {"label": "hallucinated_answer", "rationale": "answered without searching"}
        return {"label": "bad_query", "rationale": "searched but docs never had the gold"}


JUDGE_SYSTEM = f"""You audit failed runs of a question-answering agent that can search Wikipedia.
The agent reasons in <think>, searches with <search>query</search>, receives documents in
<information>...</information>, and must end with <answer>...</answer>. Its answer was graded
WRONG by exact match against the gold answers.

Assign exactly one failure label:
- bad_query: it searched, but its queries were off-target, so the documents never contained
  the answer.
- ignored_relevant_doc: a retrieved document contained or clearly implied a gold answer, but
  the agent answered something else.
- hallucinated_answer: it answered confidently without support - it did not search, or its
  answer appears nowhere in the documents.
- over_searching: it ran out of turns or kept repeating near-identical searches without
  committing to an answer.
- format_error: there is no valid <answer>...</answer> block (missing or malformed tags).
- other: none of the above, e.g. the answer is actually correct but phrased differently, or
  the gold answer is wrong.

Labels: {", ".join(LABELS)}. Give a rationale of one short sentence that cites the evidence."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "rationale": {"type": "string"},
    },
    "required": ["label", "rationale"],
    "additionalProperties": False,
}


def judge_user_message(record: dict) -> str:
    return (
        f"Question: {record['question']}\n"
        f"Gold answers: {json.dumps(record['golds'], ensure_ascii=False)}\n"
        f"Agent's extracted answer: {json.dumps(record.get('prediction'), ensure_ascii=False)}\n"
        f"Searches made: {record.get('num_searches', 0)}; finish reason: "
        f"{record.get('finish_reason', '')}\n\n"
        f"Full trajectory:\n<trajectory>\n{record.get('text', '')}\n</trajectory>"
    )


class ClaudeJudge:
    """Claude as the judge. One request per failure; structured output guarantees a valid label.

    The system prompt is identical for every request, so it carries a cache_control marker:
    once it exceeds the model's minimum cacheable prefix, repeated calls read it from cache.
    Refusal fallbacks are enabled (server-side, "default" routing).
    """

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", client: Any = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.name = f"claude:{model}:{effort}"

    def judge(self, record: dict) -> dict[str, str]:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=[{"type": "text", "text": JUDGE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": judge_user_message(record)}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": JUDGE_SCHEMA},
            },
        )
        if response.stop_reason == "refusal":
            return {"label": "other", "rationale": "judge refused to label this trajectory"}
        text = next(b.text for b in response.content if b.type == "text")
        out = json.loads(text)
        return {"label": out["label"], "rationale": out["rationale"]}


def make_judge(name: str, **kwargs: Any) -> Judge:
    if name == "rule":
        return RuleJudge(**{k: v for k, v in kwargs.items() if k == "max_turns"})
    if name == "claude":
        return ClaudeJudge(**{k: v for k, v in kwargs.items() if k in ("model", "effort")})
    raise ValueError(f"judge must be 'rule' or 'claude', got {name!r}")


# ---------------------------------------------------------------- labelling with a cache


def cache_key(judge_name: str, record: dict) -> str:
    payload = json.dumps(
        [judge_name, PROMPT_VERSION, record["question"], record["golds"], record.get("text", "")],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_cache(path: str | Path | None) -> dict[str, dict]:
    if path is None or not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["key"]: r for r in rows}


def label_failures(
    records: list[dict], judge: Judge, cache_path: str | Path | None = None
) -> list[dict]:
    """Label every failed record. Returns copies with 'label', 'rationale', 'judge' added."""
    cache = load_cache(cache_path)
    out = []
    for r in records:
        if r.get("correct"):
            continue
        key = cache_key(judge.name, r)
        if key not in cache:
            verdict = judge.judge(r)
            if verdict["label"] not in LABELS:
                verdict = {"label": "other", "rationale": f"invalid label {verdict['label']!r}"}
            cache[key] = {"key": key, **verdict}
            if cache_path is not None:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
        hit = cache[key]
        out.append({**r, "label": hit["label"], "rationale": hit["rationale"], "judge": judge.name})
    return out


def label_fractions(labelled: list[dict]) -> dict[str, float]:
    """Fraction of failures per label (all LABELS present, summing to 1 if any failures)."""
    counts = Counter(r["label"] for r in labelled)
    total = max(sum(counts.values()), 1)
    return {label: counts.get(label, 0) / total for label in LABELS}


# ---------------------------------------------------------------- agreement


def agreement(pred: list[str], gold: list[str]) -> dict[str, Any]:
    """Accuracy, Cohen's kappa and a confusion matrix (gold -> pred -> count)."""
    if len(pred) != len(gold) or not gold:
        raise ValueError("need two non-empty label lists of equal length")
    n = len(gold)
    acc = sum(p == g for p, g in zip(pred, gold, strict=True)) / n
    pc, gc = Counter(pred), Counter(gold)
    expected = sum(pc[label] * gc[label] for label in LABELS) / (n * n)
    kappa = (acc - expected) / (1 - expected) if expected < 1 else 1.0
    confusion = {
        g: dict(Counter(p for p, gg in zip(pred, gold, strict=True) if gg == g)) for g in gc
    }
    return {"n": n, "accuracy": acc, "cohen_kappa": kappa, "confusion": confusion}


def example_id(r: dict) -> str:
    return f"{r['qid']}#{r.get('sample', 0)}"


# ---------------------------------------------------------------- CLI


def _read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Failure taxonomy: export, judge, agreement")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("export", help="sample failures into a hand-labelling file")
    ex.add_argument("records")
    ex.add_argument("--n", type=int, default=50)
    ex.add_argument("--seed", type=int, default=0)
    ex.add_argument("--out", required=True)

    jd = sub.add_parser("judge", help="label the failures in a records file")
    jd.add_argument("records")
    jd.add_argument("--judge", choices=["rule", "claude"], default="rule")
    jd.add_argument("--model", default="claude-opus-5-5")
    jd.add_argument("--effort", default="low")
    jd.add_argument("--cache", default="runs/judge/cache.jsonl")
    jd.add_argument("--out", required=True)

    ag = sub.add_parser("agreement", help="judge labels vs hand labels")
    ag.add_argument("pred")
    ag.add_argument("--gold", required=True)
    args = parser.parse_args(argv)

    if args.cmd == "export":
        failures = [r for r in _read_jsonl(args.records) if not r.get("correct")]
        picked = random.Random(args.seed).sample(failures, min(args.n, len(failures)))
        _write_jsonl([{**r, "label": "", "rationale": ""} for r in picked], args.out)
        print(f"wrote {len(picked)} failures to {args.out}; fill in 'label' with one of {LABELS}")
    elif args.cmd == "judge":
        judge = make_judge(args.judge, model=args.model, effort=args.effort)
        records = [
            {k: v for k, v in r.items() if k not in ("label", "rationale")}
            for r in _read_jsonl(args.records)
        ]
        labelled = label_failures(records, judge, args.cache)
        _write_jsonl(labelled, args.out)
        print(json.dumps(label_fractions(labelled), indent=2))
    else:
        gold = {example_id(r): r["label"] for r in _read_jsonl(args.gold) if r.get("label")}
        pred = {example_id(r): r["label"] for r in _read_jsonl(args.pred)}
        ids = sorted(set(gold) & set(pred))
        print(json.dumps(agreement([pred[i] for i in ids], [gold[i] for i in ids]), indent=2))


if __name__ == "__main__":
    main()
