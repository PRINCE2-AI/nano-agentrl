"""pass@k / pass^k estimators, failure taxonomy (rule + mocked Claude judge), reliability sweep."""

from __future__ import annotations

import csv
import importlib.util
import itertools
import json
from math import comb
from pathlib import Path
from types import SimpleNamespace

import pytest

from nano_agentrl.eval.failure_taxonomy import (
    LABELS,
    ClaudeJudge,
    RuleJudge,
    agreement,
    label_failures,
    label_fractions,
)
from nano_agentrl.eval.failure_taxonomy import main as taxonomy_main
from nano_agentrl.eval.pass_k import pass_at_k, pass_hat_k, pass_metrics

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- pass@k / pass^k


def test_numeric_example_from_docstring():
    assert pass_at_k(4, 2, 2) == pytest.approx(1 - 1 / 6)
    assert pass_hat_k(4, 2, 2) == pytest.approx(1 / 6)


def test_k1_equals_accuracy_and_edge_cases():
    assert pass_at_k(8, 3, 1) == pytest.approx(3 / 8) == pass_hat_k(8, 3, 1)
    assert pass_at_k(8, 0, 4) == 0.0 and pass_hat_k(8, 8, 8) == 1.0
    assert pass_at_k(8, 6, 3) == 1.0  # fewer than k wrong samples: always one success
    with pytest.raises(ValueError):
        pass_at_k(4, 5, 1)


@pytest.mark.parametrize("n, c, k", [(5, 2, 2), (6, 4, 3), (4, 1, 4)])
def test_estimators_are_unbiased(n, c, k):
    """Average over every k-subset of the n samples equals the closed form."""
    samples = [1] * c + [0] * (n - c)
    subsets = list(itertools.combinations(samples, k))
    assert len(subsets) == comb(n, k)
    assert pass_at_k(n, c, k) == pytest.approx(sum(any(s) for s in subsets) / len(subsets))
    assert pass_hat_k(n, c, k) == pytest.approx(sum(all(s) for s in subsets) / len(subsets))


def test_pass_metrics_over_questions():
    records = [{"qid": "a", "correct": c} for c in (1, 1, 1, 1)] + [
        {"qid": "b", "correct": c} for c in (1, 0, 0, 0)
    ]
    m = pass_metrics(records)
    assert m["pass@1"] == pytest.approx((1 + 0.25) / 2)
    assert m["pass^4"] == pytest.approx(0.5)  # only question a is always right
    assert m["pass@4"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        pass_metrics(records + [{"qid": "c", "correct": 1}])


# ---------------------------------------------------------------- taxonomy


def rec(qid, text, prediction, searches, golds=("Paris",), correct=False):
    return {
        "qid": qid,
        "sample": 0,
        "question": "q?",
        "golds": list(golds),
        "text": text,
        "prediction": prediction,
        "num_searches": searches,
        "correct": correct,
        "finish_reason": "answer",
    }


INFO = "\n\n<information>Doc 1(Title: Paris) Paris is the capital of France.</information>\n\n"
CASES = {
    "format_error": rec(1, "I think it is Paris", None, 0),
    "over_searching": rec(2, "<search>x</search>" * 4, None, 4),
    "ignored_relevant_doc": rec(
        3, f"<search>france</search>{INFO}<answer>Lyon</answer>", "Lyon", 1
    ),
    "hallucinated_answer": rec(4, "<answer>Lyon</answer>", "Lyon", 0),
    "bad_query": rec(
        5,
        "<search>cats</search>\n\n<information>Cats</information>\n\n<answer>Lyon</answer>",
        "Lyon",
        1,
    ),
}


@pytest.mark.parametrize("label", list(CASES))
def test_rule_judge(label):
    assert RuleJudge(max_turns=4).judge(CASES[label])["label"] == label


class FakeClaude:
    """Mimics client.beta.messages.create with a JSON-schema text block."""

    def __init__(self, label="bad_query", stop_reason="end_turn"):
        self.calls = []
        self.label, self.stop_reason = label, stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = json.dumps({"label": self.label, "rationale": "because"})
        return SimpleNamespace(
            stop_reason=self.stop_reason, content=[SimpleNamespace(type="text", text=text)]
        )


def test_claude_judge_request_shape_and_cache(tmp_path):
    fake = FakeClaude()
    judge = ClaudeJudge(client=fake)
    records = [
        CASES["bad_query"],
        CASES["hallucinated_answer"],
        {**CASES["bad_query"], "qid": 9, "correct": True},
    ]
    cache = tmp_path / "cache.jsonl"
    out = label_failures(records, judge, cache)
    assert [r["label"] for r in out] == ["bad_query", "bad_query"]  # correct record skipped
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"]["format"]["schema"]["properties"]["label"]["enum"] == list(LABELS)
    assert call["output_config"]["effort"] == "low"
    assert call["fallbacks"] == "default"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "Gold answers" in call["messages"][0]["content"]

    label_failures(records, ClaudeJudge(client=fake), cache)  # second run: all cached
    assert len(fake.calls) == 2


def test_claude_refusal_maps_to_other():
    judge = ClaudeJudge(client=FakeClaude(stop_reason="refusal"))
    assert judge.judge(CASES["bad_query"])["label"] == "other"


def test_label_fractions_and_agreement():
    labelled = [{"label": "bad_query"}, {"label": "bad_query"}, {"label": "format_error"}]
    fr = label_fractions(labelled)
    assert fr["bad_query"] == pytest.approx(2 / 3) and sum(fr.values()) == pytest.approx(1)
    a = agreement(
        ["bad_query", "format_error", "other"], ["bad_query", "format_error", "bad_query"]
    )
    assert a["accuracy"] == pytest.approx(2 / 3) and -1 <= a["cohen_kappa"] <= 1
    assert agreement(["x", "y"], ["x", "y"])["cohen_kappa"] == pytest.approx(1.0)


def test_taxonomy_cli_export_judge_agreement(tmp_path, capsys):
    records = tmp_path / "r.jsonl"
    records.write_text("\n".join(json.dumps(r) for r in CASES.values()), encoding="utf-8")
    hand = tmp_path / "hand.jsonl"
    taxonomy_main(["export", str(records), "--n", "5", "--out", str(hand)])
    rows = [json.loads(line) for line in hand.read_text().splitlines()]
    for r in rows:  # "Prince" labels them with the true category
        r["label"] = next(k for k, v in CASES.items() if v["qid"] == r["qid"])
    hand.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    pred = tmp_path / "pred.jsonl"
    taxonomy_main(
        [
            "judge",
            str(hand),
            "--judge",
            "rule",
            "--cache",
            str(tmp_path / "c.jsonl"),
            "--out",
            str(pred),
        ]
    )
    capsys.readouterr()
    taxonomy_main(["agreement", str(pred), "--gold", str(hand)])
    report = json.loads(capsys.readouterr().out)
    assert report["n"] == 5 and report["accuracy"] == 1.0


# ---------------------------------------------------------------- sweep


def test_reliability_sweep_dry_run(tmp_path):
    run_dir = tmp_path / "run"
    train = load_script("train")
    train.main(
        [
            "--config",
            str(ROOT / "configs/search_0.5b.yaml"),
            "--dry_run",
            "--run_dir",
            str(run_dir),
            "--set",
            "save_every=1",
            "--set",
            "eval_every=0",
        ]
    )
    sweep = load_script("reliability_sweep")
    out = sweep.main(["--run_dir", str(run_dir), "--dry_run", "--max_examples", "3"])
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [int(r["step"]) for r in rows] == [1, 2]
    for key in ("pass_at_1", "pass_hat_4", "num_searches", "fail_format_error"):
        assert key in rows[0]
    # resume: records exist, nothing is regenerated, same CSV
    assert (
        list(
            csv.DictReader(
                sweep.main(["--run_dir", str(run_dir), "--dry_run", "--max_examples", "3"]).open()
            )
        )
        == rows
    )


def test_reliability_sweep_readonly_ckpts_and_config(tmp_path):
    """Checkpoints in one dir (e.g. a read-only Kaggle input), outputs in another,
    rollout limits taken from the training config."""
    run_dir = tmp_path / "run"
    train = load_script("train")
    train.main(
        [
            "--config",
            str(ROOT / "configs/search_0.5b.yaml"),
            "--dry_run",
            "--run_dir",
            str(run_dir),
            "--set",
            "eval_every=0",
        ]
    )
    out_dir = tmp_path / "out"
    sweep = load_script("reliability_sweep")
    out = sweep.main(
        [
            "--run_dir",
            str(tmp_path / "unused"),
            "--ckpt_dir",
            str(run_dir / "checkpoints"),
            "--out_dir",
            str(out_dir),
            "--config",
            str(ROOT / "configs/search_0.5b_t4.yaml"),
            "--dry_run",
            "--max_examples",
            "2",
        ]
    )
    assert out == out_dir / "reliability.csv" and out.exists()
    assert (out_dir / "reliability" / "step_000002" / "tinyqa_test.jsonl").exists()
    assert not (run_dir / "reliability").exists()  # nothing written next to the checkpoints
