"""Tests for rollout.run_search_rollouts with a scripted fake generator and fake retriever."""

from __future__ import annotations

import re

import pytest

from nano_agentrl.generation import GenerationOutput, truncate_at_stop
from nano_agentrl.prompts import INVALID_ACTION_MESSAGE
from nano_agentrl.retriever.client import TINY_CORPUS, InMemoryRetriever
from nano_agentrl.rollout import parse_action, run_search_rollouts
from nano_agentrl.trainer import collate


class ScriptedGenerator:
    """Fake generator. `script(prompt_text, response_so_far, turn)` returns the next turn's text.
    Mimics the real one: tokenizes, cuts at the first stop string, reports finish_reason."""

    def __init__(self, tokenizer, script, finish_reason="length"):
        self.tok = tokenizer
        self.script = script
        self.finish_reason = finish_reason
        self.calls: list[int] = []  # batch size of every generate() call

    def generate(self, prompt_ids, n=1, max_new_tokens=500, temperature=1.0, stop_strings=None):
        self.calls.append(len(prompt_ids))
        outs = []
        for i, ids in enumerate(prompt_ids):
            text = self.tok.decode(ids)
            prompt, _, response = text.partition("|")
            reply = self.script(prompt, response, response.count("<information>"))
            out_ids = self.tok(reply, add_special_tokens=False)["input_ids"][:max_new_tokens]
            cut, hit = truncate_at_stop(out_ids, self.tok, stop_strings or [])
            finish = "stop" if hit else self.finish_reason
            outs.append(GenerationOutput(i, cut, self.tok.decode(cut), finish))
        return outs


def encode(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


@pytest.fixture
def retriever():
    return InMemoryRetriever(TINY_CORPUS)


def run(tok, script, retriever, prompts=("Q1|",), n=1, **kwargs):
    gen = ScriptedGenerator(tok, script, kwargs.pop("finish_reason", "length"))
    trajs = run_search_rollouts(
        gen,
        tok,
        retriever,
        [encode(tok, p) for p in prompts],
        [["x"]] * len(prompts),
        n=n,
        **kwargs,
    )
    return trajs, gen


def masked_text(tok, t, value):
    return "".join(
        tok.decode([i]) for i, m in zip(t.response_ids, t.response_mask, strict=True) if m == value
    )


# ---------------------------------------------------------------- parsing


def test_parse_action():
    assert parse_action("<think>x</think><search> capital of france </search>") == (
        "search",
        "capital of france",
    )
    assert parse_action("<answer>Paris</answer>") == ("answer", "Paris")
    assert parse_action("<search></search>") == (None, "")  # empty query is invalid
    assert parse_action("<search>unclosed") == (None, "")
    assert parse_action("<search>a</answer>") == (None, "")  # mismatched tags


# ---------------------------------------------------------------- masks


def test_mask_is_zero_over_every_information_span(char_tokenizer, retriever):
    def script(prompt, response, n_info):
        if n_info < 2:
            return f"<think>t{n_info}</think><search>capital of france {n_info}</search>"
        return "<think>done</think><answer>Paris</answer>"

    (t,), _ = run(char_tokenizer, script, retriever, topk=2)
    assert t.finish_reason == "answer" and t.num_searches == 2
    assert len(t.response_ids) == len(t.response_mask)

    text = "".join(char_tokenizer.decode([i]) for i in t.response_ids)
    spans = [m.span() for m in re.finditer(r"\n\n<information>.*?</information>\n\n", text, re.S)]
    assert len(spans) == 2
    inside = {i for a, b in spans for i in range(a, b)}
    for pos, m in enumerate(t.response_mask):
        assert m == (0 if pos in inside else 1), f"position {pos} {text[pos]!r}"
    assert "<information>" not in masked_text(char_tokenizer, t, 1)
    assert masked_text(char_tokenizer, t, 1).endswith("<answer>Paris</answer>")
    assert "Paris is the capital" in masked_text(char_tokenizer, t, 0)


def test_retrieved_docs_follow_search_r1_format(char_tokenizer, retriever):
    def script(prompt, response, n_info):
        return "<search>who wrote hamlet</search>" if n_info == 0 else "<answer>x</answer>"

    (t,), _ = run(char_tokenizer, script, retriever, topk=1)
    env = masked_text(char_tokenizer, t, 0)
    assert env == (
        "\n\n<information>Doc 1(Title: Hamlet) Hamlet is a tragedy written by "
        "William Shakespeare.</information>\n\n"
    )
    assert t.queries == ["who wrote hamlet"]


# ---------------------------------------------------------------- turns


def test_max_turns_cutoff(char_tokenizer, retriever):
    def always_search(prompt, response, n_info):
        return f"<search>q{n_info}</search>"

    (t,), gen = run(char_tokenizer, always_search, retriever, max_turns=3)
    assert gen.calls == [1, 1, 1, 1]  # 3 search turns + 1 final turn
    assert t.num_searches == 4 and t.text.count("<information>") == 3  # last one not executed
    assert t.finish_reason == "max_turns"


def test_malformed_tags_are_invalid_and_retried(char_tokenizer, retriever):
    def script(prompt, response, n_info):
        return "<answer>Paris</answer>" if INVALID_ACTION_MESSAGE in response else "<search>oops"

    (t,), gen = run(char_tokenizer, script, retriever)
    assert t.num_invalid == 1 and t.num_searches == 0
    assert t.finish_reason == "answer"
    assert INVALID_ACTION_MESSAGE in masked_text(char_tokenizer, t, 0)  # env text, mask 0
    assert len(gen.calls) == 2


def test_eos_without_answer_ends_trajectory(char_tokenizer, retriever):
    (t,), gen = run(char_tokenizer, lambda *a: "I give up", retriever, finish_reason="eos")
    assert t.finish_reason == "eos" and t.num_invalid == 1 and gen.calls == [1]


def test_batch_with_mixed_finishing_turns(char_tokenizer, retriever):
    """Prompt A answers at once; prompt B searches twice then answers."""

    def script(prompt, response, n_info):
        if prompt == "A":
            return "<answer>a</answer>"
        return f"<search>berlin {n_info}</search>" if n_info < 2 else "<answer>b</answer>"

    trajs, gen = run(char_tokenizer, script, retriever, prompts=("A|", "B|"), n=2)
    assert [t.prompt_index for t in trajs] == [0, 0, 1, 1]
    assert gen.calls == [4, 2, 2]  # A drops out after turn 1
    assert [t.num_searches for t in trajs] == [0, 0, 2, 2]
    assert [t.num_turns for t in trajs] == [1, 1, 3, 3]
    assert all(t.finish_reason == "answer" for t in trajs)
    assert retriever.calls == [["berlin 0", "berlin 0"], ["berlin 1", "berlin 1"]]  # batched

    batch = collate(trajs, char_tokenizer.pad_token_id)
    for row, t in enumerate(trajs):
        p, r = len(t.prompt_ids), len(t.response_ids)
        assert batch["loss_mask"][row, :p].sum() == 0
        assert batch["loss_mask"][row, p : p + r].tolist() == t.response_mask
        assert batch["attention_mask"][row].sum() == p + r


def test_long_observation_is_truncated_but_closed(char_tokenizer, retriever):
    def script(prompt, response, n_info):
        return "<search>paris</search>" if n_info == 0 else "<answer>x</answer>"

    (t,), _ = run(char_tokenizer, script, retriever, topk=6, max_obs_tokens=60)
    env = masked_text(char_tokenizer, t, 0)
    assert len(env) == 60 and env.endswith("</information>\n\n")


def test_total_length_limit(char_tokenizer, retriever):
    (t,), gen = run(
        char_tokenizer, lambda *a: "<search>paris</search>", retriever, max_total_tokens=100
    )
    assert t.finish_reason == "length" and gen.calls == [1]
