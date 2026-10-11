"""Tests for generation.py with the HF backend, a char tokenizer and a tiny random GPT-2."""

from __future__ import annotations

import torch

from nano_agentrl.generation import Generator, StopOnStrings, truncate_at_stop
from nano_agentrl.prompts import format_information, search_r1_prompt


def test_truncate_at_stop_cuts_after_stop_string(char_tokenizer):
    ids = char_tokenizer("abc</search>xyz", add_special_tokens=False)["input_ids"]
    cut, hit = truncate_at_stop(ids, char_tokenizer, ["</search>", "</answer>"])
    assert hit
    assert char_tokenizer.decode(cut) == "abc</search>"


def test_truncate_at_stop_picks_earliest(char_tokenizer):
    ids = char_tokenizer("a</answer>b</search>", add_special_tokens=False)["input_ids"]
    cut, hit = truncate_at_stop(ids, char_tokenizer, ["</search>", "</answer>"])
    assert hit and char_tokenizer.decode(cut) == "a</answer>"


def test_truncate_at_stop_no_hit(char_tokenizer):
    ids = char_tokenizer("plain text", add_special_tokens=False)["input_ids"]
    assert truncate_at_stop(ids, char_tokenizer, ["</search>"]) == (ids, False)
    assert truncate_at_stop(ids, char_tokenizer, []) == (ids, False)


def test_stop_criterion_is_per_row(char_tokenizer):
    rows = ["PPab</search>", "PPabcdefghijk"]
    ids = torch.tensor([char_tokenizer(r, add_special_tokens=False)["input_ids"] for r in rows])
    crit = StopOnStrings(char_tokenizer, ["</search>"], prompt_len=2)
    assert crit(ids, None).tolist() == [True, False]


def test_hf_generate_layout_and_lengths(tiny_model, char_tokenizer):
    gen = Generator(tiny_model, char_tokenizer, backend="hf")
    torch.manual_seed(0)
    prompts = ["hello", "a much longer prompt"]
    outs = gen.generate_text(prompts, n=3, max_new_tokens=8, temperature=1.0)
    assert [o.prompt_index for o in outs] == [0, 0, 0, 1, 1, 1]
    for o in outs:
        assert 1 <= len(o.token_ids) <= 8
        assert o.finish_reason in ("length", "eos")
        assert char_tokenizer.pad_token_id not in o.token_ids
        assert o.text == char_tokenizer.decode(o.token_ids, skip_special_tokens=True)


def test_hf_greedy_is_deterministic(tiny_model, char_tokenizer):
    gen = Generator(tiny_model, char_tokenizer, backend="hf")
    a = gen.generate_text(["abc"], n=2, max_new_tokens=6, temperature=0)
    b = gen.generate_text(["abc"], n=2, max_new_tokens=6, temperature=0)
    assert a[0].token_ids == a[1].token_ids == b[0].token_ids


def test_hf_generate_restores_train_mode(tiny_model, char_tokenizer):
    tiny_model.train()
    Generator(tiny_model, char_tokenizer).generate_text(["x"], max_new_tokens=2)
    assert tiny_model.training


def test_search_r1_prompt_and_information_format():
    prompt = search_r1_prompt("  who wrote hamlet? ")
    assert prompt.startswith("Answer the given question. You must conduct reasoning inside <think>")
    assert prompt.endswith("Question: who wrote hamlet?\n")
    info = format_information([{"title": "Hamlet", "text": "A play."}, {"title": "B", "text": "c"}])
    assert (
        info == "\n\n<information>Doc 1(Title: Hamlet) A play.\nDoc 2(Title: B) c</information>\n\n"
    )
