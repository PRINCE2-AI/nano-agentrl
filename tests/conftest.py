"""Shared test fixtures: a character-level tokenizer and a tiny random GPT-2 (no downloads)."""

from __future__ import annotations

import pytest
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from nano_agentrl.tiny import make_char_tokenizer, make_tiny_model


@pytest.fixture(scope="session")
def char_tokenizer() -> PreTrainedTokenizerFast:
    return make_char_tokenizer()


@pytest.fixture
def tiny_model(char_tokenizer) -> GPT2LMHeadModel:
    return make_tiny_model(len(char_tokenizer)).eval()
