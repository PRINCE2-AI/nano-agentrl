"""Shared test fixtures: a character-level tokenizer and a tiny random GPT-2 (no downloads)."""

from __future__ import annotations

import string

import pytest
import torch
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

SPECIAL = ["<pad>", "<eos>", "<unk>"]


def make_char_tokenizer() -> PreTrainedTokenizerFast:
    """One token per printable character, plus <pad>=0, <eos>=1, <unk>=2."""
    vocab = {tok: i for i, tok in enumerate(SPECIAL + list(dict.fromkeys(string.printable)))}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )


def make_tiny_model(vocab_size: int, seed: int = 0) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=vocab_size,
        n_positions=512,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    return GPT2LMHeadModel(config)


@pytest.fixture(scope="session")
def char_tokenizer() -> PreTrainedTokenizerFast:
    return make_char_tokenizer()


@pytest.fixture
def tiny_model(char_tokenizer) -> GPT2LMHeadModel:
    return make_tiny_model(len(char_tokenizer)).eval()
