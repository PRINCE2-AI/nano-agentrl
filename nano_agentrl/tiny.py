"""Tiny stand-ins for a real LLM: a character-level tokenizer and a 2-layer random GPT-2.

Used by `--dry_run` and by the tests, so everything runs on CPU in seconds with no
downloads. Also a synthetic arithmetic dataset whose answers the tiny model can be scored on.
"""

from __future__ import annotations

import random
import string

import torch
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

SPECIAL_TOKENS = ["<pad>", "<eos>", "<unk>"]


def make_char_tokenizer() -> PreTrainedTokenizerFast:
    """One token per printable character, plus <pad>=0, <eos>=1, <unk>=2."""
    chars = list(dict.fromkeys(string.printable))
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS + chars)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )


def make_tiny_model(vocab_size: int, seed: int = 0) -> GPT2LMHeadModel:
    """GPT2Config(n_layer=2, n_head=2, n_embd=64), dropout off, random weights."""
    torch.manual_seed(seed)
    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=vocab_size,
        n_positions=1024,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    return GPT2LMHeadModel(config)


def make_arithmetic_dataset(n: int, seed: int = 0) -> list[dict]:
    """[{"question": "What is 3+4?", "golds": ["7"]}, ...]"""
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        data.append({"question": f"What is {a}+{b}?", "golds": [str(a + b)]})
    return data


def make_tiny_qa_dataset() -> list[dict]:
    """Questions answerable from retriever.client.TINY_CORPUS (for search-task dry runs)."""
    return [
        {"question": "What is the capital of France?", "golds": ["Paris"]},
        {"question": "Who wrote Hamlet?", "golds": ["William Shakespeare", "Shakespeare"]},
        {"question": "What is the highest mountain on Earth?", "golds": ["Mount Everest"]},
        {"question": "What is the capital of Germany?", "golds": ["Berlin"]},
        {"question": "Who wrote Pride and Prejudice?", "golds": ["Jane Austen"]},
        {"question": "Who created Python?", "golds": ["Guido", "Guido van Rossum"]},
    ]
