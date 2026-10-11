"""Tests for nano_agentrl.logprobs.get_token_logprobs.

The ground truth is a deliberately slow reference: for every response token, run the model
on the UNPADDED prefix before it and read off log p(token | prefix). A correct batched
implementation must match it for every padding layout.
"""

from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from nano_agentrl.logprobs import get_token_logprobs

VOCAB = 100
PAD_ID = 0
ATOL = 1e-5


@pytest.fixture(scope="module")
def model() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=VOCAB,
        n_positions=64,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(config).eval()


def random_sequences(lengths: list[int], seed: int = 1) -> list[list[int]]:
    """Random token ids in [1, VOCAB) so they never collide with PAD_ID."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (n,), generator=g).tolist() for n in lengths]


def make_batch(
    seqs: list[list[int]], prompt_lens: list[int], side: str, pad_value: int = PAD_ID
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Pad `seqs` to a common length on `side` ("left" or "right").

    Returns input_ids (B, T), attention_mask (B, T), response_mask (B, T) and, for each row,
    the column offset where its first real token sits.
    """
    T = max(len(s) for s in seqs)
    B = len(seqs)
    input_ids = torch.full((B, T), pad_value, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)
    response_mask = torch.zeros((B, T), dtype=torch.long)
    offsets = []
    for b, (seq, p) in enumerate(zip(seqs, prompt_lens, strict=True)):
        off = T - len(seq) if side == "left" else 0
        offsets.append(off)
        input_ids[b, off : off + len(seq)] = torch.tensor(seq)
        attention_mask[b, off : off + len(seq)] = 1
        response_mask[b, off + p : off + len(seq)] = 1
    return input_ids, attention_mask, response_mask, offsets


@torch.no_grad()
def reference_logprobs(
    model: torch.nn.Module, seqs: list[list[int]], offsets: list[int], T: int
) -> torch.Tensor:
    """Slow ground truth, laid out like the batch: (B, T) with log p(x_t | x_<t) at every
    real position t >= 1 of each row, 0 elsewhere. One forward pass per token, no padding."""
    out = torch.zeros((len(seqs), T))
    for b, (seq, off) in enumerate(zip(seqs, offsets, strict=True)):
        for t in range(1, len(seq)):
            prefix = torch.tensor([seq[:t]])
            next_token_logits = model(input_ids=prefix).logits[0, -1].float()
            out[b, off + t] = torch.log_softmax(next_token_logits, dim=-1)[seq[t]]
    return out


# Three sequences of different lengths with different prompt lengths.
LENGTHS = [9, 6, 12]
PROMPT_LENS = [3, 2, 5]


def expected_for(model, seqs, response_mask, offsets) -> torch.Tensor:
    ref = reference_logprobs(model, seqs, offsets, response_mask.shape[1])
    return ref * response_mask


def test_shape_dtype_and_range(model):
    seqs = random_sequences(LENGTHS)
    ids, attn, resp, _ = make_batch(seqs, PROMPT_LENS, side="right")
    out = get_token_logprobs(model, ids, attn, resp)
    assert out.shape == ids.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert (out <= 0).all(), "log-probabilities can never be positive"


def test_no_padding_matches_reference(model):
    seqs = random_sequences([10, 10, 10])
    ids, attn, resp, offsets = make_batch(seqs, [4, 4, 4], side="right")
    out = get_token_logprobs(model, ids, attn, resp)
    torch.testing.assert_close(out, expected_for(model, seqs, resp, offsets), atol=ATOL, rtol=0)


def test_right_padding_matches_reference(model):
    seqs = random_sequences(LENGTHS)
    ids, attn, resp, offsets = make_batch(seqs, PROMPT_LENS, side="right")
    out = get_token_logprobs(model, ids, attn, resp)
    torch.testing.assert_close(out, expected_for(model, seqs, resp, offsets), atol=ATOL, rtol=0)


def test_left_padding_matches_reference(model):
    """Fails if position ids are not computed from attention_mask (docstring step 2)."""
    seqs = random_sequences(LENGTHS)
    ids, attn, resp, offsets = make_batch(seqs, PROMPT_LENS, side="left")
    out = get_token_logprobs(model, ids, attn, resp)
    torch.testing.assert_close(out, expected_for(model, seqs, resp, offsets), atol=ATOL, rtol=0)


@pytest.mark.parametrize("side", ["left", "right"])
def test_pad_token_values_do_not_matter(model, side):
    seqs = random_sequences(LENGTHS)
    ids_a, attn, resp, _ = make_batch(seqs, PROMPT_LENS, side=side, pad_value=0)
    ids_b, _, _, _ = make_batch(seqs, PROMPT_LENS, side=side, pad_value=77)
    out_a = get_token_logprobs(model, ids_a, attn, resp)
    out_b = get_token_logprobs(model, ids_b, attn, resp)
    torch.testing.assert_close(out_a, out_b, atol=ATOL, rtol=0)


@pytest.mark.parametrize("side", ["left", "right"])
def test_masked_positions_are_exactly_zero(model, side):
    seqs = random_sequences(LENGTHS)
    ids, attn, resp, offsets = make_batch(seqs, PROMPT_LENS, side=side)
    # Simulate a retrieved <information> block in the middle of row 2's response.
    resp[2, offsets[2] + 7 : offsets[2] + 10] = 0
    out = get_token_logprobs(model, ids, attn, resp)

    masked = resp == 0
    assert (out[masked] == 0).all(), "prompt / padding / retrieved tokens must be exactly 0.0"
    assert (out[~masked] != 0).all(), "every response token should get a real log-prob"
    torch.testing.assert_close(out, expected_for(model, seqs, resp, offsets), atol=ATOL, rtol=0)


def test_first_position_is_zero_even_if_marked(model):
    """Token 0 has nothing before it, so nothing predicts it (docstring step 3)."""
    seqs = random_sequences([8, 8])
    ids, attn, _, _ = make_batch(seqs, [0, 0], side="right")
    resp = torch.ones_like(ids)
    out = get_token_logprobs(model, ids, attn, resp)
    assert (out[:, 0] == 0).all()


def test_gradients_flow_to_model(model):
    seqs = random_sequences(LENGTHS)
    ids, attn, resp, _ = make_batch(seqs, PROMPT_LENS, side="left")
    model.zero_grad()
    out = get_token_logprobs(model, ids, attn, resp)
    assert out.requires_grad, "do not use torch.no_grad() inside get_token_logprobs"
    out.sum().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    model.zero_grad()


def test_bf16_model_returns_float32(model):
    seqs = random_sequences(LENGTHS)
    ids, attn, resp, offsets = make_batch(seqs, PROMPT_LENS, side="right")
    bf16_model = GPT2LMHeadModel(model.config).eval()
    bf16_model.load_state_dict(model.state_dict())
    bf16_model.to(torch.bfloat16)
    out = get_token_logprobs(bf16_model, ids, attn, resp)
    assert out.dtype == torch.float32
    expected = expected_for(model, seqs, resp, offsets)
    torch.testing.assert_close(out, expected, atol=0.1, rtol=0)
