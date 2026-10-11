"""Per-token log-probabilities of response tokens under a causal LM, with masks.

Implements log pi_theta(o_t | q, o_<t) used in the GRPO objective (DeepSeekMath Sec. 4.1,
Eq. 3) and in the k3 KL term (Eq. 4). The same function is called three times per batch:
on the policy (with grad), on the rollout snapshot ("old" log-probs, no grad) and on the
frozen reference model (no grad). See docs/grpo_math.md, "Token log-probs".
"""

from __future__ import annotations

import torch


def position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Positions 0, 1, 2, ... counted over real tokens only, so left padding is skipped.

    attention_mask: (B, T) with 1 = real token, 0 = padding.
    Returns: (B, T) int64. Padded positions get 0 (their value never matters).

    Example, one left-padded row:
        attention_mask = [0, 0, 1, 1, 1]
        cumsum - 1     = [-1, -1, 0, 1, 2]
        clamp(min=0)   = [0, 0, 0, 1, 2]
    """
    return (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)


def get_token_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Log-probability the model assigns to every token of `input_ids`, zeroed outside the response.

    Args:
        model: a Hugging Face causal LM (e.g. `GPT2LMHeadModel`, Qwen2). Calling
            `model(input_ids=..., attention_mask=..., position_ids=...)` returns an object
            whose `.logits` has shape (B, T, V).
        input_ids: (B, T) int64. Prompt + response token ids, padded to a common length T,
            on the left or the right. Pad token values are arbitrary.
        attention_mask: (B, T) int64 or bool. 1 for real tokens, 0 for padding.
        response_mask: (B, T) int64 or bool. 1 for tokens the policy generated; 0 for
            prompt tokens, padding and every token inside a retrieved
            <information>...</information> block.

    Returns:
        logprobs: (B, T) float32, aligned with `input_ids`:
            logprobs[b, t] = log p_model(input_ids[b, t] | input_ids[b, :t])
                                 if response_mask[b, t] == 1 (and t >= 1)
            logprobs[b, t] = 0.0 exactly, otherwise
        Gradients flow back to `model`; the caller decides whether to use torch.no_grad().

    Steps:
        1. One forward pass over the padded batch, with attention_mask (no attending to
           padding) and position_ids counted over real tokens (left padding stays correct).
        2. Off-by-one shift: logits[:, t] predicts token t + 1, so logits[:, :-1] lines up
           with input_ids[:, 1:]. Token 0 has no prediction and gets 0.
        3. log-prob of the token that came next, in float32:
               log_softmax(x)[y] = x[y] - logsumexp(x)
           Gathering x[y] and computing logsumexp gives the same number as log_softmax +
           gather without materializing a second (B, T-1, V) tensor of log-probs. With
           V ~ 152k (Qwen) that tensor is the largest in training. logsumexp is stable
           (it subtracts the max first), unlike log(softmax).
        4. Zero every position outside response_mask, so non-policy tokens get no gradient
           (Search-R1 Sec. 3.1, "Loss Masking for Retrieved Tokens").
    """
    # 1. Forward pass. logits: (B, T, V)
    position_ids = position_ids_from_attention_mask(attention_mask)
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
    ).logits

    # 2. Shift: predictions at 0..T-2 score the tokens at 1..T-1.
    pred_logits = logits[:, :-1, :].float()  # (B, T-1, V), float32 for precision
    next_tokens = input_ids[:, 1:]  # (B, T-1)

    # 3. log p(y) = logit[y] - logsumexp(logits), without building all V log-probs.
    token_logits = pred_logits.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logits - torch.logsumexp(pred_logits, dim=-1)  # (B, T-1)

    # Put a 0 in front for token 0 so the output lines up with input_ids: (B, T)
    first = torch.zeros_like(token_logprobs[:, :1])
    logprobs = torch.cat([first, token_logprobs], dim=1)

    # 4. Exact zeros outside the response (torch.where avoids NaN * 0 = NaN surprises).
    return torch.where(response_mask.bool(), logprobs, torch.zeros_like(logprobs))
