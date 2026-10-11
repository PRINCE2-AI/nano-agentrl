# GRPO math notes

Short explanations of the maths behind each core file, with tiny numeric examples.

## Token log-probs

**What.** For a response o = (o_1, ..., o_n) to a question q, GRPO needs, for every response
token, the number

    log π_θ(o_t | q, o_<t)

i.e. how likely the model thinks token o_t is, given everything before it. The ratio
`exp(log π_θ − log π_old)` in the clipped loss and the k3 KL term are both built from these
numbers (DeepSeekMath Sec. 4.1, Eq. 3 and 4). Implemented in `nano_agentrl/logprobs.py`.

**One forward pass gives all of them.** A causal LM reads the whole sequence once and outputs
logits of shape (B, T, V): at every position t, a score for each of the V vocabulary tokens
saying what should come *next*. The causal attention mask means position t only saw tokens
0..t, so a single pass gives the same answer as T separate passes on growing prefixes.

**The off-by-one shift.** The logits at position t predict token t+1. So:

| position t | token | predicted by logits at | log-prob |
|---|---|---|---|
| 0 | A | nothing | 0 (unused) |
| 1 | B | position 0 | log p(B \| A) |
| 2 | C | position 1 | log p(C \| A, B) |

In tensor terms: `logits[:, :-1]` lines up with `input_ids[:, 1:]`.

**log_softmax, then gather.** Tiny example with V = 3 and logits `[2.0, 1.0, 0.1]`:

    softmax     = [0.659, 0.242, 0.099]
    log_softmax = [-0.417, -1.417, -2.317]

If the next token was id 1, we *gather* entry 1: log-prob = −1.417. We never need the other
V − 1 entries, and `log_softmax` computes them stably: it subtracts the max logit before
exponentiating. Computing `softmax` first and then `log` can underflow to log(0) = −inf for
unlikely tokens. Logits are cast to float32 first, because bf16 keeps only about 3
significant digits.

**Padding.** Batched sequences have different lengths, so we pad them.
- `attention_mask` stops real tokens attending to padding.
- With left padding, the position indices must also skip the padding. Otherwise a sequence's
  first real token is treated as position 3 instead of 0, and its log-probs change.

**Masking.** Only tokens the policy generated get a log-prob. Prompt tokens, padding and
retrieved `<information>` tokens are set to exactly 0. Those tokens were not chosen by the
policy, so they must not receive gradient (Search-R1 Sec. 3.1).
