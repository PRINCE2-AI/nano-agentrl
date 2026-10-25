# What broke and what I learned re-implementing Search-R1

*Published on The Reliable Agent as part of
[Pass@8 was 28%. Pass^8 was 4%](https://thereliableagent.substack.com/p/pass8-was-28-pass8-was-4-same-model);
the bug section below is reused almost verbatim. Every number here comes from a real run on a
free Kaggle 2× T4 notebook.*

I wanted to understand how a language model learns to *use a tool* through reinforcement
learning. So I re-implemented [Search-R1](https://arxiv.org/abs/2503.09516) from scratch.
The model thinks, searches Wikipedia, reads what comes back, and answers. GRPO rewards it
only when the final answer is right. This post covers the bugs, and what each one taught me.

## 1. Most of RL is bookkeeping about *which tokens count*

The GRPO loss is three lines. Getting the right numbers into it is the rest of the project.

**The off-by-one.** `logits[:, t]` predicts token `t + 1`, not token `t`. Every log-prob
function has to shift by one. The test that caught my mistakes was the slowest one. It runs
the model once per token on the unpadded prefix and compares the result with the fast
batched version.

**Left padding silently changes the model.** Generation batches prompts with left padding.
If you don't pass `position_ids`, HuggingFace numbers the positions 0, 1, 2… along the padded
row. The first real token then sits at position 3, and every log-prob shifts a little. The
code doesn't crash, and the loss still goes down. Only the "padding must not matter" test
fails: it computes `cumsum(attention_mask) − 1`. I would never have noticed it in a training
curve.

**The loss mask is the whole point.** Retrieved documents are part of the sequence, but the
model didn't write them. If you train on them, you teach the model to "predict Wikipedia".
Every appended token is tagged 1 (policy) or 0 (environment) at the moment it is appended,
and the test checks that the mask is exactly 0 over every `<information>` span. The
`nomask` ablation measures how much this matters (not run yet).

## 2. Precision: bf16 weights eat your updates

With lr = 1e-6, an update to a weight of size 0.01 is about 1e-4 of the weight's value. bf16
keeps about 3 significant digits, so the update rounds away to nothing. The trainer keeps
float32 master weights and runs the forward pass under bf16 autocast. The frozen reference
model never updates, so it lives in bf16.

## 3. Small bugs the tests caught before any GPU time

- **Greedy decoding with n > 1 samples.** `num_return_sequences > 1` is illegal without
  sampling, which broke greedy evaluation. The fix was to repeat each prompt row n times.
  That also keeps the prompt-major layout the advantages need.
- **Context-window overflow.** Prompt + 2 searches + 3 turns went past the tiny test model's
  1024 positions. The length check ran *after* a turn instead of before the next one. Now
  a trajectory stops when another turn *could* overflow.
- **Question ids colliding across eval batches.** Batched evaluation restarted `qid` at 0
  in every batch, so pass^k counted two different questions as one. A test with
  `batch_size=2` and 3 questions caught it.
- **A random 0.5B model still teaches you something.** Before any training, Qwen2.5-0.5B-
  Instruct ignores the `<answer>` tags completely: 100% invalid format, so EM is 0. That is
  the starting point the RL has to move. It is also why `invalid_format_rate` is logged
  next to accuracy.

## 4. Environment problems (not RL, but they cost hours)

- The laptop's disk filled up mid-download: an 800 MB-free C: drive against a 988 MB
  model.
- TRL ≥ 1.x computes GRPO log-probs with Triton kernels, and Triton exists only on Linux.
  The TRL sanity check therefore runs only on the GPU box.
- On Windows, Python's `write_text` turned a bash script's line endings into CRLF, and bash
  read `32\r` as a number. The fix was a `.gitattributes` rule: `*.sh text eol=lf`.

## 5. What actually happened in training

**GSM8K (0.5B, 200 steps, 4.8 h): 32.4% → 47.9%** on the full test set. Most of the gain
came in the first 25 steps, and most of *that* was the model learning to finish inside the
token budget: truncated answers fell from ~60% to ~3%. With no reward for the `<answer>`
tag, the tag drifted in and out of use the whole run. RL optimizes exactly what you pay
for, and nothing else.

**Search (0.5B, 90 steps, 2.9 h).** The base model never followed the protocol: in 19 of
20 cases it chatted and stopped. A 0.1 format reward gave GRPO its first gradient, and the
model promptly found the cheapest way to collect it: `<answer>guess</answer>` in 20 tokens,
no search. It sat in that local optimum for ~45 steps. Then, over five steps, searches per
question jumped from 0.03 to 1.0 and exact match followed: **13.0% NQ, 19.6% HotpotQA**,
up from ~0.

**The humbling part.** On a lenient "is the gold answer anywhere in the text" metric,
plain one-shot RAG beats the trained agent (22.0 vs 16.0 on NQ). The base model often
*knew* the answer from the retrieved passage and just never wrote it in the required
format. RL mostly taught the agent to *say* answers well, not yet to *find* more of them.
I would not have known this without building the baseline and a second metric.

## 6. What I'd tell someone starting

1. Write the slow reference implementation first, and make every fast version match it.
2. Test "padding must not matter" and "masked tokens get exactly zero gradient" before
   anything else.
3. Log the *fraction of groups with no signal*. If every sample in a group gets the same
   reward, that question taught nothing. My first search trial was 100% zero-signal.
4. Score rewards on **policy tokens only**. Search-R1's own invalid-action message
   contains `<answer>`, and scoring the full text would have paid the model for
   receiving it.
5. Always build the dumb baseline, and look at your number with a second metric.
