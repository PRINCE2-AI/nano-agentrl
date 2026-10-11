# What broke and what I learned re-implementing Search-R1

*Draft. Sections marked TODO wait for the GPU runs. Everything else happened while building
the code.*

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
`nomask` ablation measures how much this matters. TODO: result.

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

## 5. Accuracy is not reliability

TODO (after the sweep): pass@1 vs pass^8 over training, the gap between them, and which
failure types RL removes first. See [reliability.md](reliability.md).

## 6. What I'd tell someone starting

1. Write the slow reference implementation first, and make every fast version match it.
2. Test "padding must not matter" and "masked tokens get exactly zero gradient" before
   anything else.
3. Log the *fraction of groups with no signal*. If every sample in a group gets the same
   reward, that question taught nothing.
4. TODO: lessons from the real training runs.
