"""Per-token log-probabilities of response tokens under a causal LM, with masks.

HAND-WRITTEN BY PRINCE (see CLAUDE.md learning rules). Implements log pi_theta(o_t | q, o_<t)
used in the GRPO objective (DeepSeekMath Sec. 4.1, Eq. 3).
"""
