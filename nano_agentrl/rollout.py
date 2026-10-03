"""Batched multi-turn rollouts: generate -> <search> -> retrieve -> <information> -> continue.

HAND-WRITTEN BY PRINCE (see CLAUDE.md learning rules). Implements the interleaved
reasoning-and-search rollout of Search-R1 Sec. 3.1-3.2, including the retrieved-token
loss mask (Sec. 3.1, "Loss Masking for Retrieved Tokens").
"""
