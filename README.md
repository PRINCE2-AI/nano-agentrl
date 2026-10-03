# nano-AgentRL

A minimal, readable, from-scratch PyTorch re-implementation of
[Search-R1](https://arxiv.org/abs/2503.09516): multi-turn GRPO that teaches an LLM to
interleave reasoning with search-engine calls, plus an analysis of how agent
*reliability* changes during RL training.

Work in progress. See `CLAUDE.md` for the spec.

```bash
pip install -e ".[dev]"
pytest -q
```
