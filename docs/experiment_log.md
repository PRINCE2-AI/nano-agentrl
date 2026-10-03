# Experiment log

Notable results and decisions, newest last.

## 2026-10-03: Repo scaffolding (P0)

- Created package skeleton, flat `Config` dataclass + YAML loader (unknown keys raise),
  `configs/gsm8k_0.5b.yaml`, and a smoke test.
- Defaults: G=8, clip_eps=0.2, kl_coef=0.001 (k3), max_turns=4, topk=3, temperature=1.0,
  loss_agg=token_mean, format reward off.
