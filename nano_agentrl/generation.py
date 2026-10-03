"""Thin generation wrapper: sample G completions per prompt with stop strings,
using vLLM when available and HF `generate` otherwise; plus policy-weight sync into vLLM."""
