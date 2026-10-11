"""Thin generation wrapper: sample n completions per prompt with stop strings, using vLLM
when requested and available, and HF `generate` otherwise; plus policy-weight sync into vLLM.

Prompts go in as token-id lists (not strings), so a multi-turn rollout can continue from
the exact tokens of earlier turns. Outputs come back prompt-major: the n samples of
prompt 0, then the n samples of prompt 1, ... This is the layout that
`grpo.compute_group_advantages` expects.

Stop strings are matched on decoded TEXT (a stop like "</search>" can span several tokens).
Every output is cut at the shortest token prefix whose text contains a stop string, so
`token_ids` is exactly what the policy produced up to and including the stop.

Weight sync (vLLM only), the approach and its cost:
    vLLM keeps its own copy of the weights. After each optimizer step we copy the policy's
    state_dict into it with the model's `load_weights` (the same loader vLLM uses for HF
    checkpoints, so HF parameter names work), then reset vLLM's prefix cache, because cached
    KV blocks were computed with the old weights. Cost: one GPU-to-GPU copy of the model
    (~1 GB for 0.5B in bf16, ~6 GB for 3B). That is well under a second on an A100,
    negligible next to rollout time. vLLM also reserves `gpu_memory_utilization` of the GPU
    for its KV cache, which is the real price of colocating it with training. With the HF
    backend there is nothing to sync: generation uses the policy module itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


@dataclass
class GenerationOutput:
    prompt_index: int  # which input prompt this sample belongs to
    token_ids: list[int]  # generated tokens only (no prompt, no padding)
    text: str  # decoded token_ids
    finish_reason: str  # "stop" (hit a stop string), "eos", or "length"


# ---------------------------------------------------------------- stop strings


def truncate_at_stop(
    token_ids: list[int], tokenizer: Any, stop_strings: list[str]
) -> tuple[list[int], bool]:
    """Shortest prefix of `token_ids` whose decoded text contains a stop string.

    Returns (ids, hit). If no stop string occurs, returns (token_ids, False).
    Binary search over the prefix length: once a prefix contains a stop, every longer
    prefix does too, so this needs only ~log2(len) decodes.
    """
    if not stop_strings or not token_ids:
        return token_ids, False

    def contains_stop(k: int) -> bool:
        text = tokenizer.decode(token_ids[:k], skip_special_tokens=True)
        return any(s in text for s in stop_strings)

    if not contains_stop(len(token_ids)):
        return token_ids, False
    lo, hi = 1, len(token_ids)  # answer is in [lo, hi]
    while lo < hi:
        mid = (lo + hi) // 2
        if contains_stop(mid):
            hi = mid
        else:
            lo = mid + 1
    return token_ids[:lo], True


class StopOnStrings(StoppingCriteria):
    """HF stopping criterion: a row is done once its generated text contains a stop string.

    Only the last `window` generated tokens are decoded each step, which is enough to see
    any stop string that just completed.
    """

    def __init__(self, tokenizer: Any, stop_strings: list[str], prompt_len: int) -> None:
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.prompt_len = prompt_len
        self.window = max(len(s) for s in stop_strings) + 8

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> torch.Tensor:
        generated = input_ids[:, self.prompt_len :]
        tails = self.tokenizer.batch_decode(generated[:, -self.window :], skip_special_tokens=True)
        done = [any(s in t for s in self.stop_strings) for t in tails]
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


# ---------------------------------------------------------------- generator


class Generator:
    """Samples completions from the current policy.

    backend="hf": uses `model.generate` on the policy module (works everywhere, slow).
    backend="vllm": uses a vLLM engine loaded from `model_name` (fast, CUDA only).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        backend: str = "hf",
        model_name: str | None = None,
        vllm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend
        self.llm = None
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if backend == "vllm":
            from vllm import LLM  # noqa: PLC0415 (optional dependency)

            kwargs = {
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.4,
                "enable_prefix_caching": True,
            }
            kwargs.update(vllm_kwargs or {})
            self.llm = LLM(model=model_name, **kwargs)
        elif backend != "hf":
            raise ValueError(f"backend must be 'hf' or 'vllm', got {backend!r}")

    # ------------------------------------------------------------ public API

    def generate(
        self,
        prompt_ids: list[list[int]],
        n: int = 1,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        stop_strings: list[str] | None = None,
    ) -> list[GenerationOutput]:
        """n samples per prompt, prompt-major. temperature 0 means greedy."""
        stop_strings = list(stop_strings or [])
        if self.backend == "vllm":
            outputs = self._generate_vllm(prompt_ids, n, max_new_tokens, temperature, stop_strings)
        else:
            outputs = self._generate_hf(prompt_ids, n, max_new_tokens, temperature, stop_strings)
        for out in outputs:  # same truncation rule for both backends
            ids, hit = truncate_at_stop(out.token_ids, self.tokenizer, stop_strings)
            if hit:
                out.token_ids, out.finish_reason = ids, "stop"
            out.text = self.tokenizer.decode(out.token_ids, skip_special_tokens=True)
        return outputs

    def generate_text(self, prompts: list[str], **kwargs) -> list[GenerationOutput]:
        """Convenience: tokenize strings (no special tokens added) and call generate."""
        ids = [self.tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
        return self.generate(ids, **kwargs)

    def sync_weights(self, model: torch.nn.Module) -> None:
        """Make the generator sample from `model`'s current weights (see module docstring)."""
        self.model = model
        if self.llm is None:
            return
        state = {k: v.detach().to(torch.bfloat16) for k, v in model.state_dict().items()}
        # vLLM's internal layout changes across versions; try the known access paths.
        if hasattr(self.llm, "apply_model"):
            self.llm.apply_model(lambda m: m.load_weights(state.items()))
        else:
            engine_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            engine_model.load_weights(state.items())
        self.llm.reset_prefix_cache()

    # ------------------------------------------------------------ backends

    @torch.no_grad()
    def _generate_hf(
        self,
        prompt_ids: list[list[int]],
        n: int,
        max_new_tokens: int,
        temperature: float,
        stop_strings: list[str],
    ) -> list[GenerationOutput]:
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        device = next(self.model.parameters()).device

        # Left-pad so every prompt ends at the same column and generation starts together.
        T = max(len(p) for p in prompt_ids)
        input_ids = torch.full((len(prompt_ids), T), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, p in enumerate(prompt_ids):
            input_ids[i, T - len(p) :] = torch.tensor(p)
            attention_mask[i, T - len(p) :] = 1
        # n copies of each prompt, kept adjacent (prompt-major). Repeating rows instead of
        # num_return_sequences also works for greedy decoding.
        input_ids = input_ids.repeat_interleave(n, dim=0)
        attention_mask = attention_mask.repeat_interleave(n, dim=0)

        greedy = temperature == 0
        criteria = (
            StoppingCriteriaList([StopOnStrings(self.tokenizer, stop_strings, T)])
            if stop_strings
            else None
        )
        was_training = self.model.training
        self.model.eval()
        out = self.model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=None if greedy else temperature,
            top_p=None if greedy else 1.0,
            top_k=None if greedy else 0,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            stopping_criteria=criteria,
        )
        self.model.train(was_training)

        results = []
        for row in range(out.shape[0]):
            gen = out[row, T:].tolist()
            finish = "length"
            if eos_id is not None and eos_id in gen:
                gen = gen[: gen.index(eos_id) + 1]  # keep EOS: the policy chose to stop
                finish = "eos"
            elif pad_id in gen and pad_id != eos_id:
                gen = gen[: gen.index(pad_id)]  # row finished early (stop string), rest is padding
            results.append(GenerationOutput(row // n, gen, "", finish))
        return results

    def _generate_vllm(
        self,
        prompt_ids: list[list[int]],
        n: int,
        max_new_tokens: int,
        temperature: float,
        stop_strings: list[str],
    ) -> list[GenerationOutput]:
        from vllm import SamplingParams  # noqa: PLC0415
        from vllm.inputs import TokensPrompt  # noqa: PLC0415

        params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_new_tokens,
            stop=stop_strings or None,
            include_stop_str_in_output=True,
        )
        prompts = [TokensPrompt(prompt_token_ids=p) for p in prompt_ids]
        results = []
        for i, request in enumerate(self.llm.generate(prompts, params, use_tqdm=False)):
            for sample in request.outputs:
                finish = "length" if sample.finish_reason == "length" else "stop"
                if sample.finish_reason == "stop" and sample.stop_reason is None:
                    finish = "eos"
                results.append(GenerationOutput(i, list(sample.token_ids), sample.text, finish))
        return results
