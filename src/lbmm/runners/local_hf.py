"""Local HuggingFace transformers runner.

For laptop-scale eval where the user does not want to spend on API tokens
(or cannot, while writing this harness). Default model is Qwen2.5-0.5B-
Instruct (~1GB on disk, runs CPU-only in a few seconds per item). Swap to
1.5B or 3B with --model.

The 'latency_ms' here is wall clock for generate(); 'cost' stays $0 in the
price table, which makes the cost-vs-quality plot honest: local models look
like a free vertical line on the quality axis.
"""

from __future__ import annotations

import time

from loguru import logger

from ..types import ProviderResponse
from .base import Provider

_MODEL_ALIASES = {
    "local-qwen0p5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "local-qwen1p5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "local-qwen3b": "Qwen/Qwen2.5-3B-Instruct",
}


def resolve_model(alias_or_hf_id: str) -> str:
    return _MODEL_ALIASES.get(alias_or_hf_id, alias_or_hf_id)


class LocalHFProvider(Provider):
    name = "local"

    def __init__(self, model: str = "local-qwen0p5b", device: str = "cpu") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = resolve_model(model)
        self._device = device
        logger.info("loading {} onto {}...", self.model, device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)  # type: ignore[no-untyped-call]
        self._llm = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype="auto",
            device_map=device,
        )

    def generate(self, prompt: str, max_tokens: int = 64) -> ProviderResponse:
        import torch

        messages = [
            {"role": "system", "content": "You answer legal-domain questions briefly."},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        prompt_tokens = int(inputs.input_ids.shape[1])

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._llm.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # deterministic for eval
                pad_token_id=self._tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        generated_ids = out[0][prompt_tokens:]
        completion = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return ProviderResponse(
            text=completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(generated_ids.shape[0]),
            latency_ms=latency_ms,
        )
