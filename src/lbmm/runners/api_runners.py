"""Adapters for hosted providers: Anthropic, OpenAI, Google.

Each adapter follows the same Provider protocol. The provider keys come from
env vars (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``GOOGLE_API_KEY``).
We do not silently fall back to mocks; if the key is missing, the runner
raises so the harness fails loudly and the user knows what to set.
"""

from __future__ import annotations

import os
import time

from ..types import ProviderResponse
from .base import Provider


def _require(env: str) -> str:
    v = os.environ.get(env)
    if not v:
        raise RuntimeError(
            f"missing env {env}; set it in .env or your shell before invoking the runner"
        )
    return v


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str = "claude-3-5-haiku-latest") -> None:
        from anthropic import Anthropic

        self.model = model
        self._client = Anthropic(api_key=_require("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str, max_tokens: int = 64) -> ProviderResponse:
        t0 = time.perf_counter()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
        )
        return ProviderResponse(
            text=text.strip(),
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            latency_ms=latency_ms,
        )


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=_require("OPENAI_API_KEY"))

    def generate(self, prompt: str, max_tokens: int = 64) -> ProviderResponse:
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        return ProviderResponse(
            text=text,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_ms=latency_ms,
        )


class GoogleProvider(Provider):
    name = "google"

    def __init__(self, model: str = "gemini-1.5-flash") -> None:
        from google import genai

        self.model = model
        self._client = genai.Client(api_key=_require("GOOGLE_API_KEY"))

    def generate(self, prompt: str, max_tokens: int = 64) -> ProviderResponse:
        t0 = time.perf_counter()
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"max_output_tokens": max_tokens, "temperature": 0.0},
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = (getattr(resp, "text", "") or "").strip()
        usage = getattr(resp, "usage_metadata", None)
        return ProviderResponse(
            text=text,
            prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0,
            completion_tokens=int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0,
            latency_ms=latency_ms,
        )
