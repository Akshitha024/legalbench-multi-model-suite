"""Resolve a string like ``anthropic-haiku`` or ``local-qwen0p5b`` to a Provider."""

from __future__ import annotations

from .base import Provider


def build(spec: str) -> Provider:
    """Provider spec format: ``<vendor>-<short-id>`` or ``<vendor>:<full-model>``.

    Examples:
      local-qwen0p5b
      anthropic-haiku
      anthropic:claude-3-5-sonnet-latest
      openai-mini
      openai:gpt-4o
      google-flash
    """
    if spec.startswith("local-"):
        from .local_hf import LocalHFProvider

        return LocalHFProvider(model=spec)

    if ":" in spec:
        vendor, model = spec.split(":", 1)
    else:
        vendor, short = spec.split("-", 1)
        resolved = _SHORT_TO_MODEL.get((vendor, short))
        if resolved is None:
            raise ValueError(
                f"unknown short id '{short}' for vendor '{vendor}'. Use vendor:full-model"
            )
        model = resolved

    if vendor == "anthropic":
        from .api_runners import AnthropicProvider

        return AnthropicProvider(model=model)
    if vendor == "openai":
        from .api_runners import OpenAIProvider

        return OpenAIProvider(model=model)
    if vendor == "google":
        from .api_runners import GoogleProvider

        return GoogleProvider(model=model)
    raise ValueError(f"unknown vendor '{vendor}'")


_SHORT_TO_MODEL: dict[tuple[str, str], str] = {
    ("anthropic", "haiku"): "claude-3-5-haiku-latest",
    ("anthropic", "sonnet"): "claude-3-5-sonnet-latest",
    ("anthropic", "opus"): "claude-3-opus-latest",
    ("openai", "mini"): "gpt-4o-mini",
    ("openai", "gpt4o"): "gpt-4o",
    ("google", "flash"): "gemini-1.5-flash",
    ("google", "pro"): "gemini-1.5-pro",
}
