from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import ProviderResponse


class Provider(ABC):
    name: str
    model: str

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 64) -> ProviderResponse: ...

    def close(self) -> None:
        """Release resources (close clients, drop model weights)."""
        return None
