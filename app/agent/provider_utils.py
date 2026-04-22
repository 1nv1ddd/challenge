from __future__ import annotations

import math

from ..providers import AIProvider, Message


class AgentProviderUtilsMixin:
    def _validate_provider(self, provider_name: str) -> AIProvider:
        if provider_name not in self.providers:
            raise LookupError(f"Provider '{provider_name}' not configured")
        return self.providers[provider_name]

    @staticmethod
    async def _validate_model(provider: AIProvider, provider_name: str, model: str) -> None:
        model_ids = [m["id"] for m in provider.models]
        if model not in model_ids and hasattr(provider, "refresh_models"):
            await provider.refresh_models()
            model_ids = [m["id"] for m in provider.models]
        if model not in model_ids:
            raise ValueError(f"Model '{model}' not available for {provider_name}")

    @staticmethod
    def _normalize_temperature(value: float) -> float:
        return max(0.0, min(2.0, float(value)))

    @staticmethod
    def _normalize_messages(raw_messages: list[dict]) -> list[Message]:
        normalized: list[Message] = []
        for msg in raw_messages:
            role = msg.get("role")
            content = msg.get("content")
            if role and content is not None:
                normalized.append(Message(role=role, content=str(content)))
        return normalized

    @staticmethod
    def _estimate_tokens_text(text: str) -> int:
        return max(1, math.ceil(len(text) / 4))

    def _estimate_tokens_messages(self, messages: list[Message]) -> int:
        return sum(self._estimate_tokens_text(m.content) for m in messages)
