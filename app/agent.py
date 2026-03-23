from __future__ import annotations

from collections.abc import AsyncIterator

from .providers import AIProvider, Message, StreamResult


class SimpleChatAgent:
    """Encapsulates chat request/response logic for LLM providers."""

    def __init__(self, providers: dict[str, AIProvider]):
        self.providers = providers

    def list_models(self) -> dict[str, list[dict]]:
        return {name: prov.models for name, prov in self.providers.items()}

    def _validate_provider(self, provider_name: str) -> AIProvider:
        if provider_name not in self.providers:
            raise LookupError(f"Provider '{provider_name}' not configured")
        return self.providers[provider_name]

    @staticmethod
    def _validate_model(provider: AIProvider, provider_name: str, model: str) -> None:
        model_ids = [m["id"] for m in provider.models]
        if model not in model_ids:
            raise ValueError(f"Model '{model}' not available for {provider_name}")

    @staticmethod
    def _normalize_temperature(value: float) -> float:
        # Keep user value in a sane range accepted by providers.
        return max(0.0, min(2.0, float(value)))

    @staticmethod
    def _normalize_messages(raw_messages: list[dict]) -> list[Message]:
        messages: list[Message] = []
        for msg in raw_messages:
            role = msg.get("role")
            content = msg.get("content")
            if not role or content is None:
                continue
            messages.append(Message(role=role, content=str(content)))
        return messages

    async def stream_reply(
        self,
        provider_name: str,
        model: str,
        raw_messages: list[dict],
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamResult]:
        provider = self._validate_provider(provider_name)
        self._validate_model(provider, provider_name, model)
        normalized_temperature = self._normalize_temperature(temperature)
        messages = self._normalize_messages(raw_messages)

        async for result in provider.stream_chat(messages, model, normalized_temperature):
            yield result
