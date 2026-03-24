from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

from .providers import AIProvider, Message, StreamResult


class SimpleChatAgent:
    """Encapsulates chat request/response logic for LLM providers."""

    def __init__(
        self,
        providers: dict[str, AIProvider],
        memory_path: str | Path = "data/agent_memory.json",
    ):
        self.providers = providers
        self.memory_path = Path(memory_path)
        self.history_by_conversation: dict[str, list[dict[str, str]]] = {}
        self._load_history()

    def list_models(self) -> dict[str, list[dict]]:
        return {name: prov.models for name, prov in self.providers.items()}

    def _load_history(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists():
            self.history_by_conversation = {}
            return
        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.history_by_conversation = data
            else:
                self.history_by_conversation = {}
        except (OSError, json.JSONDecodeError):
            self.history_by_conversation = {}

    def _save_history(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.memory_path.with_suffix(".tmp")
        payload = json.dumps(self.history_by_conversation, ensure_ascii=False)
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(self.memory_path)

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
        conversation_id: str,
        raw_messages: list[dict],
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamResult]:
        provider = self._validate_provider(provider_name)
        self._validate_model(provider, provider_name, model)
        normalized_temperature = self._normalize_temperature(temperature)
        incoming_messages = self._normalize_messages(raw_messages)

        if not incoming_messages:
            raise ValueError("No valid messages to send")

        stored_history = self._normalize_messages(
            self.history_by_conversation.get(conversation_id, [])
        )

        # Common case for chat UI: one new user message per request.
        if len(incoming_messages) == 1 and incoming_messages[0].role == "user":
            request_messages = [*stored_history, incoming_messages[0]]
        else:
            # Fallback mode: allow full sync payloads.
            request_messages = incoming_messages

        assistant_chunks: list[str] = []

        async for result in provider.stream_chat(
            request_messages, model, normalized_temperature
        ):
            if result.text:
                assistant_chunks.append(result.text)
            yield result

        assistant_text = "".join(assistant_chunks)
        if assistant_text:
            next_history = [
                *request_messages,
                Message(role="assistant", content=assistant_text),
            ]
            self.history_by_conversation[conversation_id] = [
                {"role": msg.role, "content": msg.content} for msg in next_history
            ]
            self._save_history()
