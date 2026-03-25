from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator
from pathlib import Path

from .providers import AIProvider, Message, StreamResult

MODEL_CONTEXT_LIMITS = {
    "llama-3.1-8b-instant": 32768,
    "meta-llama/llama-4-scout-17b-16e-instruct": 32768,
    "llama-3.3-70b-versatile": 32768,
}


class SimpleChatAgent:
    """Encapsulates chat request/response logic for LLM providers."""

    def __init__(
        self,
        providers: dict[str, AIProvider],
        memory_path: str | Path = "data/agent_memory.json",
    ):
        self.providers = providers
        self.memory_path = Path(memory_path)
        self.state_by_conversation: dict[str, dict] = {}
        self._load_history()

    def list_models(self) -> dict[str, list[dict]]:
        return {name: prov.models for name, prov in self.providers.items()}

    def _load_history(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists():
            self.state_by_conversation = {}
            return
        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                self.state_by_conversation = {}
                return

            normalized: dict[str, dict] = {}
            for conv_id, conv_data in data.items():
                # Backward compatibility with old format: {conv_id: [messages...]}
                if isinstance(conv_data, list):
                    normalized[conv_id] = {
                        "messages": conv_data,
                        "stats": self._empty_stats(),
                    }
                    continue
                if isinstance(conv_data, dict):
                    messages = conv_data.get("messages", [])
                    stats = conv_data.get("stats", {})
                    normalized[conv_id] = {
                        "messages": messages if isinstance(messages, list) else [],
                        "stats": self._normalize_stats(stats),
                    }
            self.state_by_conversation = normalized
        except (OSError, json.JSONDecodeError):
            self.state_by_conversation = {}

    def _save_history(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.memory_path.with_suffix(".tmp")
        payload = json.dumps(self.state_by_conversation, ensure_ascii=False)
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(self.memory_path)

    @staticmethod
    def _empty_stats() -> dict:
        return {
            "turns": 0,
            "prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "total_tokens_total": 0,
            "cost_usd_total": 0.0,
        }

    def _normalize_stats(self, stats: dict) -> dict:
        defaults = self._empty_stats()
        if not isinstance(stats, dict):
            return defaults
        try:
            defaults["turns"] = int(stats.get("turns", defaults["turns"]))
            defaults["prompt_tokens_total"] = int(
                stats.get("prompt_tokens_total", defaults["prompt_tokens_total"])
            )
            defaults["completion_tokens_total"] = int(
                stats.get("completion_tokens_total", defaults["completion_tokens_total"])
            )
            defaults["total_tokens_total"] = int(
                stats.get("total_tokens_total", defaults["total_tokens_total"])
            )
            defaults["cost_usd_total"] = float(
                stats.get("cost_usd_total", defaults["cost_usd_total"])
            )
        except (TypeError, ValueError):
            return self._empty_stats()
        return defaults

    def _get_conversation_state(self, conversation_id: str) -> dict:
        if conversation_id not in self.state_by_conversation:
            self.state_by_conversation[conversation_id] = {
                "messages": [],
                "stats": self._empty_stats(),
            }
        return self.state_by_conversation[conversation_id]

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

    @staticmethod
    def _estimate_tokens_text(text: str) -> int:
        # Lightweight estimate when tokenizer is unavailable server-side.
        return max(1, math.ceil(len(text) / 4))

    def _estimate_tokens_messages(self, messages: list[Message]) -> int:
        return sum(self._estimate_tokens_text(msg.content) for msg in messages)

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

        conversation_state = self._get_conversation_state(conversation_id)
        stored_history = self._normalize_messages(conversation_state["messages"])
        history_tokens_before = self._estimate_tokens_messages(stored_history)
        current_request_tokens = self._estimate_tokens_messages(incoming_messages)

        # Common case for chat UI: one new user message per request.
        if len(incoming_messages) == 1 and incoming_messages[0].role == "user":
            request_messages = [*stored_history, incoming_messages[0]]
        else:
            # Fallback mode: allow full sync payloads.
            request_messages = incoming_messages

        context_limit = MODEL_CONTEXT_LIMITS.get(model, 32768)
        projected_prompt_tokens = history_tokens_before + current_request_tokens
        if projected_prompt_tokens > context_limit:
            raise ValueError(
                f"Context overflow: estimated {projected_prompt_tokens} tokens exceeds "
                f"limit {context_limit} for model '{model}'"
            )

        assistant_chunks: list[str] = []
        provider_meta: dict | None = None

        async for result in provider.stream_chat(
            request_messages, model, normalized_temperature
        ):
            if result.text:
                assistant_chunks.append(result.text)
                yield result
            if result.meta is not None:
                provider_meta = result.meta

        assistant_text = "".join(assistant_chunks)
        if assistant_text:
            next_history = [
                *request_messages,
                Message(role="assistant", content=assistant_text),
            ]
            conversation_state["messages"] = [
                {"role": msg.role, "content": msg.content} for msg in next_history
            ]
            response_tokens_est = self._estimate_tokens_text(assistant_text)
            history_tokens_after = self._estimate_tokens_messages(next_history)

            prompt_tokens = int((provider_meta or {}).get("prompt_tokens", 0))
            completion_tokens = int((provider_meta or {}).get("completion_tokens", 0))
            total_tokens = int((provider_meta or {}).get("total_tokens", 0))
            request_cost = float((provider_meta or {}).get("cost_usd", 0.0))

            stats = conversation_state["stats"]
            stats["turns"] += 1
            stats["prompt_tokens_total"] += prompt_tokens
            stats["completion_tokens_total"] += completion_tokens
            stats["total_tokens_total"] += total_tokens
            stats["cost_usd_total"] += request_cost
            self._save_history()

            enriched_meta = {
                # Existing fields preserved for backward compatibility.
                "time_ms": int((provider_meta or {}).get("time_ms", 0)),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": round(request_cost, 6),
                # Requested token counters.
                "current_request_tokens": current_request_tokens,
                "history_tokens": history_tokens_before,
                "response_tokens": completion_tokens or response_tokens_est,
                # Growth and overflow visibility.
                "history_tokens_after": history_tokens_after,
                "conversation_total_tokens": int(stats["total_tokens_total"]),
                "conversation_total_cost_usd": round(float(stats["cost_usd_total"]), 6),
                "conversation_turns": int(stats["turns"]),
                "model_context_limit_tokens": context_limit,
                "prompt_usage_percent": round(
                    (projected_prompt_tokens / context_limit) * 100, 2
                ),
                "estimated": True,
            }
            yield StreamResult(meta=enriched_meta)
