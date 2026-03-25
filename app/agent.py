from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator
from pathlib import Path

from .providers import AIProvider, Message, StreamResult

MODEL_CONTEXT_LIMITS = {
    "deepseek/deepseek-v3.2": 164000,
}
INPUT_PRICE_RUB_PER_MILLION = 27.0
OUTPUT_PRICE_RUB_PER_MILLION = 39.0


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
            "cost_rub_total": 0.0,
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
            defaults["cost_rub_total"] = float(
                stats.get("cost_rub_total", stats.get("cost_usd_total", defaults["cost_rub_total"]))
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

    def _summarize_history(
        self,
        messages: list[Message],
        max_items: int = 60,
        max_chars_per_item: int = 200,
    ) -> str:
        """Build a compact deterministic summary without extra LLM call."""
        if not messages:
            return "Диалог пока пуст."

        lines: list[str] = []
        start = max(0, len(messages) - max_items)
        for msg in messages[start:]:
            role = "Пользователь" if msg.role == "user" else "Ассистент"
            content = " ".join(msg.content.split())
            if len(content) > max_chars_per_item:
                content = content[: max_chars_per_item - 1] + "…"
            lines.append(f"- {role}: {content}")

        return (
            "Сводка предыдущего диалога (сжато):\n"
            + "\n".join(lines)
            + "\nИспользуй это как контекст и продолжай диалог."
        )

    def _fit_history_into_context(
        self,
        stored_history: list[Message],
        incoming_messages: list[Message],
        context_limit: int,
    ) -> tuple[list[Message], dict]:
        """Compress history when context would overflow."""
        if not incoming_messages:
            return stored_history, {
                "history_compacted": False,
                "history_tokens_before_compaction": self._estimate_tokens_messages(stored_history),
                "history_tokens_after_compaction": self._estimate_tokens_messages(stored_history),
            }

        original_history_tokens = self._estimate_tokens_messages(stored_history)
        keep_tail = 8
        summary_max_items = 80
        summary_item_chars = 220
        compacted = False

        while True:
            candidate_history = stored_history
            if stored_history:
                tail = stored_history[-keep_tail:] if keep_tail > 0 else []
                older = stored_history[:-keep_tail] if keep_tail > 0 else stored_history
                if older:
                    summary_text = self._summarize_history(
                        older,
                        max_items=summary_max_items,
                        max_chars_per_item=summary_item_chars,
                    )
                    summary_message = Message(
                        role="system",
                        content=summary_text,
                    )
                    candidate_history = [summary_message, *tail]
                    compacted = True

            candidate_tokens = self._estimate_tokens_messages(candidate_history + incoming_messages)
            if candidate_tokens <= context_limit:
                return candidate_history, {
                    "history_compacted": compacted,
                    "history_tokens_before_compaction": original_history_tokens,
                    "history_tokens_after_compaction": self._estimate_tokens_messages(candidate_history),
                }

            # Tighten compaction progressively.
            if keep_tail > 2:
                keep_tail = max(2, keep_tail - 2)
                summary_max_items = max(20, summary_max_items - 15)
                summary_item_chars = max(120, summary_item_chars - 20)
                continue

            # Last resort: only short system summary.
            summary_text = self._summarize_history(
                stored_history,
                max_items=20,
                max_chars_per_item=120,
            )
            candidate_history = [Message(role="system", content=summary_text)]
            candidate_tokens = self._estimate_tokens_messages(candidate_history + incoming_messages)
            if candidate_tokens <= context_limit:
                return candidate_history, {
                    "history_compacted": True,
                    "history_tokens_before_compaction": original_history_tokens,
                    "history_tokens_after_compaction": self._estimate_tokens_messages(candidate_history),
                }

            raise ValueError(
                f"Context overflow: unable to compact history under limit {context_limit}. "
                "Start a new chat."
            )

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

        context_limit = MODEL_CONTEXT_LIMITS.get(model, 164000)
        projected_prompt_tokens = history_tokens_before + current_request_tokens
        compaction_meta = {
            "history_compacted": False,
            "history_tokens_before_compaction": history_tokens_before,
            "history_tokens_after_compaction": history_tokens_before,
        }
        if projected_prompt_tokens > context_limit and len(incoming_messages) == 1:
            fitted_history, compaction_meta = self._fit_history_into_context(
                stored_history=stored_history,
                incoming_messages=incoming_messages,
                context_limit=context_limit,
            )
            request_messages = [*fitted_history, incoming_messages[0]]
            history_tokens_before = self._estimate_tokens_messages(fitted_history)
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
            request_cost_rub = (
                prompt_tokens * INPUT_PRICE_RUB_PER_MILLION
                + completion_tokens * OUTPUT_PRICE_RUB_PER_MILLION
            ) / 1_000_000
            prompt_for_limit = prompt_tokens if prompt_tokens > 0 else projected_prompt_tokens

            stats = conversation_state["stats"]
            stats["turns"] += 1
            stats["prompt_tokens_total"] += prompt_tokens
            stats["completion_tokens_total"] += completion_tokens
            stats["total_tokens_total"] += total_tokens
            stats["cost_rub_total"] += request_cost_rub
            self._save_history()

            enriched_meta = {
                # Existing fields preserved for backward compatibility.
                "time_ms": int((provider_meta or {}).get("time_ms", 0)),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_rub": round(request_cost_rub, 6),
                # Requested token counters.
                "current_request_tokens": current_request_tokens,
                "history_tokens": history_tokens_before,
                "response_tokens": completion_tokens or response_tokens_est,
                # Growth and overflow visibility.
                "history_tokens_after": history_tokens_after,
                "history_compacted": compaction_meta["history_compacted"],
                "history_tokens_before_compaction": compaction_meta["history_tokens_before_compaction"],
                "history_tokens_after_compaction": compaction_meta["history_tokens_after_compaction"],
                "conversation_total_tokens": int(stats["total_tokens_total"]),
                "conversation_total_cost_rub": round(float(stats["cost_rub_total"]), 6),
                "conversation_turns": int(stats["turns"]),
                "model_context_limit_tokens": context_limit,
                "prompt_usage_percent": round(
                    (prompt_for_limit / context_limit) * 100, 2
                ),
                "estimated": prompt_tokens == 0,
            }
            yield StreamResult(meta=enriched_meta)
