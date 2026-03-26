from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator
from pathlib import Path

from .providers import AIProvider, Message, StreamResult

MODEL_CONTEXT_LIMITS = {
    "openai/gpt-4o-mini": 128000,
}
INPUT_PRICE_RUB_PER_MILLION = 15.0
OUTPUT_PRICE_RUB_PER_MILLION = 63.0

KEEP_LAST_MESSAGES = 12
SUMMARIZE_BATCH_MESSAGES = 10


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
                # Legacy format: {conv_id: [messages...]}
                if isinstance(conv_data, list):
                    normalized[conv_id] = {
                        "recent_messages": conv_data,
                        "summary": "",
                        "summary_source_messages": 0,
                        "summary_source_tokens_est": 0,
                        "stats": self._empty_stats(),
                    }
                    continue

                if isinstance(conv_data, dict):
                    # Old format used "messages"; new uses "recent_messages".
                    recent = conv_data.get("recent_messages", conv_data.get("messages", []))
                    summary = conv_data.get("summary", "")
                    summary_source_messages = int(conv_data.get("summary_source_messages", 0))
                    summary_source_tokens_est = int(conv_data.get("summary_source_tokens_est", 0))
                    stats = conv_data.get("stats", {})
                    normalized[conv_id] = {
                        "recent_messages": recent if isinstance(recent, list) else [],
                        "summary": summary if isinstance(summary, str) else "",
                        "summary_source_messages": max(0, summary_source_messages),
                        "summary_source_tokens_est": max(0, summary_source_tokens_est),
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
                "recent_messages": [],
                "summary": "",
                "summary_source_messages": 0,
                "summary_source_tokens_est": 0,
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

    def _summarize_chunk(self, messages: list[Message], max_chars_per_item: int = 180) -> str:
        """Deterministic cheap summary for a chunk of messages."""
        lines: list[str] = []
        for msg in messages:
            role = "Пользователь" if msg.role == "user" else "Ассистент"
            content = " ".join(msg.content.split())
            if len(content) > max_chars_per_item:
                content = content[: max_chars_per_item - 1] + "…"
            lines.append(f"- {role}: {content}")
        return "\n".join(lines)

    def _merge_summary(self, old_summary: str, chunk_messages: list[Message]) -> str:
        chunk_summary = self._summarize_chunk(chunk_messages)
        if old_summary.strip():
            merged = old_summary.strip() + "\n" + chunk_summary
        else:
            merged = chunk_summary
        # Keep summary bounded in size.
        max_chars = 12000
        if len(merged) > max_chars:
            merged = "…\n" + merged[-max_chars:]
        return merged

    def _compress_state_history(self, state: dict) -> bool:
        """Keep latest N raw messages; move older messages into summary every 10 messages."""
        recent = self._normalize_messages(state.get("recent_messages", []))
        if len(recent) <= KEEP_LAST_MESSAGES:
            state["recent_messages"] = [{"role": m.role, "content": m.content} for m in recent]
            return False

        changed = False
        while len(recent) > KEEP_LAST_MESSAGES:
            overflow = len(recent) - KEEP_LAST_MESSAGES
            take = min(SUMMARIZE_BATCH_MESSAGES, overflow)
            chunk = recent[:take]
            recent = recent[take:]

            state["summary"] = self._merge_summary(state.get("summary", ""), chunk)
            state["summary_source_messages"] = int(state.get("summary_source_messages", 0)) + len(chunk)
            state["summary_source_tokens_est"] = int(state.get("summary_source_tokens_est", 0)) + self._estimate_tokens_messages(chunk)
            changed = True

        state["recent_messages"] = [{"role": m.role, "content": m.content} for m in recent]
        return changed

    def _build_request_messages(
        self,
        model: str,
        state: dict,
        incoming_messages: list[Message],
        context_limit: int,
    ) -> tuple[list[Message], dict]:
        recent_history = self._normalize_messages(state.get("recent_messages", []))
        summary_text = state.get("summary", "").strip()
        summary_msg_tokens = self._estimate_tokens_text(summary_text) if summary_text else 0

        if len(incoming_messages) != 1 or incoming_messages[0].role != "user":
            # Fallback for non-chat payloads from clients.
            fallback_messages = incoming_messages
            fallback_tokens = self._estimate_tokens_messages(fallback_messages)
            return fallback_messages, {
                "summary_used": bool(summary_text),
                "summary_tokens": summary_msg_tokens,
                "recent_history_tokens": self._estimate_tokens_messages(recent_history),
                "history_tokens": summary_msg_tokens + self._estimate_tokens_messages(recent_history),
                "prompt_tokens_with_compression_est": fallback_tokens,
                "prompt_tokens_no_compression_est": fallback_tokens,
                "saved_tokens_est": 0,
            }

        current_user = incoming_messages[0]
        current_request_tokens = self._estimate_tokens_messages([current_user])
        recent_tokens = self._estimate_tokens_messages(recent_history)
        source_tokens = int(state.get("summary_source_tokens_est", 0))

        request_messages: list[Message] = []
        if summary_text:
            request_messages.append(
                Message(
                    role="system",
                    content=(
                        "Краткая сводка прошлой части диалога:\n"
                        f"{summary_text}\n"
                        "Используй эту сводку как контекст и продолжай разговор."
                    ),
                )
            )
        request_messages.extend(recent_history)
        request_messages.append(current_user)

        compressed_prompt_tokens = summary_msg_tokens + recent_tokens + current_request_tokens
        uncompressed_prompt_tokens = source_tokens + recent_tokens + current_request_tokens
        saved_tokens = max(0, uncompressed_prompt_tokens - compressed_prompt_tokens)

        if compressed_prompt_tokens > context_limit:
            # Force extra compaction of recent history into summary until it fits.
            temp_state = {
                "recent_messages": [{"role": m.role, "content": m.content} for m in recent_history],
                "summary": summary_text,
                "summary_source_messages": int(state.get("summary_source_messages", 0)),
                "summary_source_tokens_est": source_tokens,
            }
            while compressed_prompt_tokens > context_limit and len(temp_state["recent_messages"]) > 2:
                extra_take = min(SUMMARIZE_BATCH_MESSAGES, len(temp_state["recent_messages"]) - 2)
                chunk = self._normalize_messages(temp_state["recent_messages"][:extra_take])
                temp_state["recent_messages"] = temp_state["recent_messages"][extra_take:]
                temp_state["summary"] = self._merge_summary(temp_state.get("summary", ""), chunk)
                temp_state["summary_source_messages"] += len(chunk)
                temp_state["summary_source_tokens_est"] += self._estimate_tokens_messages(chunk)

                recent_history = self._normalize_messages(temp_state["recent_messages"])
                summary_text = temp_state["summary"]
                summary_msg_tokens = self._estimate_tokens_text(summary_text)
                recent_tokens = self._estimate_tokens_messages(recent_history)
                source_tokens = int(temp_state["summary_source_tokens_est"])
                compressed_prompt_tokens = summary_msg_tokens + recent_tokens + current_request_tokens
                uncompressed_prompt_tokens = source_tokens + recent_tokens + current_request_tokens
                saved_tokens = max(0, uncompressed_prompt_tokens - compressed_prompt_tokens)

            state["recent_messages"] = temp_state["recent_messages"]
            state["summary"] = temp_state["summary"]
            state["summary_source_messages"] = temp_state["summary_source_messages"]
            state["summary_source_tokens_est"] = temp_state["summary_source_tokens_est"]

            request_messages = []
            if summary_text:
                request_messages.append(
                    Message(
                        role="system",
                        content=(
                            "Краткая сводка прошлой части диалога:\n"
                            f"{summary_text}\n"
                            "Используй эту сводку как контекст и продолжай разговор."
                        ),
                    )
                )
            request_messages.extend(recent_history)
            request_messages.append(current_user)

        if compressed_prompt_tokens > context_limit:
            raise ValueError(
                f"Context overflow: estimated {compressed_prompt_tokens} tokens exceeds "
                f"limit {context_limit} for model '{model}'"
            )

        return request_messages, {
            "summary_used": bool(summary_text),
            "summary_tokens": summary_msg_tokens,
            "recent_history_tokens": recent_tokens,
            "history_tokens": summary_msg_tokens + recent_tokens,
            "prompt_tokens_with_compression_est": compressed_prompt_tokens,
            "prompt_tokens_no_compression_est": uncompressed_prompt_tokens,
            "saved_tokens_est": saved_tokens,
        }

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

        state = self._get_conversation_state(conversation_id)
        self._compress_state_history(state)

        context_limit = MODEL_CONTEXT_LIMITS.get(model, 128000)
        request_messages, compression_meta = self._build_request_messages(
            model=model,
            state=state,
            incoming_messages=incoming_messages,
            context_limit=context_limit,
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
        if not assistant_text:
            return

        recent_history = self._normalize_messages(state.get("recent_messages", []))
        next_recent = [*recent_history, incoming_messages[-1], Message(role="assistant", content=assistant_text)]
        state["recent_messages"] = [{"role": m.role, "content": m.content} for m in next_recent]
        self._compress_state_history(state)

        response_tokens_est = self._estimate_tokens_text(assistant_text)

        prompt_tokens = int((provider_meta or {}).get("prompt_tokens", 0))
        completion_tokens = int((provider_meta or {}).get("completion_tokens", 0))
        total_tokens = int((provider_meta or {}).get("total_tokens", 0))
        request_cost_rub = (
            prompt_tokens * INPUT_PRICE_RUB_PER_MILLION
            + completion_tokens * OUTPUT_PRICE_RUB_PER_MILLION
        ) / 1_000_000

        stats = state["stats"]
        stats["turns"] += 1
        stats["prompt_tokens_total"] += prompt_tokens
        stats["completion_tokens_total"] += completion_tokens
        stats["total_tokens_total"] += total_tokens
        stats["cost_rub_total"] += request_cost_rub
        self._save_history()

        prompt_for_limit = (
            prompt_tokens
            if prompt_tokens > 0
            else int(compression_meta["prompt_tokens_with_compression_est"])
        )
        prompt_pct = round((prompt_for_limit / context_limit) * 100, 2)

        enriched_meta = {
            "time_ms": int((provider_meta or {}).get("time_ms", 0)),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_rub": round(request_cost_rub, 6),
            "current_request_tokens": self._estimate_tokens_messages(incoming_messages),
            "history_tokens": int(compression_meta["history_tokens"]),
            "response_tokens": completion_tokens or response_tokens_est,
            "conversation_total_tokens": int(stats["total_tokens_total"]),
            "conversation_total_cost_rub": round(float(stats["cost_rub_total"]), 6),
            "conversation_turns": int(stats["turns"]),
            "model_context_limit_tokens": context_limit,
            "prompt_usage_percent": prompt_pct,
            # Compression-related metrics.
            "summary_used": bool(compression_meta["summary_used"]),
            "summary_tokens": int(compression_meta["summary_tokens"]),
            "recent_history_tokens": int(compression_meta["recent_history_tokens"]),
            "prompt_tokens_no_compression_est": int(compression_meta["prompt_tokens_no_compression_est"]),
            "prompt_tokens_with_compression_est": int(compression_meta["prompt_tokens_with_compression_est"]),
            "saved_tokens_est": int(compression_meta["saved_tokens_est"]),
            "compressed_messages_count": int(state.get("summary_source_messages", 0)),
            "estimated": prompt_tokens == 0,
        }
        yield StreamResult(meta=enriched_meta)
