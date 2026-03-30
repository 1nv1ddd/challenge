from __future__ import annotations

import json
import math
import time
from collections.abc import AsyncIterator
from pathlib import Path

from .providers import AIProvider, Message, StreamResult

MODEL_CONTEXT_LIMITS = {
    "openai/gpt-4o-mini": 128000,
}
INPUT_PRICE_RUB_PER_MILLION = 15.0
OUTPUT_PRICE_RUB_PER_MILLION = 63.0
WINDOW_SIZE_MESSAGES = 12
ALLOWED_STRATEGIES = {"sliding", "facts", "branching"}
SHORT_TERM_WINDOW = 12


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
            raw = json.loads(self.memory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.state_by_conversation = {}
            return

        if not isinstance(raw, dict):
            self.state_by_conversation = {}
            return

        normalized: dict[str, dict] = {}
        for conv_id, conv_data in raw.items():
            normalized[conv_id] = self._normalize_conversation_state(conv_data)
        self.state_by_conversation = normalized

    def _normalize_conversation_state(self, conv_data: dict | list) -> dict:
        if isinstance(conv_data, list):
            full = conv_data
            return {
                "full_messages": full,
                "facts": {},
                "branches": {
                    "main": {
                        "name": "main",
                        "from_checkpoint": None,
                        "messages": full,
                    }
                },
                "checkpoints": {},
                "memory_layers": self._empty_memory_layers(),
                "stats": self._empty_stats(),
            }

        if not isinstance(conv_data, dict):
            return {
                "full_messages": [],
                "facts": {},
                "branches": {"main": {"name": "main", "from_checkpoint": None, "messages": []}},
                "checkpoints": {},
                "memory_layers": self._empty_memory_layers(),
                "stats": self._empty_stats(),
            }

        full = conv_data.get("full_messages")
        if not isinstance(full, list):
            full = conv_data.get("messages", conv_data.get("recent_messages", []))
        if not isinstance(full, list):
            full = []

        facts = conv_data.get("facts", {})
        if not isinstance(facts, dict):
            facts = {}

        raw_branches = conv_data.get("branches", {})
        branches: dict[str, dict] = {}
        if isinstance(raw_branches, dict):
            for bid, bval in raw_branches.items():
                if not isinstance(bval, dict):
                    continue
                bmsg = bval.get("messages", [])
                if not isinstance(bmsg, list):
                    bmsg = []
                branches[bid] = {
                    "name": str(bval.get("name", bid)),
                    "from_checkpoint": bval.get("from_checkpoint"),
                    "messages": bmsg,
                }
        if "main" not in branches:
            branches["main"] = {
                "name": "main",
                "from_checkpoint": None,
                "messages": full,
            }

        checkpoints = conv_data.get("checkpoints", {})
        if not isinstance(checkpoints, dict):
            checkpoints = {}
        memory_layers = self._normalize_memory_layers(conv_data.get("memory_layers", {}))

        return {
            "full_messages": full,
            "facts": facts,
            "branches": branches,
            "checkpoints": checkpoints,
            "memory_layers": memory_layers,
            "stats": self._normalize_stats(conv_data.get("stats", {})),
        }

    def _save_history(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.memory_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state_by_conversation, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.memory_path)

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
            defaults["turns"] = int(stats.get("turns", 0))
            defaults["prompt_tokens_total"] = int(stats.get("prompt_tokens_total", 0))
            defaults["completion_tokens_total"] = int(stats.get("completion_tokens_total", 0))
            defaults["total_tokens_total"] = int(stats.get("total_tokens_total", 0))
            defaults["cost_rub_total"] = float(
                stats.get("cost_rub_total", stats.get("cost_usd_total", 0.0))
            )
        except (TypeError, ValueError):
            return self._empty_stats()
        return defaults

    def _get_conversation_state(self, conversation_id: str) -> dict:
        if conversation_id not in self.state_by_conversation:
            self.state_by_conversation[conversation_id] = {
                "full_messages": [],
                "facts": {},
                "branches": {"main": {"name": "main", "from_checkpoint": None, "messages": []}},
                "checkpoints": {},
                "memory_layers": self._empty_memory_layers(),
                "stats": self._empty_stats(),
            }
        return self.state_by_conversation[conversation_id]

    @staticmethod
    def _empty_memory_layers() -> dict:
        return {
            "short_term": {"notes": []},
            "working_memory": {},
            "long_term": {},
        }

    def _normalize_memory_layers(self, layers: dict) -> dict:
        base = self._empty_memory_layers()
        if not isinstance(layers, dict):
            return base

        short_term = layers.get("short_term", {})
        if isinstance(short_term, dict):
            notes = short_term.get("notes", [])
            if isinstance(notes, list):
                base["short_term"]["notes"] = [str(x)[:240] for x in notes[-20:]]

        working = layers.get("working_memory", {})
        if isinstance(working, dict):
            for k, v in list(working.items())[-40:]:
                base["working_memory"][str(k)[:64]] = str(v)[:320]

        long_term = layers.get("long_term", {})
        if isinstance(long_term, dict):
            for k, v in list(long_term.items())[-80:]:
                base["long_term"][str(k)[:64]] = str(v)[:320]

        return base

    def list_memory_layers(self, conversation_id: str) -> dict:
        state = self._get_conversation_state(conversation_id)
        return state.get("memory_layers", self._empty_memory_layers())

    def list_branches(self, conversation_id: str) -> dict:
        state = self._get_conversation_state(conversation_id)
        branches = state.get("branches", {})
        out = []
        for bid, b in branches.items():
            out.append(
                {
                    "id": bid,
                    "name": b.get("name", bid),
                    "message_count": len(b.get("messages", [])),
                    "from_checkpoint": b.get("from_checkpoint"),
                }
            )
        out.sort(key=lambda x: (x["id"] != "main", x["id"]))
        return {"branches": out}

    def create_checkpoint(self, conversation_id: str, branch_id: str = "main") -> dict:
        state = self._get_conversation_state(conversation_id)
        branch = state.get("branches", {}).get(branch_id)
        if branch is None:
            raise ValueError(f"Branch '{branch_id}' not found")
        checkpoint_id = f"cp_{int(time.time() * 1000)}"
        state["checkpoints"][checkpoint_id] = {
            "branch_id": branch_id,
            "full_messages": branch.get("messages", []),
            "facts": dict(state.get("facts", {})),
            "memory_layers": self._normalize_memory_layers(state.get("memory_layers", {})),
            "created_at": int(time.time()),
        }
        self._save_history()
        return {"checkpoint_id": checkpoint_id}

    def create_branch(
        self,
        conversation_id: str,
        checkpoint_id: str,
        branch_name: str | None = None,
    ) -> dict:
        state = self._get_conversation_state(conversation_id)
        checkpoint = state.get("checkpoints", {}).get(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found")

        n = 1
        while True:
            bid = f"branch-{n}"
            if bid not in state["branches"]:
                break
            n += 1

        state["branches"][bid] = {
            "name": (branch_name or bid),
            "from_checkpoint": checkpoint_id,
            "messages": checkpoint.get("full_messages", []),
        }
        self._save_history()
        return {"branch_id": bid, "name": state["branches"][bid]["name"]}

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

    def _update_facts(self, state: dict, user_text: str) -> None:
        facts = state.get("facts", {})
        if not isinstance(facts, dict):
            facts = {}

        lines = [x.strip() for x in user_text.splitlines() if x.strip()]
        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                key = k.strip().lower()[:40]
                val = v.strip()[:240]
                if key and val:
                    facts[key] = val

            lower = line.lower()
            if "цель" in lower:
                facts["goal"] = line[:240]
            if "огранич" in lower:
                facts["constraints"] = line[:240]
            if "предпочт" in lower:
                facts["preferences"] = line[:240]
            if "решени" in lower:
                facts["decisions"] = line[:240]
            if "договор" in lower:
                facts["agreements"] = line[:240]

        # Keep facts compact and stable.
        max_items = 20
        trimmed = {}
        for k in list(facts.keys())[-max_items:]:
            trimmed[k] = facts[k]
        state["facts"] = trimmed

    def _facts_system_message(self, facts: dict) -> Message | None:
        if not facts:
            return None
        lines = ["Sticky facts from dialogue:"]
        for k, v in facts.items():
            lines.append(f"- {k}: {v}")
        return Message(role="system", content="\n".join(lines))

    def _memory_system_messages(self, state: dict) -> list[Message]:
        layers = self._normalize_memory_layers(state.get("memory_layers", {}))
        msgs: list[Message] = []

        short_notes = layers["short_term"]["notes"][-5:]
        if short_notes:
            short_lines = ["Short-term notes (recent focus):"]
            short_lines.extend([f"- {x}" for x in short_notes])
            msgs.append(Message(role="system", content="\n".join(short_lines)))

        working = layers["working_memory"]
        if working:
            work_lines = ["Working memory (current task data):"]
            for k, v in working.items():
                work_lines.append(f"- {k}: {v}")
            msgs.append(Message(role="system", content="\n".join(work_lines)))

        long_term = layers["long_term"]
        if long_term:
            long_lines = ["Long-term memory (profile/decisions/knowledge):"]
            for k, v in long_term.items():
                long_lines.append(f"- {k}: {v}")
            msgs.append(Message(role="system", content="\n".join(long_lines)))

        return msgs

    def _apply_explicit_memory_save(
        self,
        state: dict,
        memory_save: dict | None,
        fallback_text: str,
    ) -> dict:
        result = {"saved": False, "layer": None, "key": None}
        if not isinstance(memory_save, dict):
            return result

        layer = str(memory_save.get("layer", "")).strip().lower()
        if layer not in {"short_term", "working_memory", "long_term"}:
            return result

        key = str(memory_save.get("key", "")).strip()[:64]
        value = str(memory_save.get("value", "")).strip()
        if not value:
            value = fallback_text.strip()
        if not value:
            return result

        layers = self._normalize_memory_layers(state.get("memory_layers", {}))
        if layer == "short_term":
            layers["short_term"]["notes"].append(value[:240])
            layers["short_term"]["notes"] = layers["short_term"]["notes"][-20:]
            result = {"saved": True, "layer": layer, "key": "note"}
        else:
            if not key:
                return result
            layers[layer][key] = value[:320]
            result = {"saved": True, "layer": layer, "key": key}

        state["memory_layers"] = layers
        return result

    def _refresh_short_term_from_messages(self, state: dict, messages: list[Message]) -> None:
        layers = self._normalize_memory_layers(state.get("memory_layers", {}))
        notes: list[str] = []
        for msg in messages[-SHORT_TERM_WINDOW:]:
            role = "U" if msg.role == "user" else "A"
            compact = " ".join(msg.content.split())
            if len(compact) > 100:
                compact = compact[:99] + "…"
            notes.append(f"{role}: {compact}")
        layers["short_term"]["notes"] = notes[-20:]
        state["memory_layers"] = layers

    def _build_context(
        self,
        state: dict,
        incoming_user: Message,
        strategy: str,
        branch_id: str,
        context_limit: int,
    ) -> tuple[list[Message], dict]:
        full_history = self._normalize_messages(state.get("full_messages", []))
        user_tokens = self._estimate_tokens_messages([incoming_user])
        memory_messages = self._memory_system_messages(state)
        memory_tokens = self._estimate_tokens_messages(memory_messages)
        full_tokens = self._estimate_tokens_messages([*memory_messages, *full_history, incoming_user])

        if strategy == "branching":
            branches = state.get("branches", {})
            branch = branches.get(branch_id)
            if branch is None:
                branch = branches.get("main")
                branch_id = "main"
            branch_history = self._normalize_messages(branch.get("messages", []))
            tail = branch_history[-WINDOW_SIZE_MESSAGES:]
            request_messages = [*memory_messages, *tail, incoming_user]
            with_tokens = self._estimate_tokens_messages(request_messages)
            no_tokens = self._estimate_tokens_messages([*memory_messages, *branch_history, incoming_user])
            dropped = max(0, len(branch_history) - len(tail))
            history_tokens = self._estimate_tokens_messages(tail)
            meta = {
                "context_strategy": "branching",
                "branch_id": branch_id,
                "history_tokens": history_tokens,
                "recent_history_tokens": history_tokens,
                "memory_tokens": memory_tokens,
                "facts_count": len(state.get("facts", {})),
                "prompt_tokens_with_compression_est": with_tokens,
                "prompt_tokens_no_compression_est": no_tokens,
                "saved_tokens_est": max(0, no_tokens - with_tokens),
                "compressed_messages_count": dropped,
                "summary_used": False,
                "summary_tokens": 0,
                "compression_enabled": False,
            }
        elif strategy == "facts":
            tail = full_history[-WINDOW_SIZE_MESSAGES:]
            facts_msg = self._facts_system_message(state.get("facts", {}))
            request_messages = [*memory_messages, *([facts_msg] if facts_msg else []), *tail, incoming_user]
            with_tokens = self._estimate_tokens_messages(request_messages)
            dropped = max(0, len(full_history) - len(tail))
            history_tokens = self._estimate_tokens_messages(tail)
            meta = {
                "context_strategy": "facts",
                "branch_id": "main",
                "history_tokens": history_tokens,
                "recent_history_tokens": history_tokens,
                "memory_tokens": memory_tokens,
                "facts_count": len(state.get("facts", {})),
                "prompt_tokens_with_compression_est": with_tokens,
                "prompt_tokens_no_compression_est": full_tokens + self._estimate_tokens_messages([facts_msg]) if facts_msg else full_tokens,
                "saved_tokens_est": max(0, full_tokens - with_tokens),
                "compressed_messages_count": dropped,
                "summary_used": False,
                "summary_tokens": 0,
                "compression_enabled": False,
            }
        else:  # sliding
            tail = full_history[-WINDOW_SIZE_MESSAGES:]
            request_messages = [*memory_messages, *tail, incoming_user]
            with_tokens = self._estimate_tokens_messages(request_messages)
            dropped = max(0, len(full_history) - len(tail))
            history_tokens = self._estimate_tokens_messages(tail)
            meta = {
                "context_strategy": "sliding",
                "branch_id": "main",
                "history_tokens": history_tokens,
                "recent_history_tokens": history_tokens,
                "memory_tokens": memory_tokens,
                "facts_count": len(state.get("facts", {})),
                "prompt_tokens_with_compression_est": with_tokens,
                "prompt_tokens_no_compression_est": full_tokens,
                "saved_tokens_est": max(0, full_tokens - with_tokens),
                "compressed_messages_count": dropped,
                "summary_used": False,
                "summary_tokens": 0,
                "compression_enabled": False,
            }

        if meta["prompt_tokens_with_compression_est"] > context_limit:
            raise ValueError(
                f"Context overflow: estimated {meta['prompt_tokens_with_compression_est']} tokens "
                f"exceeds limit {context_limit}"
            )
        meta["current_request_tokens"] = user_tokens
        return request_messages, meta

    def _append_turn(
        self,
        state: dict,
        strategy: str,
        branch_id: str,
        user_msg: Message,
        assistant_msg: Message,
    ) -> None:
        if strategy == "branching":
            branch = state["branches"].setdefault(
                branch_id,
                {"name": branch_id, "from_checkpoint": None, "messages": []},
            )
            branch_messages = self._normalize_messages(branch.get("messages", []))
            branch_messages.extend([user_msg, assistant_msg])
            branch["messages"] = [{"role": m.role, "content": m.content} for m in branch_messages]
            self._refresh_short_term_from_messages(state, branch_messages)

            # Keep main dialog unchanged unless branch is main.
            if branch_id == "main":
                state["full_messages"] = branch["messages"]
        else:
            history = self._normalize_messages(state.get("full_messages", []))
            history.extend([user_msg, assistant_msg])
            serialized = [{"role": m.role, "content": m.content} for m in history]
            state["full_messages"] = serialized
            state["branches"]["main"] = {
                "name": "main",
                "from_checkpoint": None,
                "messages": serialized,
            }
            self._refresh_short_term_from_messages(state, history)

    async def stream_reply(
        self,
        provider_name: str,
        model: str,
        conversation_id: str,
        raw_messages: list[dict],
        temperature: float = 0.7,
        context_strategy: str = "sliding",
        branch_id: str = "main",
        memory_save: dict | None = None,
    ) -> AsyncIterator[StreamResult]:
        provider = self._validate_provider(provider_name)
        self._validate_model(provider, provider_name, model)
        strategy = context_strategy if context_strategy in ALLOWED_STRATEGIES else "sliding"
        normalized_temperature = self._normalize_temperature(temperature)
        incoming = self._normalize_messages(raw_messages)
        if not incoming:
            raise ValueError("No valid messages to send")

        # Fallback for non-chat clients: use payload as-is.
        if len(incoming) != 1 or incoming[0].role != "user":
            request_messages = incoming
            context_meta = {
                "context_strategy": strategy,
                "branch_id": branch_id,
                "history_tokens": self._estimate_tokens_messages(incoming),
                "recent_history_tokens": self._estimate_tokens_messages(incoming),
                "memory_tokens": 0,
                "facts_count": 0,
                "prompt_tokens_with_compression_est": self._estimate_tokens_messages(incoming),
                "prompt_tokens_no_compression_est": self._estimate_tokens_messages(incoming),
                "saved_tokens_est": 0,
                "compressed_messages_count": 0,
                "summary_used": False,
                "summary_tokens": 0,
                "compression_enabled": False,
                "current_request_tokens": self._estimate_tokens_messages(incoming),
                "memory_save": {"saved": False, "layer": None, "key": None},
            }
            user_msg = incoming[-1]
        else:
            state = self._get_conversation_state(conversation_id)
            user_msg = incoming[0]
            self._update_facts(state, user_msg.content)
            memory_save_meta = self._apply_explicit_memory_save(
                state=state,
                memory_save=memory_save,
                fallback_text=user_msg.content,
            )
            context_limit = MODEL_CONTEXT_LIMITS.get(model, 128000)
            request_messages, context_meta = self._build_context(
                state=state,
                incoming_user=user_msg,
                strategy=strategy,
                branch_id=branch_id,
                context_limit=context_limit,
            )
            context_meta["memory_save"] = memory_save_meta

        assistant_chunks: list[str] = []
        provider_meta: dict | None = None
        async for result in provider.stream_chat(request_messages, model, normalized_temperature):
            if result.text:
                assistant_chunks.append(result.text)
                yield result
            if result.meta is not None:
                provider_meta = result.meta

        assistant_text = "".join(assistant_chunks)
        if not assistant_text:
            return

        state = self._get_conversation_state(conversation_id)
        assistant_msg = Message(role="assistant", content=assistant_text)
        self._append_turn(state, strategy, context_meta["branch_id"], user_msg, assistant_msg)

        prompt_tokens = int((provider_meta or {}).get("prompt_tokens", 0))
        completion_tokens = int((provider_meta or {}).get("completion_tokens", 0))
        total_tokens = int((provider_meta or {}).get("total_tokens", 0))
        response_tokens_est = self._estimate_tokens_text(assistant_text)
        context_limit = MODEL_CONTEXT_LIMITS.get(model, 128000)

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
            else int(context_meta["prompt_tokens_with_compression_est"])
        )
        prompt_pct = round((prompt_for_limit / context_limit) * 100, 2)
        memory_layers = self._normalize_memory_layers(state.get("memory_layers", {}))

        enriched_meta = {
            "time_ms": int((provider_meta or {}).get("time_ms", 0)),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_rub": round(request_cost_rub, 6),
            "current_request_tokens": int(context_meta["current_request_tokens"]),
            "history_tokens": int(context_meta["history_tokens"]),
            "recent_history_tokens": int(context_meta["recent_history_tokens"]),
            "memory_tokens": int(context_meta.get("memory_tokens", 0)),
            "facts_count": int(context_meta["facts_count"]),
            "response_tokens": completion_tokens or response_tokens_est,
            "conversation_total_tokens": int(stats["total_tokens_total"]),
            "conversation_total_cost_rub": round(float(stats["cost_rub_total"]), 6),
            "conversation_turns": int(stats["turns"]),
            "model_context_limit_tokens": context_limit,
            "prompt_usage_percent": prompt_pct,
            "summary_used": False,
            "summary_tokens": 0,
            "prompt_tokens_no_compression_est": int(context_meta["prompt_tokens_no_compression_est"]),
            "prompt_tokens_with_compression_est": int(context_meta["prompt_tokens_with_compression_est"]),
            "saved_tokens_est": int(context_meta["saved_tokens_est"]),
            "compressed_messages_count": int(context_meta["compressed_messages_count"]),
            "context_strategy": context_meta["context_strategy"],
            "branch_id": context_meta["branch_id"],
            "memory_short_notes_count": len(memory_layers["short_term"]["notes"]),
            "memory_working_count": len(memory_layers["working_memory"]),
            "memory_long_count": len(memory_layers["long_term"]),
            "memory_saved": bool(context_meta.get("memory_save", {}).get("saved", False)),
            "memory_saved_layer": context_meta.get("memory_save", {}).get("layer"),
            "memory_saved_key": context_meta.get("memory_save", {}).get("key"),
            "compression_enabled": False,
            "estimated": prompt_tokens == 0,
        }
        yield StreamResult(meta=enriched_meta)
