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
WINDOW_SIZE_MESSAGES = 10
ALLOWED_STRATEGIES = {"sliding", "facts", "branching"}
GLOBAL_KEY = "__global__"


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
        self.global_memory: dict = {"long_term": {}}
        self._load_history()

    def list_models(self) -> dict[str, list[dict]]:
        return {name: prov.models for name, prov in self.providers.items()}

    def _load_history(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists():
            self.state_by_conversation = {}
            self.global_memory = {"long_term": {}}
            return
        try:
            raw = json.loads(self.memory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.state_by_conversation = {}
            self.global_memory = {"long_term": {}}
            return

        if not isinstance(raw, dict):
            self.state_by_conversation = {}
            self.global_memory = {"long_term": {}}
            return

        self.global_memory = self._normalize_global_memory(raw.get(GLOBAL_KEY, {}))

        normalized: dict[str, dict] = {}
        for conv_id, conv_data in raw.items():
            if conv_id == GLOBAL_KEY:
                continue
            normalized[conv_id] = self._normalize_conversation_state(conv_data)
        self.state_by_conversation = normalized

    def _save_history(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**self.state_by_conversation, GLOBAL_KEY: self.global_memory}
        tmp = self.memory_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
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

    def _normalize_conversation_state(self, conv_data: dict | list) -> dict:
        if isinstance(conv_data, list):
            full = conv_data
            return {
                "full_messages": full,
                "short_term_messages": full[-WINDOW_SIZE_MESSAGES:],
                "working_memory": {},
                "facts": {},
                "branches": {
                    "main": {
                        "name": "main",
                        "from_checkpoint": None,
                        "messages": full,
                    }
                },
                "checkpoints": {},
                "stats": self._empty_stats(),
            }

        if not isinstance(conv_data, dict):
            return {
                "full_messages": [],
                "short_term_messages": [],
                "working_memory": {},
                "facts": {},
                "branches": {"main": {"name": "main", "from_checkpoint": None, "messages": []}},
                "checkpoints": {},
                "stats": self._empty_stats(),
            }

        full = conv_data.get("full_messages")
        if not isinstance(full, list):
            full = conv_data.get("messages", [])
        if not isinstance(full, list):
            full = []

        short_term = conv_data.get("short_term_messages", full[-WINDOW_SIZE_MESSAGES:])
        if not isinstance(short_term, list):
            short_term = full[-WINDOW_SIZE_MESSAGES:]

        working_memory = conv_data.get("working_memory")
        if not isinstance(working_memory, dict):
            legacy_layers = conv_data.get("memory_layers", {})
            if isinstance(legacy_layers, dict):
                working_memory = legacy_layers.get("working_memory", {})
            else:
                working_memory = {}
        working_memory = self._normalize_kv_dict(working_memory, max_items=40)

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

        return {
            "full_messages": full,
            "short_term_messages": short_term,
            "working_memory": working_memory,
            "facts": facts,
            "branches": branches,
            "checkpoints": checkpoints,
            "stats": self._normalize_stats(conv_data.get("stats", {})),
        }

    def _normalize_global_memory(self, data: dict) -> dict:
        if not isinstance(data, dict):
            return {"long_term": {}}
        long_term = data.get("long_term", {})
        return {"long_term": self._normalize_kv_dict(long_term, max_items=120)}

    @staticmethod
    def _normalize_kv_dict(data: dict, max_items: int) -> dict:
        if not isinstance(data, dict):
            return {}
        out: dict[str, str] = {}
        for k, v in list(data.items())[-max_items:]:
            key = str(k).strip()[:64]
            val = str(v).strip()[:320]
            if key and val:
                out[key] = val
        return out

    def _get_conversation_state(self, conversation_id: str) -> dict:
        if conversation_id not in self.state_by_conversation:
            self.state_by_conversation[conversation_id] = {
                "full_messages": [],
                "short_term_messages": [],
                "working_memory": {},
                "facts": {},
                "branches": {"main": {"name": "main", "from_checkpoint": None, "messages": []}},
                "checkpoints": {},
                "stats": self._empty_stats(),
            }
        return self.state_by_conversation[conversation_id]

    def list_memory_layers(self, conversation_id: str, branch_id: str = "main") -> dict:
        state = self._get_conversation_state(conversation_id)
        branches = state.get("branches", {})
        branch = branches.get(branch_id) or branches.get("main", {"messages": []})
        source_msgs = self._normalize_messages(branch.get("messages", []))
        short_msgs = source_msgs[-WINDOW_SIZE_MESSAGES:]
        return {
            "short_term": {
                "window_size": WINDOW_SIZE_MESSAGES,
                "messages": [{"role": m.role, "content": m.content} for m in short_msgs],
            },
            "working_memory": state.get("working_memory", {}),
            "long_term": self.global_memory.get("long_term", {}),
        }

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
            "working_memory": dict(state.get("working_memory", {})),
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

        trimmed = {}
        for k in list(facts.keys())[-20:]:
            trimmed[k] = facts[k]
        state["facts"] = trimmed

    def _auto_update_long_term(self, user_text: str) -> list[str]:
        long_term = dict(self.global_memory.get("long_term", {}))
        updated_keys: list[str] = []
        for line in [x.strip() for x in user_text.splitlines() if x.strip()]:
            if ":" in line:
                k, v = line.split(":", 1)
                key = k.strip().lower().replace(" ", "_")[:64]
                val = v.strip()[:320]
                if key and val and len(val) >= 3:
                    long_term[key] = val
                    updated_keys.append(key)
                continue

            lower = line.lower()
            if "предпочт" in lower:
                long_term["preferences"] = line[:320]
                updated_keys.append("preferences")
            if "решен" in lower:
                long_term["decisions"] = line[:320]
                updated_keys.append("decisions")
            if "профил" in lower:
                long_term["profile"] = line[:320]
                updated_keys.append("profile")
            if "бюджет" in lower:
                long_term["budget"] = line[:320]
                updated_keys.append("budget")
            if "дедлайн" in lower:
                long_term["deadline"] = line[:320]
                updated_keys.append("deadline")

        self.global_memory["long_term"] = self._normalize_kv_dict(long_term, max_items=120)
        uniq: list[str] = []
        for key in updated_keys:
            if key not in uniq:
                uniq.append(key)
        return uniq

    def _working_memory_system_message(self, working: dict) -> Message | None:
        if not working:
            return None
        lines = ["Working memory (current task):"]
        for k, v in working.items():
            lines.append(f"- {k}: {v}")
        return Message(role="system", content="\n".join(lines))

    def _long_term_system_message(self) -> Message | None:
        long_term = self.global_memory.get("long_term", {})
        if not long_term:
            return None
        lines = ["Long-term memory (profile/decisions/knowledge):"]
        for k, v in long_term.items():
            lines.append(f"- {k}: {v}")
        return Message(role="system", content="\n".join(lines))

    def _facts_system_message(self, facts: dict) -> Message | None:
        if not facts:
            return None
        lines = ["Sticky facts from dialogue:"]
        for k, v in facts.items():
            lines.append(f"- {k}: {v}")
        return Message(role="system", content="\n".join(lines))

    def _auto_update_working_memory(self, state: dict, user_text: str) -> list[str]:
        working = dict(state.get("working_memory", {}))
        updated: list[str] = []
        lines = [x.strip() for x in user_text.splitlines() if x.strip()]

        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                key = k.strip().lower().replace(" ", "_")[:64]
                val = v.strip()[:320]
                if key and val:
                    working[key] = val
                    updated.append(key)
                continue

            lower = line.lower()
            if "задач" in lower or "scope" in lower:
                working["task_scope"] = line[:320]
                updated.append("task_scope")
            if "цель" in lower:
                working["task_goal"] = line[:320]
                updated.append("task_goal")
            if "дедлайн" in lower:
                working["deadline"] = line[:320]
                updated.append("deadline")
            if "бюджет" in lower:
                working["budget"] = line[:320]
                updated.append("budget")
            if "огранич" in lower:
                working["constraints"] = line[:320]
                updated.append("constraints")

        state["working_memory"] = self._normalize_kv_dict(working, max_items=40)
        uniq: list[str] = []
        for key in updated:
            if key not in uniq:
                uniq.append(key)
        return uniq

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

        working_msg = self._working_memory_system_message(state.get("working_memory", {}))
        long_term_msg = self._long_term_system_message()
        memory_messages = [m for m in [working_msg, long_term_msg] if m is not None]
        memory_tokens = self._estimate_tokens_messages(memory_messages)

        if strategy == "branching":
            branches = state.get("branches", {})
            branch = branches.get(branch_id)
            if branch is None:
                branch = branches.get("main")
                branch_id = "main"
            branch_history = self._normalize_messages(branch.get("messages", []))
            short_history = branch_history[-WINDOW_SIZE_MESSAGES:]
            request_messages = [*memory_messages, *short_history, incoming_user]
            no_tokens = self._estimate_tokens_messages([*memory_messages, *branch_history, incoming_user])
            dropped = max(0, len(branch_history) - len(short_history))
            history_tokens = self._estimate_tokens_messages(short_history)
        elif strategy == "facts":
            short_history = full_history[-WINDOW_SIZE_MESSAGES:]
            facts_msg = self._facts_system_message(state.get("facts", {}))
            fact_list = [facts_msg] if facts_msg else []
            request_messages = [*memory_messages, *fact_list, *short_history, incoming_user]
            no_tokens = self._estimate_tokens_messages([*memory_messages, *fact_list, *full_history, incoming_user])
            dropped = max(0, len(full_history) - len(short_history))
            history_tokens = self._estimate_tokens_messages(short_history)
            branch_id = "main"
        else:
            short_history = full_history[-WINDOW_SIZE_MESSAGES:]
            request_messages = [*memory_messages, *short_history, incoming_user]
            no_tokens = self._estimate_tokens_messages([*memory_messages, *full_history, incoming_user])
            dropped = max(0, len(full_history) - len(short_history))
            history_tokens = self._estimate_tokens_messages(short_history)
            branch_id = "main"

        with_tokens = self._estimate_tokens_messages(request_messages)
        if with_tokens > context_limit:
            raise ValueError(
                f"Context overflow: estimated {with_tokens} tokens exceeds limit {context_limit}"
            )

        meta = {
            "context_strategy": strategy,
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
            "current_request_tokens": user_tokens,
            "short_term_count": len(short_history),
        }
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
            state["short_term_messages"] = branch["messages"][-WINDOW_SIZE_MESSAGES:]
            if branch_id == "main":
                state["full_messages"] = branch["messages"]
        else:
            history = self._normalize_messages(state.get("full_messages", []))
            history.extend([user_msg, assistant_msg])
            serialized = [{"role": m.role, "content": m.content} for m in history]
            state["full_messages"] = serialized
            state["short_term_messages"] = serialized[-WINDOW_SIZE_MESSAGES:]
            state["branches"]["main"] = {
                "name": "main",
                "from_checkpoint": None,
                "messages": serialized,
            }

    async def stream_reply(
        self,
        provider_name: str,
        model: str,
        conversation_id: str,
        raw_messages: list[dict],
        temperature: float = 0.7,
        context_strategy: str = "sliding",
        branch_id: str = "main",
    ) -> AsyncIterator[StreamResult]:
        provider = self._validate_provider(provider_name)
        self._validate_model(provider, provider_name, model)
        strategy = context_strategy if context_strategy in ALLOWED_STRATEGIES else "sliding"
        normalized_temperature = self._normalize_temperature(temperature)
        incoming = self._normalize_messages(raw_messages)
        if not incoming:
            raise ValueError("No valid messages to send")

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
                "memory_auto_working_keys": [],
                "memory_auto_keys": [],
                "short_term_count": len(incoming),
            }
            user_msg = incoming[-1]
            state = self._get_conversation_state(conversation_id)
        else:
            state = self._get_conversation_state(conversation_id)
            user_msg = incoming[0]
            self._update_facts(state, user_msg.content)
            working_keys = self._auto_update_working_memory(state, user_msg.content)
            auto_keys = self._auto_update_long_term(user_msg.content)
            context_limit = MODEL_CONTEXT_LIMITS.get(model, 128000)
            request_messages, context_meta = self._build_context(
                state=state,
                incoming_user=user_msg,
                strategy=strategy,
                branch_id=branch_id,
                context_limit=context_limit,
            )
            context_meta["memory_auto_working_keys"] = working_keys
            context_meta["memory_auto_keys"] = auto_keys

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

        working_memory = self._normalize_kv_dict(state.get("working_memory", {}), max_items=40)
        long_term = self._normalize_kv_dict(self.global_memory.get("long_term", {}), max_items=120)

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
            "memory_short_notes_count": int(context_meta.get("short_term_count", WINDOW_SIZE_MESSAGES)),
            "memory_working_count": len(working_memory),
            "memory_long_count": len(long_term),
            "memory_auto_working_count": len(context_meta.get("memory_auto_working_keys", [])),
            "memory_auto_working_keys": context_meta.get("memory_auto_working_keys", []),
            "memory_auto_saved_count": len(context_meta.get("memory_auto_keys", [])),
            "memory_auto_saved_keys": context_meta.get("memory_auto_keys", []),
            "compression_enabled": False,
            "estimated": prompt_tokens == 0,
        }
        yield StreamResult(meta=enriched_meta)
