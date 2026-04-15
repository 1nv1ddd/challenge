from __future__ import annotations

import json
import re
import time
from pathlib import Path

from ..agent_constants import (
    GLOBAL_KEY,
    INVARIANT_KEY_MAX_LEN,
    INVARIANT_VAL_MAX_LEN,
    INVARIANTS_MAX_ITEMS,
    LONG_TERM_ALLOWED_KEYS,
    TASK_PHASE_TO_DEFAULTS,
    TASK_PHASES,
    WINDOW_SIZE_MESSAGES,
)


class AgentStateMixin:
    def _load_history(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists():
            self.state_by_conversation = {}
            self.global_memory = self._normalize_global_memory({})
            return
        try:
            raw = json.loads(self.memory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.state_by_conversation = {}
            self.global_memory = self._normalize_global_memory({})
            return

        if not isinstance(raw, dict):
            self.state_by_conversation = {}
            self.global_memory = self._normalize_global_memory({})
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

    @staticmethod
    def _empty_task_state() -> dict:
        defaults = TASK_PHASE_TO_DEFAULTS["planning"]
        return {
            "phase": "planning",
            "current_step": defaults["current_step"],
            "expected_action": defaults["expected_action"],
            "status": "running",
            "is_paused": False,
            "task_active": False,
            "last_event": "init",
            "updated_at": int(time.time()),
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
                "invariants": {},
                "branches": {
                    "main": {
                        "name": "main",
                        "from_checkpoint": None,
                        "messages": full,
                    }
                },
                "checkpoints": {},
                "stats": self._empty_stats(),
                "task_state": self._empty_task_state(),
            }

        if not isinstance(conv_data, dict):
            return {
                "full_messages": [],
                "short_term_messages": [],
                "working_memory": {},
                "facts": {},
                "invariants": {},
                "branches": {"main": {"name": "main", "from_checkpoint": None, "messages": []}},
                "checkpoints": {},
                "stats": self._empty_stats(),
                "task_state": self._empty_task_state(),
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

        raw_inv = conv_data.get("invariants", {})
        invariants = self._normalize_invariants(raw_inv if isinstance(raw_inv, dict) else {})

        return {
            "full_messages": full,
            "short_term_messages": short_term,
            "working_memory": working_memory,
            "facts": facts,
            "invariants": invariants,
            "branches": branches,
            "checkpoints": checkpoints,
            "stats": self._normalize_stats(conv_data.get("stats", {})),
            "task_state": self._normalize_task_state(conv_data.get("task_state", {})),
        }

    def _normalize_task_state(self, data: dict) -> dict:
        base = self._empty_task_state()
        if not isinstance(data, dict):
            return base
        phase = str(data.get("phase", base["phase"])).strip().lower()
        if phase not in TASK_PHASES:
            phase = "planning"
        current_step = str(
            data.get("current_step", TASK_PHASE_TO_DEFAULTS[phase]["current_step"])
        ).strip()[:220]
        expected_action = str(
            data.get("expected_action", TASK_PHASE_TO_DEFAULTS[phase]["expected_action"])
        ).strip()[:260]
        if not current_step:
            current_step = TASK_PHASE_TO_DEFAULTS[phase]["current_step"]
        if not expected_action:
            expected_action = TASK_PHASE_TO_DEFAULTS[phase]["expected_action"]
        status = str(data.get("status", "paused" if data.get("is_paused") else "running")).strip().lower()
        if status not in {"running", "paused"}:
            status = "running"
        is_paused = status == "paused"
        return {
            "phase": phase,
            "current_step": current_step,
            "expected_action": expected_action,
            "status": status,
            "is_paused": is_paused,
            "task_active": bool(data.get("task_active", False)),
            "last_event": str(data.get("last_event", ""))[:48],
            "updated_at": int(data.get("updated_at", int(time.time()))),
        }

    def _normalize_global_memory(self, data: dict) -> dict:
        if not isinstance(data, dict):
            data = {}
        long_term = self._sanitize_long_term(
            self._normalize_kv_dict(data.get("long_term", {}), max_items=120)
        )

        raw_profiles = data.get("profiles", {})
        profiles: dict[str, dict] = {}
        if isinstance(raw_profiles, dict):
            for profile_id, profile in raw_profiles.items():
                if not isinstance(profile, dict):
                    continue
                pid = str(profile_id).strip()[:64]
                if not pid:
                    continue
                profiles[pid] = {
                    "name": str(profile.get("name", pid)).strip()[:80] or pid,
                    "style": str(profile.get("style", "")).strip()[:500],
                    "format": str(profile.get("format", "")).strip()[:500],
                    "constraints": str(profile.get("constraints", "")).strip()[:500],
                }
        if "default" not in profiles:
            profiles["default"] = {
                "name": "Default",
                "style": "",
                "format": "",
                "constraints": "",
            }
        default_profile_id = str(data.get("default_profile_id", "default")).strip()
        if default_profile_id not in profiles:
            default_profile_id = "default"

        return {
            "long_term": long_term,
            "profiles": profiles,
            "default_profile_id": default_profile_id,
        }

    @staticmethod
    def _sanitize_long_term(data: dict) -> dict:
        if not isinstance(data, dict):
            return {}
        filtered: dict[str, str] = {}
        for key, val in data.items():
            k = str(key).strip().lower()
            v = str(val).strip()
            if k not in LONG_TERM_ALLOWED_KEYS or not v:
                continue
            # Keep only compact user-profile facts in long-term memory.
            if k == "budget":
                if len(v) > 48:
                    continue
                if not re.search(r"\d", v):
                    continue
                filtered[k] = v[:48]
                continue
            if k == "deadline":
                if len(v) > 64:
                    continue
                filtered[k] = v[:64]
                continue
            if len(v) > 220:
                continue
            filtered[k] = v[:220]
        return filtered

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

    def _normalize_invariants(self, data: dict) -> dict[str, str]:
        if not isinstance(data, dict):
            return {}
        out: dict[str, str] = {}
        for k, v in list(data.items())[-INVARIANTS_MAX_ITEMS:]:
            key = str(k).strip()[:INVARIANT_KEY_MAX_LEN]
            val = str(v).strip()[:INVARIANT_VAL_MAX_LEN]
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
                "invariants": {},
                "branches": {"main": {"name": "main", "from_checkpoint": None, "messages": []}},
                "checkpoints": {},
                "stats": self._empty_stats(),
                "task_state": self._empty_task_state(),
            }
        return self.state_by_conversation[conversation_id]
