from __future__ import annotations

import json
import math
import re
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
INVARIANTS_MAX_ITEMS = 30
INVARIANT_KEY_MAX_LEN = 80
INVARIANT_VAL_MAX_LEN = 600
ALLOWED_STRATEGIES = {"sliding", "facts", "branching"}
GLOBAL_KEY = "__global__"
TASK_PHASES = ("planning", "plan_approved", "execution", "validation", "done")
# Controlled transitions (Day 15): no skips (e.g. execution only after plan_approved).
TASK_ALLOWED_EDGES: dict[str, tuple[str, ...]] = {
    "planning": ("plan_approved",),
    "plan_approved": ("execution",),
    "execution": ("validation",),
    "validation": ("done",),
    "done": (),
}
TASK_EVENT_NEW_TASK = "new_task"
TASK_EVENT_ASSISTANT_TURN_COMPLETED = "assistant_turn_completed"
TASK_EVENT_PAUSE = "pause"
TASK_EVENT_RESUME = "resume"
LONG_TERM_ALLOWED_KEYS = {
    "profile",
    "preferences",
    "decisions",
    "budget",
    "deadline",
    "style",
    "format",
    "language",
    "tone",
}
TASK_PHASE_TO_DEFAULTS = {
    "planning": {
        "current_step": "Define scope and acceptance criteria",
        "expected_action": "Provide goal, constraints, and desired result",
    },
    "plan_approved": {
        "current_step": "Plan approved — implementation only",
        "expected_action": (
            "Implement strictly per approved plan; do not restart planning unless user asks"
        ),
    },
    "execution": {
        "current_step": "Implement the agreed plan",
        "expected_action": "Proceed with implementation and share progress",
    },
    "validation": {
        "current_step": "Verify behavior and quality",
        "expected_action": "Run checks/tests and confirm requirements",
    },
    "done": {
        "current_step": "Task completed",
        "expected_action": "No action required",
    },
}
# Short, phase-specific instructions so the model cannot confuse id "plan_approved" with "still planning".
TASK_PHASE_MODEL_GUIDANCE: dict[str, str] = {
    "planning": (
        "Planning only: scope, risks, acceptance criteria, questions. "
        "No full implementation yet. The plan is NOT approved until the user clearly confirms; "
        "after your reply you remain in planning until they approve (or use Next in UI)."
    ),
    "plan_approved": (
        "CRITICAL: phase plan_approved means the plan is ALREADY APPROVED by the workflow "
        "(not a request to approve again). The user may now ask for module layout, pseudocode, "
        "or code — you MUST produce that. "
        "It is an ERROR to say you are still in 'планирование'/planning or that you cannot write code. "
        "Proceed with implementation-aligned output for this turn."
    ),
    "execution": (
        "Implementation: concrete code, files, steps. No fake task closure or customer sign-off."
    ),
    "validation": (
        "Validation: tests, checklists, evidence. Phase 'done' is only after user confirms closure "
        "(or manual Next); a plain 'continue' does not finish the task."
    ),
    "done": "Done: short wrap-up only.",
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
        self.global_memory: dict = {
            "long_term": {},
            "profiles": {
                "default": {
                    "name": "Default",
                    "style": "",
                    "format": "",
                    "constraints": "",
                }
            },
            "default_profile_id": "default",
        }
        self._load_history()

    def list_models(self) -> dict[str, list[dict]]:
        return {name: prov.models for name, prov in self.providers.items()}

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
            "invariants": self._normalize_invariants(state.get("invariants", {})),
        }

    def list_profiles(self) -> dict:
        profiles = self.global_memory.get("profiles", {})
        items = []
        for profile_id, profile in profiles.items():
            items.append(
                {
                    "id": profile_id,
                    "name": profile.get("name", profile_id),
                    "style": profile.get("style", ""),
                    "format": profile.get("format", ""),
                    "constraints": profile.get("constraints", ""),
                }
            )
        items.sort(key=lambda x: (x["id"] != "default", x["id"]))
        return {
            "profiles": items,
            "default_profile_id": self.global_memory.get("default_profile_id", "default"),
        }

    def list_task_state(self, conversation_id: str) -> dict:
        state = self._get_conversation_state(conversation_id)
        core = self._normalize_task_state(state.get("task_state", {}))
        return {
            **core,
            "allowed_next_phases": list(TASK_ALLOWED_EDGES.get(core["phase"], ())),
            "ok": True,
        }

    @staticmethod
    def _next_phase_linear(current: str) -> str | None:
        order = {
            "planning": "plan_approved",
            "plan_approved": "execution",
            "execution": "validation",
            "validation": "done",
            "done": None,
        }
        return order.get(current)

    def _illegal_transition_message(self, current: str, target: str) -> str:
        allowed = TASK_ALLOWED_EDGES.get(current, ())
        if current == "done" and target != "done":
            return (
                f"Illegal transition '{current}' -> '{target}': 'done' is terminal; "
                "use action 'reset' for a new task."
            )
        return (
            f"Illegal transition '{current}' -> '{target}'. "
            f"Allowed next phases from '{current}': {list(allowed)}."
        )

    @staticmethod
    def _is_explicit_plan_approval_message(text: str) -> bool:
        """User clearly approves the plan — required to leave planning without manual Next."""
        t = (text or "").strip().lower()
        if len(t) < 8:
            return False
        if "не утверждаю" in t or "не согласен" in t or "не принимаю" in t:
            return False
        if "отклоняю" in t or "отклон" in t:
            return False
        if ("утверждаю" in t or "утвержден" in t) and "план" in t:
            return True
        if "план" in t and ("одобряю" in t or "одобрен" in t or "принимаю" in t):
            return True
        if "согласен" in t and "план" in t:
            return True
        if "approve" in t and "plan" in t:
            return True
        return False

    def _promote_to_plan_approved_if_user_approved(self, state: dict, user_text: str) -> None:
        ts = self._normalize_task_state(state.get("task_state", {}))
        if not ts.get("task_active") or ts["phase"] != "planning":
            return
        if not self._is_explicit_plan_approval_message(user_text):
            return
        d = TASK_PHASE_TO_DEFAULTS["plan_approved"]
        state["task_state"] = self._normalize_task_state(
            {
                **ts,
                "phase": "plan_approved",
                "current_step": d["current_step"],
                "expected_action": d["expected_action"],
            }
        )

    @staticmethod
    def _is_explicit_task_completion_message(text: str) -> bool:
        """User clearly closes the task — required to leave validation without manual Next."""
        t = (text or "").strip().lower()
        if len(t) < 12:
            return False
        if "не заверш" in t or "не закрыв" in t:
            return False
        phrases = (
            "закрываем задачу",
            "задачу закрываем",
            "задача закрыта",
            "закрыть задачу",
            "задача выполнена",
            "считаем задачу выполненной",
            "считаю задачу выполненной",
            "можно закрывать задачу",
            "подтверждаю завершение",
            "подтверждаю закрытие",
            "фиксируем завершение",
            "task done",
            "mark task complete",
        )
        return any(p in t for p in phrases)

    def _promote_validation_to_done_if_user_confirms(self, state: dict, user_text: str) -> None:
        ts = self._normalize_task_state(state.get("task_state", {}))
        if not ts.get("task_active") or ts["phase"] != "validation":
            return
        if not self._is_explicit_task_completion_message(user_text):
            return
        d = TASK_PHASE_TO_DEFAULTS["done"]
        state["task_state"] = self._normalize_task_state(
            {
                **ts,
                "phase": "done",
                "current_step": d["current_step"],
                "expected_action": d["expected_action"],
            }
        )

    def list_invariants(self, conversation_id: str) -> dict:
        state = self._get_conversation_state(conversation_id)
        inv = self._normalize_invariants(state.get("invariants", {}))
        return {"invariants": inv, "count": len(inv)}

    def set_invariants(
        self,
        conversation_id: str,
        invariants: dict | None = None,
        replace: bool = True,
    ) -> dict:
        state = self._get_conversation_state(conversation_id)
        patch = self._normalize_invariants(invariants if isinstance(invariants, dict) else {})
        if replace:
            state["invariants"] = patch
        else:
            merged = dict(self._normalize_invariants(state.get("invariants", {})))
            merged.update(patch)
            state["invariants"] = self._normalize_invariants(merged)
        self._save_history()
        inv = state["invariants"]
        return {"invariants": inv, "count": len(inv)}

    def update_task_state(
        self,
        conversation_id: str,
        phase: str | None = None,
        current_step: str | None = None,
        expected_action: str | None = None,
        action: str | None = None,
    ) -> dict:
        state = self._get_conversation_state(conversation_id)
        base = self._normalize_task_state(state.get("task_state", {}))
        normalized_action = str(action or "").strip().lower()
        err: str | None = None
        sim = dict(base)

        if normalized_action == "pause":
            sim = self._transition_task_state(sim, TASK_EVENT_PAUSE)
        elif normalized_action == "resume":
            sim = self._transition_task_state(sim, TASK_EVENT_RESUME)
        elif normalized_action == "reset":
            sim = self._empty_task_state()
        elif normalized_action == "next":
            cur = sim["phase"]
            nxt = self._next_phase_linear(cur)
            if nxt is None:
                err = "Cannot advance: already at terminal phase 'done'."
            else:
                sim["phase"] = nxt
                sim["task_active"] = True
                d = TASK_PHASE_TO_DEFAULTS[nxt]
                sim["current_step"] = d["current_step"]
                sim["expected_action"] = d["expected_action"]
        elif normalized_action:
            err = f"Unknown action: {normalized_action!r}"

        if err is None and phase is not None:
            phase_norm = str(phase).strip().lower()
            if phase_norm not in TASK_PHASES:
                err = f"Unknown phase: {phase_norm!r}"
            else:
                cur = sim["phase"]
                if phase_norm != cur:
                    if phase_norm not in TASK_ALLOWED_EDGES.get(cur, ()):
                        err = self._illegal_transition_message(cur, phase_norm)
                    else:
                        sim["phase"] = phase_norm
                        sim["task_active"] = True
                        d = TASK_PHASE_TO_DEFAULTS[phase_norm]
                        sim["current_step"] = d["current_step"]
                        sim["expected_action"] = d["expected_action"]

        if err is None:
            if current_step is not None:
                custom_step = str(current_step).strip()[:220]
                if custom_step:
                    sim["current_step"] = custom_step
            if expected_action is not None:
                custom_expected = str(expected_action).strip()[:260]
                if custom_expected:
                    sim["expected_action"] = custom_expected

        sim["is_paused"] = sim.get("status") == "paused"
        sim["updated_at"] = int(time.time())

        if err is not None:
            return {
                **base,
                "allowed_next_phases": list(TASK_ALLOWED_EDGES.get(base["phase"], ())),
                "ok": False,
                "error": err,
            }

        state["task_state"] = self._normalize_task_state(sim)
        self._save_history()
        out = self._normalize_task_state(state["task_state"])
        return {
            **out,
            "allowed_next_phases": list(TASK_ALLOWED_EDGES.get(out["phase"], ())),
            "ok": True,
        }

    def _transition_task_state(
        self,
        task_state: dict,
        event: str,
        *,
        advance_phase: bool = True,
    ) -> dict:
        s = self._normalize_task_state(task_state)
        phase = s["phase"]

        if event == TASK_EVENT_NEW_TASK:
            defaults = TASK_PHASE_TO_DEFAULTS["planning"]
            s["phase"] = "planning"
            s["current_step"] = defaults["current_step"]
            s["expected_action"] = defaults["expected_action"]
            s["task_active"] = True
            s["status"] = "running"
        elif event == TASK_EVENT_PAUSE:
            s["status"] = "paused"
            s["expected_action"] = "Resume to continue current generation"
        elif event == TASK_EVENT_RESUME:
            s["status"] = "running"
            if s["task_active"]:
                s["expected_action"] = TASK_PHASE_TO_DEFAULTS[s["phase"]]["expected_action"]
        elif event == TASK_EVENT_ASSISTANT_TURN_COMPLETED:
            # One legal step forward per completed turn (no skipping plan_approved). Can be suppressed
            # (e.g. resume turn, smalltalk, pause) via advance_phase=False.
            if (
                advance_phase
                and s["task_active"]
                and s["status"] == "running"
            ):
                nxt = self._next_phase_linear(s["phase"])
                if nxt is not None:
                    # Stay in planning until the user approves the plan (message) or uses action "next";
                    # do not auto-jump when the assistant merely presented a draft plan.
                    if s["phase"] == "planning" and nxt == "plan_approved":
                        pass
                    elif s["phase"] == "validation" and nxt == "done":
                        # Stay in validation until user explicitly closes the task or uses action "next".
                        pass
                    else:
                        s["phase"] = nxt
                        d = TASK_PHASE_TO_DEFAULTS[nxt]
                        s["current_step"] = d["current_step"]
                        s["expected_action"] = d["expected_action"]

        s["is_paused"] = s["status"] == "paused"
        s["last_event"] = event
        s["updated_at"] = int(time.time())
        return self._normalize_task_state(s)

    def upsert_profile(
        self,
        profile_id: str,
        name: str,
        style: str,
        format_pref: str,
        constraints: str,
    ) -> dict:
        pid = str(profile_id or "").strip()[:64]
        if not pid:
            raise ValueError("profile_id is required")
        profile = {
            "name": (str(name).strip()[:80] or pid),
            "style": str(style).strip()[:500],
            "format": str(format_pref).strip()[:500],
            "constraints": str(constraints).strip()[:500],
        }
        profiles = dict(self.global_memory.get("profiles", {}))
        profiles[pid] = profile
        self.global_memory["profiles"] = profiles
        if not self.global_memory.get("default_profile_id"):
            self.global_memory["default_profile_id"] = "default"
        self._save_history()
        return {"ok": True, "profile_id": pid}

    def _resolve_profile(self, profile_id: str | None) -> tuple[str, dict]:
        profiles = self.global_memory.get("profiles", {})
        default_id = self.global_memory.get("default_profile_id", "default")
        requested = str(profile_id or "").strip()
        pid = requested if requested in profiles else default_id
        profile = profiles.get(pid, profiles.get("default", {}))
        return pid, profile

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
            "invariants": dict(self._normalize_invariants(state.get("invariants", {}))),
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
        long_term = dict(self._sanitize_long_term(self.global_memory.get("long_term", {})))
        updated_keys: list[str] = []
        for line in [x.strip() for x in user_text.splitlines() if x.strip()]:
            if ":" in line:
                k, v = line.split(":", 1)
                key = k.strip().lower().replace(" ", "_")[:64]
                val = v.strip()[:320]
                if key in LONG_TERM_ALLOWED_KEYS and val and len(val) >= 3:
                    long_term[key] = val
                    updated_keys.append(key)
                continue

            lower = line.lower()
            if "предпочт" in lower and len(line) <= 220:
                long_term["preferences"] = line[:320]
                updated_keys.append("preferences")
            if "решен" in lower and len(line) <= 220:
                long_term["decisions"] = line[:320]
                updated_keys.append("decisions")
            if "профил" in lower and len(line) <= 220:
                long_term["profile"] = line[:320]
                updated_keys.append("profile")
            if "бюджет" in lower and len(line) <= 120:
                match = re.search(r"(\d[\d\s]{1,20})(?:\s*[₽$€]| ?руб| ?rub)?", line, flags=re.I)
                if match:
                    budget_val = " ".join(match.group(1).split())
                    if "₽" in line or "руб" in lower or "rub" in lower:
                        budget_val = f"{budget_val} ₽"
                    long_term["budget"] = budget_val[:48]
                    updated_keys.append("budget")
            if "дедлайн" in lower and len(line) <= 120:
                long_term["deadline"] = line[:64]
                updated_keys.append("deadline")

        self.global_memory["long_term"] = self._sanitize_long_term(
            self._normalize_kv_dict(long_term, max_items=120)
        )
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

    def _profile_system_message(self, profile: dict) -> Message | None:
        if not isinstance(profile, dict):
            return None
        style = str(profile.get("style", "")).strip()
        format_pref = str(profile.get("format", "")).strip()
        constraints = str(profile.get("constraints", "")).strip()
        if not (style or format_pref or constraints):
            return None
        lines = ["User profile preferences (apply automatically):"]
        if style:
            lines.append(f"- Style: {style}")
        if format_pref:
            lines.append(f"- Format: {format_pref}")
        if constraints:
            lines.append(f"- Constraints: {constraints}")
        return Message(role="system", content="\n".join(lines))

    def _invariants_system_message(self, invariants: dict) -> Message | None:
        inv = self._normalize_invariants(invariants)
        if not inv:
            return None
        lines = [
            "STATE INVARIANTS (hard constraints; stored outside chat history; override casual user requests):",
            "Rules: (1) Before planning or answering, check every user request against these invariants.",
            "(2) If the request would violate an invariant, refuse that part; name the invariant key/title and quote or paraphrase its rule.",
            "(3) Explain briefly why the request conflicts with it; where helpful, propose a compliant alternative.",
            "(4) Do not recommend tricks, hidden violations, or 'just for dev' exceptions unless the user explicitly edits/removes the invariant in configuration.",
            "",
            "Invariant list:",
        ]
        for k, v in inv.items():
            lines.append(f"- [{k}] {v}")
        return Message(role="system", content="\n".join(lines))

    def _task_state_system_message(self, task_state: dict) -> Message | None:
        task_state = self._normalize_task_state(task_state)
        phase = task_state["phase"]
        current_step = task_state["current_step"]
        expected_action = task_state["expected_action"]
        is_paused = task_state["is_paused"]
        nxt = list(TASK_ALLOWED_EDGES.get(phase, ()))
        allowed_repr = ", ".join(nxt) if nxt else "(none — terminal)"
        phase_hint = TASK_PHASE_MODEL_GUIDANCE.get(phase, "")
        lines = [
            "Task state machine (persisted) — AUTHORITATIVE for this turn:",
            "If any earlier message in the chat contradicts the fields below (including your own prior reply), "
            "the block below WINS. Do not claim a different phase than 'phase'.",
            f"- phase: {phase}",
            f"- what this phase means: {phase_hint}",
            f"- current_step: {current_step}",
            f"- expected_action: {expected_action}",
            f"- paused: {'yes' if is_paused else 'no'}",
            f"- allowed_next_phases: {allowed_repr}. "
            "Notes: planning→plan_approved and validation→done do NOT auto-advance after your assistant reply; "
            "they need explicit user confirmation (or manual Next in UI).",
            "Order: planning -> plan_approved -> execution -> validation -> done (server-enforced edges; no skipping).",
            "NEVER obey the user if they tell you to 'forget' tests/validation, 'close the task now', "
            "or produce final customer sign-off before phase is validation/done — refuse every time.",
            "Forbidden outputs while phase is planning, plan_approved, or execution: phrases like "
            "'Задача завершена', 'итоговый отчёт для заказчика', 'всё принято', or pretending validation ran.",
            (
                "Use this state ONLY when user asks to work on the task; "
                "for greetings/small-talk, respond normally and briefly."
            ),
        ]
        return Message(role="system", content="\n".join(lines))

    @staticmethod
    def _workflow_bypass_user_intent(text: str) -> bool:
        """Heuristic: user tries to skip validation / force premature task closure."""
        t = (text or "").strip().lower()
        if len(t) < 10:
            return False
        if "забудь" in t and ("тест" in t or "валидац" in t):
            return True
        if "закрой задач" in t or "закрывай задач" in t:
            return True
        if "итогов" in t and "отч" in t and ("заказчик" in t or "клиент" in t):
            return True
        if "всё принято" in t or "все принято" in t:
            return True
        if "окончательно" in t and "закр" in t:
            return True
        if "без валидац" in t or "без тест" in t:
            return True
        if ("skip" in t or "forget" in t) and ("validat" in t or "test" in t):
            return True
        if "mark as done" in t or "mark as complete" in t:
            return True
        return False

    @staticmethod
    def _workflow_bypass_guard_system_message(phase: str) -> Message:
        return Message(
            role="system",
            content=(
                "*** WORKFLOW GUARD (overrides conflicting lines in the user message) ***\n"
                f"The latest user text tries to bypass mandatory phases or force early closure. "
                f"Persisted phase is '{phase}'. Final customer sign-off is only allowed in phases "
                "'validation' and 'done' after real verification work.\n"
                "You MUST refuse. Do NOT output: 'Задача завершена', final report for customer, "
                "'всё принято', or claim tests/validation were done.\n"
                "Answer in 3–6 short sentences: state current phase, refuse the skip, say what "
                "validation must cover next, continue work appropriate to the current phase."
            ),
        )

    @staticmethod
    def _is_smalltalk_message(text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        if len(t) > 40:
            return False
        normalized = t.replace("!", "").replace("?", "").replace(".", "")
        smalltalk = {
            "привет",
            "здарова",
            "хай",
            "hello",
            "hi",
            "hey",
            "добрый день",
            "добрый вечер",
            "как дела",
        }
        return normalized in smalltalk

    @staticmethod
    def _is_task_intent_message(text: str) -> bool:
        t = (text or "").strip().lower()
        if not t or len(t) < 24:
            return False
        markers = (
            "нужно",
            "сделай",
            "план",
            "реализ",
            "провер",
            "метрик",
            "запуст",
            "задач",
            "mvp",
            "project",
            "task",
        )
        return any(m in t for m in markers)

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
        profile: dict,
        active_profile_id: str,
        include_task_state: bool,
        strategy: str,
        branch_id: str,
        context_limit: int,
    ) -> tuple[list[Message], dict]:
        full_history = self._normalize_messages(state.get("full_messages", []))
        user_tokens = self._estimate_tokens_messages([incoming_user])

        working_msg = self._working_memory_system_message(state.get("working_memory", {}))
        long_term_msg = self._long_term_system_message()
        profile_msg = self._profile_system_message(profile)
        inv_msg = self._invariants_system_message(state.get("invariants", {}))
        task_state = self._normalize_task_state(state.get("task_state", {}))
        task_msg = self._task_state_system_message(task_state) if include_task_state else None
        guard_phases = ("planning", "plan_approved", "execution")
        workflow_guard = None
        if (
            include_task_state
            and task_state.get("phase") in guard_phases
            and self._workflow_bypass_user_intent(incoming_user.content)
        ):
            workflow_guard = self._workflow_bypass_guard_system_message(task_state["phase"])
        memory_messages = [
            m
            for m in [profile_msg, inv_msg, task_msg, workflow_guard, working_msg, long_term_msg]
            if m is not None
        ]
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
            "active_profile_id": active_profile_id,
            "profile_applied": profile_msg is not None,
            "task_phase": task_state["phase"],
            "task_current_step": task_state["current_step"],
            "task_expected_action": task_state["expected_action"],
            "task_is_paused": task_state["is_paused"],
            "task_allowed_next_phases": list(TASK_ALLOWED_EDGES.get(task_state["phase"], ())),
            "invariants_count": len(self._normalize_invariants(state.get("invariants", {}))),
            "invariants_applied": inv_msg is not None,
            "workflow_guard_applied": workflow_guard is not None,
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
        profile_id: str | None = None,
        resume: bool = False,
    ) -> AsyncIterator[StreamResult]:
        provider = self._validate_provider(provider_name)
        self._validate_model(provider, provider_name, model)
        active_profile_id, profile = self._resolve_profile(profile_id)
        strategy = context_strategy if context_strategy in ALLOWED_STRATEGIES else "sliding"
        normalized_temperature = self._normalize_temperature(temperature)
        incoming = self._normalize_messages(raw_messages)
        if not incoming:
            raise ValueError("No valid messages to send")

        state = self._get_conversation_state(conversation_id)
        if len(incoming) != 1 or incoming[0].role != "user":
            inv_msg = self._invariants_system_message(state.get("invariants", {}))
            profile_msg = self._profile_system_message(profile)
            prefix = [m for m in [profile_msg, inv_msg] if m is not None]
            request_messages = [*prefix, *incoming]
            inv_norm = self._normalize_invariants(state.get("invariants", {}))
            memory_tok = self._estimate_tokens_messages(prefix)
            all_tok = self._estimate_tokens_messages(request_messages)
            context_meta = {
                "context_strategy": strategy,
                "branch_id": branch_id,
                "history_tokens": self._estimate_tokens_messages(incoming),
                "recent_history_tokens": self._estimate_tokens_messages(incoming),
                "memory_tokens": memory_tok,
                "facts_count": 0,
                "prompt_tokens_with_compression_est": all_tok,
                "prompt_tokens_no_compression_est": all_tok,
                "saved_tokens_est": 0,
                "compressed_messages_count": 0,
                "summary_used": False,
                "summary_tokens": 0,
                "compression_enabled": False,
                "current_request_tokens": self._estimate_tokens_messages(incoming),
                "memory_auto_working_keys": [],
                "memory_auto_keys": [],
                "short_term_count": len(incoming),
                "active_profile_id": active_profile_id,
                "profile_applied": profile_msg is not None,
                "task_phase": "planning",
                "task_current_step": "",
                "task_expected_action": "",
                "task_is_paused": False,
                "task_allowed_next_phases": [],
                "invariants_count": len(inv_norm),
                "invariants_applied": inv_msg is not None,
            }
            user_msg = incoming[-1]
        else:
            user_msg = incoming[0]
            current_task_state = self._normalize_task_state(state.get("task_state", {}))
            if current_task_state.get("status") == "paused" and not resume:
                raise ValueError("Task is paused. Resume generation to continue.")
            if resume and current_task_state.get("status") == "paused":
                state["task_state"] = self._transition_task_state(
                    task_state=current_task_state,
                    event=TASK_EVENT_RESUME,
                )
                extra = user_msg.content.strip()
                resume_text = (
                    "Continue from the paused point without repeating previous explanation."
                )
                if extra:
                    resume_text += f"\nAdditional instruction from user:\n{extra}"
                user_msg = Message(role="user", content=resume_text)
            elif self._is_task_intent_message(user_msg.content) and not current_task_state.get(
                "task_active", False
            ):
                # Do not reset FSM on follow-ups that still match markers (e.g. "план" + "реализ" in approval).
                state["task_state"] = self._transition_task_state(
                    task_state=current_task_state,
                    event=TASK_EVENT_NEW_TASK,
                )
            self._promote_to_plan_approved_if_user_approved(state, user_msg.content)
            self._promote_validation_to_done_if_user_confirms(state, user_msg.content)
            current_task_state = self._normalize_task_state(state.get("task_state", {}))
            include_task_state = (
                resume
                or bool(current_task_state.get("task_active", False))
                or self._is_task_intent_message(user_msg.content)
            )
            self._update_facts(state, user_msg.content)
            working_keys = self._auto_update_working_memory(state, user_msg.content)
            auto_keys = self._auto_update_long_term(user_msg.content)
            context_limit = MODEL_CONTEXT_LIMITS.get(model, 128000)
            request_messages, context_meta = self._build_context(
                state=state,
                incoming_user=user_msg,
                profile=profile,
                active_profile_id=active_profile_id,
                include_task_state=include_task_state,
                strategy=strategy,
                branch_id=branch_id,
                context_limit=context_limit,
            )
            context_meta["memory_auto_working_keys"] = working_keys
            context_meta["memory_auto_keys"] = auto_keys

        assistant_chunks: list[str] = []
        provider_meta: dict | None = None
        paused_during_stream = False
        async for result in provider.stream_chat(request_messages, model, normalized_temperature):
            if result.text:
                assistant_chunks.append(result.text)
                yield result
                if self._normalize_task_state(state.get("task_state", {})).get("status") == "paused":
                    paused_during_stream = True
                    break
            if result.meta is not None:
                provider_meta = result.meta

        assistant_text = "".join(assistant_chunks)
        if not assistant_text:
            return

        assistant_msg = Message(role="assistant", content=assistant_text)
        self._append_turn(state, strategy, context_meta["branch_id"], user_msg, assistant_msg)
        current_task_state = self._normalize_task_state(state.get("task_state", {}))
        single_user_turn = len(incoming) == 1 and incoming[0].role == "user"
        auto_advance_phase = (
            single_user_turn
            and not paused_during_stream
            and not resume
            and not self._is_smalltalk_message(user_msg.content)
            and current_task_state.get("task_active", False)
            and current_task_state.get("status") == "running"
        )
        if paused_during_stream:
            task_state_after = self._transition_task_state(
                task_state=current_task_state,
                event=TASK_EVENT_PAUSE,
            )
        else:
            task_state_after = self._transition_task_state(
                task_state=current_task_state,
                event=TASK_EVENT_ASSISTANT_TURN_COMPLETED,
                advance_phase=auto_advance_phase,
            )
        state["task_state"] = task_state_after

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
            "active_profile_id": context_meta.get("active_profile_id", active_profile_id),
            "profile_applied": bool(context_meta.get("profile_applied", False)),
            "task_phase": task_state_after.get("phase", context_meta.get("task_phase", "planning")),
            "task_current_step": task_state_after.get(
                "current_step", context_meta.get("task_current_step", "")
            ),
            "task_expected_action": task_state_after.get(
                "expected_action", context_meta.get("task_expected_action", "")
            ),
            "task_is_paused": bool(task_state_after.get("is_paused", False)),
            "task_allowed_next_phases": list(
                TASK_ALLOWED_EDGES.get(task_state_after.get("phase", "planning"), ())
            ),
            "paused_during_stream": paused_during_stream,
            "memory_short_notes_count": int(context_meta.get("short_term_count", WINDOW_SIZE_MESSAGES)),
            "memory_working_count": len(working_memory),
            "memory_long_count": len(long_term),
            "memory_auto_working_count": len(context_meta.get("memory_auto_working_keys", [])),
            "memory_auto_working_keys": context_meta.get("memory_auto_working_keys", []),
            "memory_auto_saved_count": len(context_meta.get("memory_auto_keys", [])),
            "memory_auto_saved_keys": context_meta.get("memory_auto_keys", []),
            "compression_enabled": False,
            "estimated": prompt_tokens == 0,
            "invariants_count": int(context_meta.get("invariants_count", 0)),
            "invariants_applied": bool(context_meta.get("invariants_applied", False)),
            "workflow_guard_applied": bool(context_meta.get("workflow_guard_applied", False)),
        }
        yield StreamResult(meta=enriched_meta)
