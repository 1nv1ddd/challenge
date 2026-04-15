from __future__ import annotations

import time

from ..agent_constants import (
    TASK_ALLOWED_EDGES,
    TASK_EVENT_ASSISTANT_TURN_COMPLETED,
    TASK_EVENT_NEW_TASK,
    TASK_EVENT_PAUSE,
    TASK_EVENT_RESUME,
    TASK_PHASE_TO_DEFAULTS,
    TASK_PHASES,
)


class AgentTaskFsmMixin:
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
