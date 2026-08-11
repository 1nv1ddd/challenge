"""FSM задачи: легальные переходы, отказ на скачках и терминальном done, pause/resume."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent import SimpleChatAgent  # noqa: E402
from app.agent_constants import (  # noqa: E402
    TASK_EVENT_ASSISTANT_TURN_COMPLETED,
    TASK_EVENT_NEW_TASK,
    TASK_PHASE_TO_DEFAULTS,
)

_CONV = "test-task-fsm"


class TestTaskFsmTransitions(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.memory_path = Path(self._tmp.name) / "agent_memory.json"
        self.agent = SimpleChatAgent({}, memory_path=self.memory_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _advance_to(self, phase: str) -> None:
        """Довести задачу до нужной фазы легальными шагами 'next'."""
        for _ in range(5):
            out = self.agent.list_task_state(_CONV)
            if out["phase"] == phase:
                return
            step = self.agent.update_task_state(_CONV, action="next")
            self.assertTrue(step["ok"], step)
        self.fail(f"не удалось дойти до фазы {phase!r}")

    def test_initial_phase_is_planning_and_inactive(self) -> None:
        out = self.agent.list_task_state(_CONV)
        self.assertEqual(out["phase"], "planning")
        self.assertFalse(out["task_active"])
        self.assertEqual(out["allowed_next_phases"], ["plan_approved"])

    def test_next_advances_one_phase_and_applies_defaults(self) -> None:
        out = self.agent.update_task_state(_CONV, action="next")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["phase"], "plan_approved")
        self.assertTrue(out["task_active"])
        self.assertEqual(out["current_step"], TASK_PHASE_TO_DEFAULTS["plan_approved"]["current_step"])
        self.assertEqual(
            out["expected_action"],
            TASK_PHASE_TO_DEFAULTS["plan_approved"]["expected_action"],
        )

    def test_phase_skip_is_rejected_and_state_unchanged(self) -> None:
        out = self.agent.update_task_state(_CONV, phase="execution")
        self.assertFalse(out["ok"], out)
        self.assertIn("Illegal transition", out["error"])
        self.assertIn("plan_approved", out["error"])
        self.assertEqual(out["phase"], "planning")
        self.assertEqual(self.agent.list_task_state(_CONV)["phase"], "planning")

    def test_rejected_transition_is_not_persisted(self) -> None:
        self.agent.update_task_state(_CONV, action="next")
        self.agent.update_task_state(_CONV, phase="validation")
        reloaded = SimpleChatAgent({}, memory_path=self.memory_path)
        self.assertEqual(reloaded.list_task_state(_CONV)["phase"], "plan_approved")

    def test_next_from_done_is_rejected(self) -> None:
        self._advance_to("done")
        out = self.agent.update_task_state(_CONV, action="next")
        self.assertFalse(out["ok"], out)
        self.assertIn("done", out["error"])
        self.assertEqual(out["phase"], "done")
        self.assertEqual(out["allowed_next_phases"], [])

    def test_leaving_done_by_phase_hints_reset(self) -> None:
        self._advance_to("done")
        out = self.agent.update_task_state(_CONV, phase="execution")
        self.assertFalse(out["ok"], out)
        self.assertIn("terminal", out["error"])
        self.assertIn("reset", out["error"])

    def test_reset_returns_to_planning_and_deactivates(self) -> None:
        self._advance_to("execution")
        out = self.agent.update_task_state(_CONV, action="reset")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["phase"], "planning")
        self.assertFalse(out["task_active"])

    def test_unknown_phase_is_rejected(self) -> None:
        out = self.agent.update_task_state(_CONV, phase="deployment")
        self.assertFalse(out["ok"], out)
        self.assertIn("Unknown phase", out["error"])

    def test_unknown_action_is_rejected(self) -> None:
        out = self.agent.update_task_state(_CONV, action="teleport")
        self.assertFalse(out["ok"], out)
        self.assertIn("Unknown action", out["error"])
        self.assertEqual(self.agent.list_task_state(_CONV)["phase"], "planning")

    def test_pause_then_resume_keeps_phase(self) -> None:
        self._advance_to("execution")
        paused = self.agent.update_task_state(_CONV, action="pause")
        self.assertTrue(paused["ok"], paused)
        self.assertEqual(paused["status"], "paused")
        self.assertTrue(paused["is_paused"])
        self.assertEqual(paused["phase"], "execution")

        resumed = self.agent.update_task_state(_CONV, action="resume")
        self.assertEqual(resumed["status"], "running")
        self.assertFalse(resumed["is_paused"])
        self.assertEqual(resumed["phase"], "execution")
        self.assertEqual(
            resumed["expected_action"],
            TASK_PHASE_TO_DEFAULTS["execution"]["expected_action"],
        )

    def test_custom_step_overrides_default_but_blank_does_not(self) -> None:
        out = self.agent.update_task_state(_CONV, action="next", current_step="Свой шаг")
        self.assertEqual(out["current_step"], "Свой шаг")
        blank = self.agent.update_task_state(_CONV, current_step="   ")
        self.assertEqual(blank["current_step"], "Свой шаг")

    def test_custom_step_is_truncated(self) -> None:
        out = self.agent.update_task_state(_CONV, current_step="ш" * 400)
        self.assertEqual(len(out["current_step"]), 220)


class TestTaskFsmAutoAdvance(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.agent = SimpleChatAgent({}, memory_path=Path(self._tmp.name) / "agent_memory.json")
        self.active = self.agent._transition_task_state({}, TASK_EVENT_NEW_TASK)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _turn(self, task_state: dict, *, advance_phase: bool = True) -> dict:
        return self.agent._transition_task_state(
            task_state,
            TASK_EVENT_ASSISTANT_TURN_COMPLETED,
            advance_phase=advance_phase,
        )

    def test_new_task_starts_active_planning(self) -> None:
        self.assertEqual(self.active["phase"], "planning")
        self.assertTrue(self.active["task_active"])
        self.assertEqual(self.active["status"], "running")

    def test_turn_does_not_auto_approve_plan(self) -> None:
        self.assertEqual(self._turn(self.active)["phase"], "planning")

    def test_turn_advances_plan_approved_to_execution(self) -> None:
        approved = {**self.active, "phase": "plan_approved"}
        self.assertEqual(self._turn(approved)["phase"], "execution")

    def test_turn_does_not_auto_close_task_from_validation(self) -> None:
        validation = {**self.active, "phase": "validation"}
        self.assertEqual(self._turn(validation)["phase"], "validation")

    def test_paused_turn_does_not_advance(self) -> None:
        paused = {**self.active, "phase": "plan_approved", "status": "paused"}
        self.assertEqual(self._turn(paused)["phase"], "plan_approved")

    def test_advance_phase_false_suppresses_step(self) -> None:
        approved = {**self.active, "phase": "plan_approved"}
        self.assertEqual(self._turn(approved, advance_phase=False)["phase"], "plan_approved")

    def test_inactive_task_does_not_advance(self) -> None:
        idle = {**self.active, "phase": "plan_approved", "task_active": False}
        self.assertEqual(self._turn(idle)["phase"], "plan_approved")


class TestTaskFsmUserIntentPromotion(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.agent = SimpleChatAgent({}, memory_path=Path(self._tmp.name) / "agent_memory.json")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _state(self, phase: str) -> dict:
        base = self.agent._transition_task_state({}, TASK_EVENT_NEW_TASK)
        return {"task_state": self.agent._normalize_task_state({**base, "phase": phase})}

    def test_explicit_approval_promotes_to_plan_approved(self) -> None:
        state = self._state("planning")
        self.agent._promote_to_plan_approved_if_user_approved(state, "Утверждаю план, начинаем")
        self.assertEqual(state["task_state"]["phase"], "plan_approved")

    def test_rejection_wording_does_not_promote(self) -> None:
        state = self._state("planning")
        self.agent._promote_to_plan_approved_if_user_approved(state, "Не утверждаю план, переделай")
        self.assertEqual(state["task_state"]["phase"], "planning")

    def test_short_ack_does_not_promote(self) -> None:
        state = self._state("planning")
        self.agent._promote_to_plan_approved_if_user_approved(state, "ок")
        self.assertEqual(state["task_state"]["phase"], "planning")

    def test_approval_outside_planning_is_ignored(self) -> None:
        state = self._state("execution")
        self.agent._promote_to_plan_approved_if_user_approved(state, "Утверждаю план, начинаем")
        self.assertEqual(state["task_state"]["phase"], "execution")

    def test_explicit_completion_promotes_validation_to_done(self) -> None:
        state = self._state("validation")
        self.agent._promote_validation_to_done_if_user_confirms(state, "Закрываем задачу, всё готово")
        self.assertEqual(state["task_state"]["phase"], "done")

    def test_negated_completion_keeps_validation(self) -> None:
        state = self._state("validation")
        self.agent._promote_validation_to_done_if_user_confirms(state, "Задача не завершена, продолжаем")
        self.assertEqual(state["task_state"]["phase"], "validation")

    def test_plain_continue_does_not_close_task(self) -> None:
        state = self._state("validation")
        self.agent._promote_validation_to_done_if_user_confirms(state, "продолжаем работу дальше")
        self.assertEqual(state["task_state"]["phase"], "validation")


if __name__ == "__main__":
    unittest.main()
