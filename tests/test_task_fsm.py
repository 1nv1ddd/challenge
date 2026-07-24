"""FSM задачи: нелегальный переход с пропуском фазы (planning -> execution)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.core import SimpleChatAgent  # noqa: E402
from app.agent_constants import TASK_EVENT_NEW_TASK  # noqa: E402


class TestTaskFsmIllegalTransition(unittest.TestCase):
    """update_task_state должен запрещать skip-переход planning -> execution."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.agent = SimpleChatAgent(
            providers={},
            memory_path=Path(self._dir.name) / "agent_memory.json",
        )
        self.conversation_id = "conv-fsm"
        # Поднимаем реальную активную задачу в фазе planning.
        state = self.agent._get_conversation_state(self.conversation_id)
        state["task_state"] = self.agent._transition_task_state(
            state["task_state"], TASK_EVENT_NEW_TASK
        )

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_planning_to_execution_skips_phase_returns_not_ok(self) -> None:
        # execution достижим только через plan_approved — прямой прыжок недопустим.
        result = self.agent.update_task_state(self.conversation_id, phase="execution")

        self.assertFalse(result["ok"])
        self.assertIn("Illegal transition", result["error"])
        # Фаза не должна сдвинуться при отклонённом переходе.
        self.assertEqual(result["phase"], "planning")

    def test_rejected_transition_does_not_mutate_stored_state(self) -> None:
        # Отклонённый переход не должен просачиваться в сохранённое состояние.
        self.agent.update_task_state(self.conversation_id, phase="execution")

        stored = self.agent.list_task_state(self.conversation_id)
        self.assertEqual(stored["phase"], "planning")

    def test_planning_to_plan_approved_is_allowed(self) -> None:
        # Контроль: разрешённый переход по ребру всё ещё проходит (ok == True).
        result = self.agent.update_task_state(
            self.conversation_id, phase="plan_approved"
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["phase"], "plan_approved")


if __name__ == "__main__":
    unittest.main()
