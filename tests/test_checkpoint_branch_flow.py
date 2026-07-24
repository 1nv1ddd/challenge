"""HTTP happy-path: POST /api/checkpoints → GET /api/branches, консистентность связки."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent import SimpleChatAgent  # noqa: E402
from app.main import app  # noqa: E402
from app.routers import hub  # noqa: E402


class TestCheckpointBranchFlow(unittest.TestCase):
    """Связка checkpoint→branch через HTTP: чекпойнт создаётся, ветка от него читается консистентно."""

    def setUp(self) -> None:
        # Изолируем память агента во временный файл: реальный data/agent_memory.json не трогаем.
        self._dir = tempfile.mkdtemp()
        self._mem_path = Path(self._dir) / "agent_memory.json"
        self._orig_agent = hub.agent
        hub.agent = SimpleChatAgent({}, memory_path=self._mem_path)

    def tearDown(self) -> None:
        hub.agent = self._orig_agent
        shutil.rmtree(self._dir, ignore_errors=True)

    def _seed_main_messages(self, conversation_id: str, messages: list[dict]) -> None:
        state = hub.agent._get_conversation_state(conversation_id)
        state["branches"]["main"]["messages"] = messages

    def test_checkpoint_then_branch_is_consistent(self) -> None:
        conv = "t11-flow"
        self._seed_main_messages(
            conv,
            [
                {"role": "user", "content": "первый вопрос"},
                {"role": "assistant", "content": "первый ответ"},
            ],
        )
        with TestClient(app) as client:
            r_cp = client.post("/api/checkpoints", json={"conversation_id": conv})
            self.assertEqual(r_cp.status_code, 200, r_cp.text)
            checkpoint_id = r_cp.json().get("checkpoint_id")
            self.assertIsInstance(checkpoint_id, str)
            self.assertTrue(checkpoint_id.startswith("cp_"), checkpoint_id)

            # Ветки читаются: main присутствует и хранит засеянные сообщения.
            r_b0 = client.get("/api/branches", params={"conversation_id": conv})
            self.assertEqual(r_b0.status_code, 200, r_b0.text)
            branches0 = {b["id"]: b for b in r_b0.json()["branches"]}
            self.assertIn("main", branches0)
            self.assertEqual(branches0["main"]["message_count"], 2)

            # Ветвление ОТ созданного чекпойнта.
            r_new = client.post(
                "/api/branches",
                json={"conversation_id": conv, "checkpoint_id": checkpoint_id},
            )
            self.assertEqual(r_new.status_code, 200, r_new.text)
            new_branch_id = r_new.json().get("branch_id")
            self.assertTrue(new_branch_id)
            self.assertNotEqual(new_branch_id, "main")

            # Список веток консистентен: новая ветка ссылается на чекпойнт и унаследовала сообщения.
            r_b1 = client.get("/api/branches", params={"conversation_id": conv})
            self.assertEqual(r_b1.status_code, 200, r_b1.text)
            branches1 = {b["id"]: b for b in r_b1.json()["branches"]}
            self.assertIn("main", branches1)
            self.assertIn(new_branch_id, branches1)
            new_branch = branches1[new_branch_id]
            self.assertEqual(new_branch["from_checkpoint"], checkpoint_id)
            self.assertEqual(new_branch["message_count"], 2)

    def test_checkpoint_persisted_to_isolated_file_only(self) -> None:
        conv = "t11-persist"
        with TestClient(app) as client:
            r_cp = client.post("/api/checkpoints", json={"conversation_id": conv})
            self.assertEqual(r_cp.status_code, 200, r_cp.text)
            checkpoint_id = r_cp.json()["checkpoint_id"]

        # Запись ушла в изолированный файл, а не в реальный data/agent_memory.json.
        self.assertTrue(self._mem_path.exists())
        saved = self._mem_path.read_text(encoding="utf-8")
        self.assertIn(checkpoint_id, saved)
        self.assertIn(conv, saved)


if __name__ == "__main__":
    unittest.main()
