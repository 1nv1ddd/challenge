"""API веток: создание без checkpoint_id должно отвечать ошибкой, а не «тихим» успехом."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent import SimpleChatAgent  # noqa: E402
from app.main import app  # noqa: E402

_CONV = "test-branch-api"


class TestCreateBranchApi(unittest.TestCase):
    def setUp(self) -> None:
        # Роут дёргает глобальный agent из bootstrap; подменяем его агентом на временном файле,
        # чтобы тест не писал в рабочий data/agent_memory.json.
        self._tmp = tempfile.TemporaryDirectory()
        self.agent = SimpleChatAgent({}, memory_path=Path(self._tmp.name) / "agent_memory.json")
        self._patcher = patch("app.routers.hub.agent", self.agent)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmp.cleanup()

    def _branch_ids(self, client: TestClient, conversation_id: str = _CONV) -> list[str]:
        r = client.get("/api/branches", params={"conversation_id": conversation_id})
        self.assertEqual(r.status_code, 200, r.text)
        return [b["id"] for b in r.json()["branches"]]

    def test_missing_checkpoint_id_returns_400(self) -> None:
        with TestClient(app) as client:
            r = client.post(
                "/api/branches",
                json={"conversation_id": _CONV, "branch_name": "Branch A"},
            )
            branches_after = self._branch_ids(client)
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("checkpoint_id", r.json()["detail"])
        # Отказ должен быть без побочного эффекта: ветка не создана.
        self.assertEqual(branches_after, ["main"])

    def test_blank_checkpoint_id_returns_400(self) -> None:
        with TestClient(app) as client:
            r = client.post(
                "/api/branches",
                json={"conversation_id": _CONV, "checkpoint_id": "", "branch_name": "Branch A"},
            )
            branches_after = self._branch_ids(client)
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(branches_after, ["main"])

    def test_checkpoint_then_branch_creates_branch(self) -> None:
        with TestClient(app) as client:
            cp = client.post("/api/checkpoints", json={"conversation_id": _CONV, "branch_id": "main"})
            self.assertEqual(cp.status_code, 200, cp.text)
            checkpoint_id = cp.json()["checkpoint_id"]

            r = client.post(
                "/api/branches",
                json={
                    "conversation_id": _CONV,
                    "checkpoint_id": checkpoint_id,
                    "branch_name": "Branch A",
                },
            )
            self.assertEqual(r.status_code, 200, r.text)
            created = r.json()
            self.assertEqual(created["branch_id"], "branch-1")
            self.assertEqual(created["name"], "Branch A")

            listed = client.get("/api/branches", params={"conversation_id": _CONV}).json()["branches"]
        by_id = {b["id"]: b for b in listed}
        self.assertEqual(sorted(by_id), ["branch-1", "main"])
        self.assertEqual(by_id["branch-1"]["from_checkpoint"], checkpoint_id)

    def test_branch_is_isolated_per_conversation(self) -> None:
        with TestClient(app) as client:
            cp = client.post("/api/checkpoints", json={"conversation_id": _CONV})
            client.post(
                "/api/branches",
                json={"conversation_id": _CONV, "checkpoint_id": cp.json()["checkpoint_id"]},
            )
            other = self._branch_ids(client, "test-branch-api-other")
        self.assertEqual(other, ["main"])


if __name__ == "__main__":
    unittest.main()
