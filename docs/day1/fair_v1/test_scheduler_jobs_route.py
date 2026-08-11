# ВСТАВИТЬ В: tests/test_scheduler_jobs_route.py  (новый файл теста)
#
# Что проверяет: эндпоинт `GET /api/scheduler/jobs` возвращает
# зарегистрированные задачи с полями task_id, task_type, next_run,
# interval_seconds, взятыми из слоя хранилища планировщика.
#
# Тест изолирован: работает поверх временной SQLite-БД через переменную
# окружения SCHEDULER_SQLITE_PATH (тот же приём, что и в
# tests/test_scheduler_store.py). Роутер зовёт list_jobs() в момент запроса,
# а _db_file() читает env при каждом обращении — поэтому app.main можно
# импортировать после установки env без reload.

from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


class TestSchedulerJobsRoute(unittest.TestCase):
    def setUp(self) -> None:
        self._fd, self._path = tempfile.mkstemp(suffix=".sqlite")
        os.close(self._fd)
        os.environ["SCHEDULER_SQLITE_PATH"] = self._path

        import app.scheduler_store as ss

        importlib.reload(ss)
        self.ss = ss

        import app.main as main

        importlib.reload(main)
        self.client = TestClient(main.app)

    def tearDown(self) -> None:
        os.environ.pop("SCHEDULER_SQLITE_PATH", None)
        import app.scheduler_store as ss

        importlib.reload(ss)
        Path(self._path).unlink(missing_ok=True)

    def test_jobs_empty(self) -> None:
        resp = self.client.get("/api/scheduler/jobs")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["count"], 0)
        self.assertEqual(body["jobs"], [])

    def test_jobs_lists_registered_task(self) -> None:
        reg = self.ss.register_job(
            task_id="job1",
            interval_seconds=120,
            task_type="heartbeat_rollup",
            payload="p1",
            first_run_in_seconds=30,
        )

        resp = self.client.get("/api/scheduler/jobs")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()

        self.assertEqual(body["count"], 1)
        job = body["jobs"][0]

        # ровно публичный контракт из четырёх полей
        self.assertEqual(set(job.keys()), {"task_id", "task_type", "next_run", "interval_seconds"})
        self.assertEqual(job["task_id"], "job1")
        self.assertEqual(job["task_type"], "heartbeat_rollup")
        self.assertEqual(job["interval_seconds"], 120)
        # next_run совпадает с тем, что записал слой хранилища
        self.assertAlmostEqual(job["next_run"], reg["next_run_epoch"], places=3)

        # payload/last_run/created_at наружу не утекают
        self.assertNotIn("payload", job)


if __name__ == "__main__":
    unittest.main()
