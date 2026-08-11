"""Тесты для list_scheduler_jobs() и эндпоинта GET /api/scheduler/jobs.

Изолируем БД через временный файл + переменную окружения
SCHEDULER_SQLITE_PATH и reload модуля — так же, как в
tests/test_scheduler_store.py.
"""

from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path


class TestSchedulerJobs(unittest.TestCase):
    def setUp(self) -> None:
        self._fd, self._path = tempfile.mkstemp(suffix=".sqlite")
        os.close(self._fd)
        os.environ["SCHEDULER_SQLITE_PATH"] = self._path
        import app.scheduler_store as ss

        importlib.reload(ss)
        self.ss = ss

    def tearDown(self) -> None:
        os.environ.pop("SCHEDULER_SQLITE_PATH", None)
        import app.scheduler_store as ss

        importlib.reload(ss)
        Path(self._path).unlink(missing_ok=True)

    def test_empty_by_default(self) -> None:
        self.assertEqual(self.ss.list_scheduler_jobs(), [])

    def test_lists_registered_job_fields(self) -> None:
        self.ss.register_job(
            task_id="t1",
            interval_seconds=120,
            task_type="heartbeat_rollup",
            payload="p1",
            first_run_in_seconds=10,
        )
        jobs = self.ss.list_scheduler_jobs()
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        # Ровно нужный набор полей, ничего лишнего (payload/created_at не отдаём).
        self.assertEqual(
            set(job.keys()),
            {"task_id", "task_type", "next_run", "interval_seconds"},
        )
        self.assertEqual(job["task_id"], "t1")
        self.assertEqual(job["task_type"], "heartbeat_rollup")
        self.assertEqual(job["interval_seconds"], 120)
        self.assertIsInstance(job["next_run"], float)
        self.assertGreater(job["next_run"], 0)

    def test_sorted_by_next_run(self) -> None:
        # far — позже (next_run дальше), near — раньше.
        self.ss.register_job(
            task_id="far",
            interval_seconds=120,
            task_type="reminder",
            payload="",
            first_run_in_seconds=3600,
        )
        self.ss.register_job(
            task_id="near",
            interval_seconds=120,
            task_type="reminder",
            payload="",
            first_run_in_seconds=5,
        )
        ids = [j["task_id"] for j in self.ss.list_scheduler_jobs()]
        self.assertEqual(ids, ["near", "far"])


class TestSchedulerJobsEndpoint(unittest.TestCase):
    """Проверка HTTP-эндпоинта GET /api/scheduler/jobs через FastAPI TestClient."""

    def setUp(self) -> None:
        self._fd, self._path = tempfile.mkstemp(suffix=".sqlite")
        os.close(self._fd)
        os.environ["SCHEDULER_SQLITE_PATH"] = self._path
        import app.scheduler_store as ss

        importlib.reload(ss)
        self.ss = ss

    def tearDown(self) -> None:
        os.environ.pop("SCHEDULER_SQLITE_PATH", None)
        import app.scheduler_store as ss

        importlib.reload(ss)
        Path(self._path).unlink(missing_ok=True)

    def test_endpoint_returns_registered_jobs(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import app.scheduler_routes as routes

        importlib.reload(routes)

        self.ss.register_job(
            task_id="t1",
            interval_seconds=120,
            task_type="heartbeat_rollup",
            payload="p1",
            first_run_in_seconds=10,
        )

        app_obj = FastAPI()
        app_obj.include_router(routes.router)
        client = TestClient(app_obj)

        resp = client.get("/api/scheduler/jobs")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["count"], 1)
        self.assertEqual(len(data["jobs"]), 1)
        job = data["jobs"][0]
        self.assertEqual(
            set(job.keys()),
            {"task_id", "task_type", "next_run", "interval_seconds"},
        )
        self.assertEqual(job["task_id"], "t1")
        self.assertEqual(job["task_type"], "heartbeat_rollup")
        self.assertEqual(job["interval_seconds"], 120)


if __name__ == "__main__":
    unittest.main()
