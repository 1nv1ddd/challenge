"""Тесты для списка задач планировщика.

Проверяют:
  1. функцию хранилища `app.scheduler_store.list_scheduler_jobs()` —
     что она возвращает зарегистрированные задачи с полями
     task_id, task_type, next_run, interval_seconds и сортировкой по next_run;
  2. эндпоинт `GET /api/scheduler/jobs` — что он отдаёт те же задачи в JSON.

Предназначен для вставки в `tests/` (например, tests/test_scheduler_jobs.py).
Паттерн изоляции БД (временный SQLite + reload модуля) взят из
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

    def test_list_scheduler_jobs_empty(self) -> None:
        self.assertEqual(self.ss.list_scheduler_jobs(), [])

    def test_list_scheduler_jobs_fields_and_order(self) -> None:
        # Регистрируем две задачи с разным first_run_in_seconds,
        # чтобы проверить сортировку по next_run (ближайшая — первой).
        self.ss.register_job(
            task_id="later",
            interval_seconds=300,
            task_type="heartbeat_rollup",
            payload="p-later",
            first_run_in_seconds=3600,
        )
        self.ss.register_job(
            task_id="sooner",
            interval_seconds=120,
            task_type="reminder",
            payload="p-sooner",
            first_run_in_seconds=5,
        )

        jobs = self.ss.list_scheduler_jobs()
        self.assertEqual(len(jobs), 2)

        # Порядок: "sooner" раньше "later".
        self.assertEqual(jobs[0]["task_id"], "sooner")
        self.assertEqual(jobs[1]["task_id"], "later")

        # Ровно нужный набор ключей.
        self.assertEqual(
            set(jobs[0].keys()),
            {"task_id", "task_type", "next_run", "interval_seconds"},
        )

        first = jobs[0]
        self.assertEqual(first["task_type"], "reminder")
        self.assertEqual(first["interval_seconds"], 120)
        self.assertIsInstance(first["interval_seconds"], int)
        self.assertIsInstance(first["next_run"], float)
        self.assertLessEqual(jobs[0]["next_run"], jobs[1]["next_run"])

    def test_endpoint_returns_jobs(self) -> None:
        from fastapi.testclient import TestClient

        # Модуль роутов держит ссылку на scheduler_store, перезагружаем и его,
        # чтобы он видел свежесозданную (временную) БД.
        import app.scheduler_routes as sr

        importlib.reload(sr)
        import app.main as main

        importlib.reload(main)

        self.ss.register_job(
            task_id="job-api",
            interval_seconds=60,
            task_type="heartbeat_rollup",
            payload="via-api",
            first_run_in_seconds=10,
        )

        with TestClient(main.app) as client:
            resp = client.get("/api/scheduler/jobs")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["count"], 1)

        job = data["jobs"][0]
        self.assertEqual(job["task_id"], "job-api")
        self.assertEqual(job["task_type"], "heartbeat_rollup")
        self.assertEqual(job["interval_seconds"], 60)
        self.assertIn("next_run", job)


if __name__ == "__main__":
    unittest.main()
