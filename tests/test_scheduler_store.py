"""Периодический планировщик (SQLite), без MCP-процесса."""

from __future__ import annotations

import importlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path


class TestSchedulerStore(unittest.TestCase):
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

    def test_register_and_process_heartbeat(self) -> None:
        self.ss.register_job(
            task_id="t1",
            interval_seconds=120,
            task_type="heartbeat_rollup",
            payload="p1",
            first_run_in_seconds=10,
        )
        with sqlite3.connect(self._path) as conn:
            conn.execute("UPDATE jobs SET next_run = 0 WHERE task_id = ?", ("t1",))
            conn.commit()
        n = self.ss.process_due_jobs()
        self.assertEqual(n, 1)
        agg = self.ss.get_aggregated_results("t1")
        self.assertEqual(agg["total_stored_runs"], 1)
        self.assertGreater(len(agg["summary_text"]), 3)
