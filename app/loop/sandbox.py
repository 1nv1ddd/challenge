"""Песочница цикла: код из ответа модели, проверка синтаксиса и прогон тестов в отдельном процессе."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from ..agent_constants import (
    LOOP_MAX_CODE_CHARS,
    LOOP_MAX_OUTPUT_CHARS,
    LOOP_SOLUTION_MODULE,
    LOOP_TEST_TIMEOUT_SEC,
    LOOP_TESTS_MODULE,
)
from .schema import CheckResult

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(?P<code>.*?)```", re.DOTALL | re.IGNORECASE)
# Признаки того, что ответ и без ограждения — код, а не проза с извинениями.
_CODE_HINT_RE = re.compile(r"^\s*(?:import |from |def |class |#!)", re.MULTILINE)
# Переменные окружения, которые в песочницу не уходят: код пишет модель, ключи ей не нужны.
_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT")


def extract_code(raw: str) -> str:
    """Модуль из ответа модели: содержимое первого ```python-блока или весь ответ, если он код."""
    blocks = [m.group("code") for m in _FENCE_RE.finditer(raw or "")]
    if blocks:
        code = max(blocks, key=len)
    elif _CODE_HINT_RE.search(raw or ""):
        code = raw or ""
    else:
        return ""
    return code.strip()[:LOOP_MAX_CODE_CHARS]


def _trim(text: str) -> str:
    body = (text or "").strip()
    if len(body) <= LOOP_MAX_OUTPUT_CHARS:
        return body
    return f"{body[:LOOP_MAX_OUTPUT_CHARS]}… (вывод обрезан)"


def check_syntax(code: str) -> CheckResult:
    """Синтаксис модуля. Компиляция не исполняет код — это безопасно делать в своём процессе."""
    if not (code or "").strip():
        return CheckResult(stage="syntax", ok=False, output="модуль пуст: в ответе не было кода")
    try:
        compile(code, LOOP_SOLUTION_MODULE, "exec")
    except SyntaxError as exc:
        return CheckResult(
            stage="syntax",
            ok=False,
            output=f"SyntaxError: {exc.msg} (строка {exc.lineno})",
        )
    return CheckResult(stage="syntax", ok=True, output="синтаксис в порядке")


def _sandbox_env(home: Path) -> dict[str, str]:
    """Окружение подпроцесса: ни ключей провайдера, ни прочих секретов проекта."""
    env = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    env["HOME"] = str(home)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_tests(code: str, tests: str, timeout: int = LOOP_TEST_TIMEOUT_SEC) -> CheckResult:
    """Тесты задачи против сгенерированного модуля во временной директории.

    Здесь исполняется код, написанный моделью. Полной изоляции без контейнера не получится, поэтому
    ограничиваем то, что дёшево и реально помогает: своя директория, отдельный процесс, жёсткий
    таймаут и окружение без ключей проекта.
    """
    with tempfile.TemporaryDirectory(prefix="loop-sandbox-") as tmp:
        workdir = Path(tmp)
        (workdir / LOOP_SOLUTION_MODULE).write_text(code, encoding="utf-8")
        (workdir / LOOP_TESTS_MODULE).write_text(tests, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "unittest", "-v", Path(LOOP_TESTS_MODULE).stem],
                cwd=workdir,
                env=_sandbox_env(workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CheckResult(
                stage="tests",
                ok=False,
                output=f"тесты не уложились в {timeout} с — вероятно, бесконечный цикл или сеть",
            )
        except OSError as exc:
            return CheckResult(stage="tests", ok=False, output=f"не удалось запустить тесты: {exc}")
    output = _trim(f"{proc.stdout}\n{proc.stderr}")
    return CheckResult(stage="tests", ok=proc.returncode == 0, output=output)


def run_checks(code: str, tests: str, timeout: int = LOOP_TEST_TIMEOUT_SEC) -> list[CheckResult]:
    """Синтаксис, а следом тесты. Упавший синтаксис тесты не запускает: смысла нет."""
    syntax = check_syntax(code)
    if not syntax.ok:
        return [syntax]
    return [syntax, run_tests(code, tests, timeout)]


def checks_feedback(checks: list[CheckResult]) -> str:
    """Фидбек генератору по упавшим проверкам — с выводом, а не с «тесты не прошли»."""
    failed = [c for c in checks if not c.ok]
    if not failed:
        return ""
    parts = [f"Этап `{c.stage}` не прошёл:\n\n```\n{c.output}\n```" for c in failed]
    return "\n\n".join(parts)
