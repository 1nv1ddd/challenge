"""Сборка датасета для файнтюна: курируемый источник → raw/train/eval JSONL."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Фиксированный system-промпт: закрепляет конвенции проекта из CLAUDE.md.
# Он одинаков во всех примерах — модель учится и содержанию, и формату/стилю.
SYSTEM = (
    "Ты — код-ассистент проекта «AI Chat Hub» (FastAPI, Python 3.12). "
    "Пиши код строго в стиле проекта: первой строкой модуля после docstring — "
    "`from __future__ import annotations`; в начале модуля однострочный русский "
    "docstring; современные type hints (str | None, list[dict], AsyncIterator[...]); "
    "DTO — через @dataclass; интерфейсы — через ABC + @abstractmethod; FastAPI-роуты "
    "через APIRouter, ошибки — HTTPException со status_code и русским detail; доступ к "
    "SQLite планировщика — только через функции app/scheduler_store.py и только "
    "параметризованным SQL; snake_case для функций и переменных, PascalCase для классов, "
    "приватные имена — с префиксом `_`, константы модуля — UPPER_SNAKE. "
    "Исключения лови точечно и пробрасывай причину через `from exc`. "
    "Запрещено: голый except / except Exception без нужды, print() в коде приложения, "
    "Any в аннотациях, дублирование существующих хелперов, f-string/конкатенация в SQL. "
    "Сначала переиспользуй существующие функции модуля, потом пиши новые. "
    "Выдавай только код, без пояснений вокруг."
)

# Каждый пример: (source, user, assistant). source ∈ {"real", "synthetic"}.
# "real" — извлечено (иногда слегка сокращено) из реальных модулей проекта.
EXAMPLES: list[tuple[str, str, str]] = [
    # ---------------------------------------------------------------- REAL ---
    (
        "real",
        "Заведи DTO для сообщения чата и результата стрима, плюс абстрактный "
        "интерфейс провайдера LLM со стриминговым методом.",
        r'''"""Провайдеры LLM: интерфейс и DTO для стрима."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class StreamResult:
    text: str | None = None
    meta: dict | None = None


class AIProvider(ABC):
    name: str
    models: list[dict]

    @abstractmethod
    async def stream_chat(
        self, messages: list[Message], model: str, temperature: float = 0.7
    ) -> AsyncIterator[StreamResult]:
        yield StreamResult()''',
    ),
    (
        "real",
        "Нужен payload-класс запроса чата: разбирает JSON-тело через from_body, "
        "без ручного разбора dict в обработчике.",
        r'''"""Разбор JSON-тел HTTP-запросов для API (тонкий слой без FastAPI)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatRequestPayload:
    provider_name: str
    model: str
    conversation_id: str
    raw_messages: list[dict]
    temperature: float
    branch_id: str
    resume: bool

    @classmethod
    def from_body(cls, body: dict[str, Any]) -> ChatRequestPayload:
        raw_msgs = body.get("messages", [])
        if not isinstance(raw_msgs, list):
            raw_msgs = []
        return cls(
            provider_name=str(body.get("provider", "")),
            model=str(body.get("model", "")),
            conversation_id=str(body.get("conversation_id", "default")),
            raw_messages=raw_msgs,
            temperature=float(body.get("temperature", 0.7)),
            branch_id=str(body.get("branch_id", "main")),
            resume=bool(body.get("resume", False)),
        )''',
    ),
    (
        "real",
        "Собери одну строку SSE с префиксом [ERROR] из исключения для стрима чата.",
        r'''def sse_error_line(exc: BaseException) -> str:
    """Одна строка SSE с префиксом [ERROR] для стрима чата."""
    if isinstance(exc, (LookupError, ValueError)):
        msg = str(exc).strip() or type(exc).__name__
    else:
        msg = str(exc).replace("\n", " ").strip() or type(exc).__name__
    return f"data: [ERROR] {msg}\n\n"''',
    ),
    (
        "real",
        "Слой хранилища планировщика: коннект к SQLite с row_factory и "
        "идемпотентная инициализация схемы.",
        r'''"""SQLite-хранилище для периодических MCP-задач и их результатов."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def _db_file() -> Path:
    raw = (os.environ.get("SCHEDULER_SQLITE_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "data" / "mcp_scheduler.sqlite"


def _connect() -> sqlite3.Connection:
    path = _db_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              task_id TEXT PRIMARY KEY,
              interval_seconds INTEGER NOT NULL,
              task_type TEXT NOT NULL,
              payload TEXT,
              next_run REAL NOT NULL,
              last_run REAL,
              created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_next ON jobs(next_run);
            """
        )
        conn.commit()''',
    ),
    (
        "real",
        "Верни список всех задач планировщика из хранилища.",
        r'''def list_jobs() -> list[dict]:
    init_schema()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT task_id, interval_seconds, task_type, payload, next_run, "
            "last_run, created_at FROM jobs",
        ).fetchall()
    return [dict(r) for r in rows]''',
    ),
    (
        "real",
        "Удали задачу планировщика вместе с её результатами по task_id, с "
        "валидацией идентификатора.",
        r'''def delete_job(task_id: str) -> bool:
    init_schema()
    if not _TASK_ID_RE.match(task_id or ""):
        raise ValueError("некорректный task_id")
    with _connect() as conn:
        conn.execute("DELETE FROM results WHERE task_id = ?", (task_id,))
        cur = conn.execute("DELETE FROM jobs WHERE task_id = ?", (task_id,))
        conn.commit()
        return cur.rowcount > 0''',
    ),
    (
        "real",
        "FastAPI-роут POST /api/rag/compare: валидирует payload через from_body, "
        "503 если нет провайдера, 400 на пустые поля, ошибки агента → HTTPException.",
        r'''@router.post("/api/rag/compare")
async def rag_compare(request: Request):
    """Один вопрос — два ответа (без контекста из индекса и с RAG)."""
    if not providers:
        raise HTTPException(
            status_code=503,
            detail="Нет настроенного провайдера (нужен ROUTERAI_API_KEY).",
        )
    body = await request.json()
    rc = RagComparePayload.from_body(body if isinstance(body, dict) else {})
    if not rc.provider_name or not rc.model or not rc.message:
        raise HTTPException(
            status_code=400,
            detail="Нужны поля provider, model и непустой message.",
        )
    try:
        return await agent.compare_rag_answers(
            rc.provider_name, rc.model, rc.message,
            temperature=rc.temperature, top_k=rc.top_k,
        )
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "LookupError") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "ValueError") from exc''',
    ),
    (
        "real",
        "GET /api/models: обнови модели Ollama, если провайдер локальный, и верни "
        "список моделей агента.",
        r'''@router.get("/api/models")
async def list_models():
    from ..providers import OllamaProvider

    ollama = providers.get("ollama")
    if isinstance(ollama, OllamaProvider):
        await ollama.refresh_models()
    return agent.list_models()''',
    ),
    (
        "real",
        "Метод FSM: собери человекочитаемое сообщение о запрещённом переходе между "
        "фазами задачи, с учётом терминальной фазы done.",
        r'''def _illegal_transition_message(self, current: str, target: str) -> str:
    allowed = TASK_ALLOWED_EDGES.get(current, ())
    if current == "done" and target != "done":
        return (
            f"Illegal transition '{current}' -> '{target}': 'done' is terminal; "
            "use action 'reset' for a new task."
        )
    return (
        f"Illegal transition '{current}' -> '{target}'. "
        f"Allowed next phases from '{current}': {list(allowed)}."
    )''',
    ),
    (
        "real",
        "Нормализуй content из message/delta: строка или список частей "
        "(OpenAI-совместимо) → плоская строка.",
        r'''def _normalize_stream_content(content: object) -> str:
    """Текст из message/delta content: строка или список частей (OpenAI-совместимо)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return str(content)''',
    ),
    (
        "real",
        "Определи, что пользователь явно одобряет план (для выхода из фазы planning "
        "без ручного Next). Учти отрицания.",
        r'''@staticmethod
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
    if "план" in t and ("одобряю" in t or "принимаю" in t):
        return True
    if "approve" in t and "plan" in t:
        return True
    return False''',
    ),
    (
        "real",
        "Красивый label для модели Ollama из её сырого имени: обрежь -GGUF, вынеси "
        "квантизацию в скобки, добавь суффикс — Local.",
        r'''def _label_for_ollama_model(model_id: str) -> str:
    """'.../Vikhr-Nemo-12B-Instruct-GGUF:Q4_K_M' → 'Vikhr-Nemo-12B-Instruct (Q4_K_M) — Local'."""
    tag = ""
    base = model_id
    if ":" in model_id:
        base, tag = model_id.split(":", 1)
    short = base.rsplit("/", 1)[-1]
    for suf in ("-GGUF", "-gguf"):
        if short.endswith(suf):
            short = short[: -len(suf)]
    qpart = f" ({tag})" if tag else ""
    return f"{short}{qpart} — Local"''',
    ),
    (
        "real",
        "Тест на планировщик: изоляция БД через tempfile + SCHEDULER_SQLITE_PATH + "
        "importlib.reload; проверь регистрацию и обработку задачи heartbeat.",
        r'''"""Периодический планировщик (SQLite), без MCP-процесса."""

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
            task_id="t1", interval_seconds=120, task_type="heartbeat_rollup",
            payload="p1", first_run_in_seconds=10,
        )
        with sqlite3.connect(self._path) as conn:
            conn.execute("UPDATE jobs SET next_run = 0 WHERE task_id = ?", ("t1",))
            conn.commit()
        n = self.ss.process_due_jobs()
        self.assertEqual(n, 1)''',
    ),
    (
        "real",
        "Валидация регистрации задачи планировщика: task_id по regex, task_type из "
        "разрешённого множества, интервал и первый запуск с клампом границ.",
        r'''_TASK_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_ALLOWED_TYPES = frozenset({"reminder", "http_sample", "heartbeat_rollup"})


def _validate_job_args(task_id: str, task_type: str) -> None:
    if not _TASK_ID_RE.match(task_id or ""):
        raise ValueError("task_id: только латиница, цифры, _ и -, длина 1–64")
    if (task_type or "").strip() not in _ALLOWED_TYPES:
        raise ValueError(
            f"task_type должен быть одним из: {', '.join(sorted(_ALLOWED_TYPES))}",
        )''',
    ),
    (
        "real",
        "Payload сравнения RAG-режимов: from_body с дефолтами и безопасным разбором "
        "min_similarity через try/except.",
        r'''@dataclass(frozen=True)
class RagModesComparePayload:
    provider_name: str
    model: str
    message: str
    temperature: float
    top_k: int
    min_similarity: float

    @classmethod
    def from_body(cls, body: dict[str, Any]) -> RagModesComparePayload:
        raw_min = body.get("min_similarity", 0.28)
        try:
            min_sim = float(raw_min)
        except (TypeError, ValueError):
            min_sim = 0.28
        return cls(
            provider_name=str(body.get("provider") or "").strip(),
            model=str(body.get("model") or "").strip(),
            message=str(body.get("message") or "").strip(),
            temperature=float(body.get("temperature", 0.35)),
            top_k=int(body.get("top_k") or 8),
            min_similarity=min_sim,
        )''',
    ),
    (
        "real",
        "GET /api/branches?conversation_id=...: верни список веток разговора у агента.",
        r'''@router.get("/api/branches")
async def list_branches(conversation_id: str):
    return agent.list_branches(conversation_id)''',
    ),
    # ----------------------------------------------------------- SYNTHETIC ---
    (
        "synthetic",
        "Добавь роут GET /api/health, который возвращает статус сервиса и текущее "
        "время в epoch.",
        r'''@router.get("/api/health")
async def health():
    """Простой health-check: сервис жив и его время."""
    return {"ok": True, "time": time.time()}''',
    ),
    (
        "synthetic",
        "Сделай payload-класс для запроса создания ветки: conversation_id и name, "
        "разбор через from_body с дефолтами.",
        r'''@dataclass(frozen=True)
class CreateBranchPayload:
    conversation_id: str
    name: str

    @classmethod
    def from_body(cls, body: dict[str, Any]) -> CreateBranchPayload:
        return cls(
            conversation_id=str(body.get("conversation_id") or "default").strip(),
            name=str(body.get("name") or "").strip(),
        )''',
    ),
    (
        "synthetic",
        "Роут POST /api/branches: создаёт ветку через payload, 400 если имя пустое, "
        "ошибку значения агента пробрасывает как HTTPException.",
        r'''@router.post("/api/branches")
async def create_branch(request: Request):
    """Создать ветку разговора."""
    body = await request.json()
    p = CreateBranchPayload.from_body(body if isinstance(body, dict) else {})
    if not p.name:
        raise HTTPException(status_code=400, detail="Нужно непустое поле name.")
    try:
        return agent.create_branch(p.conversation_id, p.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "ValueError") from exc''',
    ),
    (
        "synthetic",
        "Функция хранилища: посчитай, сколько результатов сохранено для задачи по "
        "task_id. Валидируй идентификатор.",
        r'''def count_results(task_id: str) -> int:
    init_schema()
    if not _TASK_ID_RE.match(task_id or ""):
        raise ValueError("некорректный task_id")
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM results WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    return int(row["c"]) if row else 0''',
    ),
    (
        "synthetic",
        "Функция хранилища: верни задачу планировщика по task_id или None, если её нет.",
        r'''def get_job(task_id: str) -> dict | None:
    init_schema()
    if not _TASK_ID_RE.match(task_id or ""):
        raise ValueError("некорректный task_id")
    with _connect() as conn:
        row = conn.execute(
            "SELECT task_id, interval_seconds, task_type, payload, next_run, "
            "last_run, created_at FROM jobs WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    return dict(row) if row else None''',
    ),
    (
        "synthetic",
        "Помести N последних результатов задачи (новые сверху) в список dict. "
        "Ограничь лимит диапазоном 1..200.",
        r'''def recent_results(task_id: str, limit: int = 20) -> list[dict]:
    init_schema()
    if not _TASK_ID_RE.match(task_id or ""):
        raise ValueError("некорректный task_id")
    lim = max(1, min(200, int(limit)))
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, ts, content FROM results WHERE task_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (task_id, lim),
        ).fetchall()
    return [dict(r) for r in rows]''',
    ),
    (
        "synthetic",
        "Абстрактный интерфейс хранилища памяти разговоров: load и save одного "
        "разговора по id.",
        r'''"""Интерфейс хранилища памяти разговоров."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ConversationStore(ABC):
    @abstractmethod
    def load(self, conversation_id: str) -> dict:
        ...

    @abstractmethod
    def save(self, conversation_id: str, state: dict) -> None:
        ...''',
    ),
    (
        "synthetic",
        "DTO результата RAG-поиска: чанк текста, источник и косинусная близость.",
        r'''@dataclass
class RagHit:
    chunk: str
    source: str
    score: float''',
    ),
    (
        "synthetic",
        "Хелпер: обрежь текст до max_chars, добавив многоточие, если пришлось резать.",
        r'''def _truncate(text: str, max_chars: int = 800) -> str:
    """Обрезать текст до max_chars, добавив многоточие при обрезке."""
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"''',
    ),
    (
        "synthetic",
        "Стриминговый роут SSE POST /api/echo: отдаёт присланный текст словами через "
        "text/event-stream с заголовками no-store.",
        r'''@router.post("/api/echo")
async def echo(request: Request):
    """Демо-SSE: возвращает текст пословно."""
    body = await request.json()
    text = str(body.get("text") or "").strip()

    async def event_stream():
        for word in text.split():
            yield f"data: {json.dumps(word, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )''',
    ),
    (
        "synthetic",
        "Косинусное сходство двух векторов на numpy. Верни 0.0, если любой из "
        "векторов нулевой.",
        r'''def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство двух векторов; 0.0 если один из них нулевой."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))''',
    ),
    (
        "synthetic",
        "Разбей текст на чанки фиксированной длины с перекрытием (по символам). "
        "Верни список непустых чанков.",
        r'''def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Фиксированное окно с перекрытием: список непустых чанков."""
    t = (text or "").strip()
    if not t:
        return []
    step = max(1, size - overlap)
    chunks: list[str] = []
    for start in range(0, len(t), step):
        piece = t[start : start + size].strip()
        if piece:
            chunks.append(piece)
    return chunks''',
    ),
    (
        "synthetic",
        "Роут GET /api/tasks/{task_id}/results: агрегированные результаты задачи; "
        "404 если задачи нет, 400 на некорректный id.",
        r'''@router.get("/api/tasks/{task_id}/results")
async def task_results(task_id: str):
    """Агрегированные результаты периодической задачи."""
    try:
        job = scheduler_store.get_job(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена.")
    return scheduler_store.get_aggregated_results(task_id)''',
    ),
    (
        "synthetic",
        "Тест на cosine_similarity: одинаковые векторы дают ~1.0, ортогональные — "
        "0.0, нулевой вектор — 0.0.",
        r'''"""Тест косинусного сходства."""

from __future__ import annotations

import unittest

import numpy as np

from app.rag.similarity import cosine_similarity


class TestCosineSimilarity(unittest.TestCase):
    def test_identical(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=6)

    def test_orthogonal(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0, places=6)

    def test_zero_vector(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        self.assertEqual(cosine_similarity(a, b), 0.0)''',
    ),
    (
        "synthetic",
        "Провайдер-заглушка EchoProvider на базе AIProvider: стримит обратно "
        "контент последнего пользовательского сообщения.",
        r'''class EchoProvider(AIProvider):
    """Тестовый провайдер: возвращает последнее сообщение пользователя."""

    name = "echo"
    models = [{"id": "echo", "label": "Echo — тест"}]

    async def stream_chat(
        self, messages: list[Message], model: str, temperature: float = 0.7
    ) -> AsyncIterator[StreamResult]:
        last_user = ""
        for m in messages:
            if m.role == "user":
                last_user = m.content
        yield StreamResult(text=last_user)
        yield StreamResult(meta={"model": model, "echo": True})''',
    ),
    (
        "synthetic",
        "Хелпер безопасного парса JSON-строки: верни dict или пустой dict при ошибке "
        "разбора. Не глотай другие исключения.",
        r'''def _parse_json_obj(raw: str) -> dict:
    """JSON-строка → dict; при ошибке разбора — пустой dict."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}''',
    ),
    (
        "synthetic",
        "Роут DELETE /api/tasks/{task_id}: удалить задачу планировщика; 404 если "
        "нечего удалять, 400 на некорректный id.",
        r'''@router.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """Удалить периодическую задачу и её результаты."""
    try:
        removed = scheduler_store.delete_job(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="Задача не найдена.")
    return {"ok": True, "task_id": task_id}''',
    ),
    (
        "synthetic",
        "Константы фаз задачи и разрешённые переходы FSM вынеси в модуль констант.",
        r'''"""Константы FSM задачи: фазы и разрешённые переходы."""

from __future__ import annotations

TASK_PHASES = ("planning", "plan_approved", "execution", "validation", "done")

TASK_ALLOWED_EDGES = {
    "planning": ("plan_approved",),
    "plan_approved": ("execution",),
    "execution": ("validation",),
    "validation": ("done", "execution"),
    "done": (),
}''',
    ),
    (
        "synthetic",
        "Функция: по текущей фазе задачи верни следующую линейную фазу или None, "
        "если это терминальная фаза.",
        r'''def _next_phase_linear(current: str) -> str | None:
    """Следующая фаза по линейному порядку; None для терминальной 'done'."""
    order = {
        "planning": "plan_approved",
        "plan_approved": "execution",
        "execution": "validation",
        "validation": "done",
        "done": None,
    }
    return order.get(current)''',
    ),
    (
        "synthetic",
        "Асинхронный httpx-клиент: GET по url с таймаутом, верни JSON или подними "
        "ValueError с русским текстом при ошибке сети/разбора.",
        r'''async def fetch_json(url: str, timeout: float = 15.0) -> dict:
    """GET url → JSON dict; сетевые/JSON-ошибки → ValueError с русским текстом."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as exc:
        raise ValueError(f"Сетевая ошибка при запросе {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ответ {url} не является JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("Ожидался JSON-объект")
    return data''',
    ),
    (
        "synthetic",
        "DTO настроек RAG-поиска: стратегия, top_k, порог сходства и опциональный "
        "путь к индексу, с from_body.",
        r'''@dataclass(frozen=True)
class RagSearchPayload:
    strategy: str
    top_k: int
    min_similarity: float
    index_path: str | None

    @classmethod
    def from_body(cls, body: dict[str, Any]) -> RagSearchPayload:
        raw_idx = body.get("index_path")
        ip = str(raw_idx).strip() if isinstance(raw_idx, str) and raw_idx.strip() else None
        return cls(
            strategy=str(body.get("strategy") or "fixed").lower().strip(),
            top_k=int(body.get("top_k") or 8),
            min_similarity=float(body.get("min_similarity") or 0.28),
            index_path=ip,
        )''',
    ),
    (
        "synthetic",
        "Тест на payload ChatRequestPayload.from_body: дефолты при пустом теле и "
        "корректный разбор заполненного.",
        r'''"""Тест разбора тела запроса чата."""

from __future__ import annotations

import unittest

from app.payloads import ChatRequestPayload


class TestChatRequestPayload(unittest.TestCase):
    def test_defaults_on_empty_body(self) -> None:
        p = ChatRequestPayload.from_body({})
        self.assertEqual(p.conversation_id, "default")
        self.assertEqual(p.branch_id, "main")
        self.assertEqual(p.raw_messages, [])
        self.assertFalse(p.resume)

    def test_parses_filled_body(self) -> None:
        p = ChatRequestPayload.from_body(
            {"provider": "routerai", "model": "gpt", "temperature": 0.2,
             "messages": [{"role": "user", "content": "hi"}]}
        )
        self.assertEqual(p.provider_name, "routerai")
        self.assertEqual(p.temperature, 0.2)
        self.assertEqual(len(p.raw_messages), 1)''',
    ),
    (
        "synthetic",
        "Хелпер: собери список dict-сообщений {role, content} из списка Message для "
        "тела запроса к API.",
        r'''def _messages_to_payload(messages: list[Message]) -> list[dict]:
    """Список Message → список dict {role, content} для тела запроса."""
    return [{"role": m.role, "content": m.content} for m in messages]''',
    ),
    (
        "synthetic",
        "Роут POST /api/checkpoints: создаёт чекпойнт разговора; conversation_id из "
        "тела с дефолтом default.",
        r'''@router.post("/api/checkpoints")
async def create_checkpoint(request: Request):
    """Зафиксировать текущее состояние разговора как чекпойнт."""
    body = await request.json()
    conversation_id = str(body.get("conversation_id") or "default").strip()
    label = str(body.get("label") or "").strip()
    return agent.create_checkpoint(conversation_id, label)''',
    ),
    (
        "synthetic",
        "Функция: клампни целое значение в границы [lo, hi], безопасно приведя вход "
        "к int (при ошибке — вернуть lo).",
        r'''def _clamp_int(value: object, lo: int, hi: int) -> int:
    """Привести value к int и зажать в [lo, hi]; при ошибке разбора — lo."""
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, n))''',
    ),
    (
        "synthetic",
        "Простой скользящий контекст: оставь последние n сообщений, но всегда "
        "сохрани системное сообщение в начале, если оно есть.",
        r'''def sliding_window(messages: list[Message], n: int = 10) -> list[Message]:
    """Последние n сообщений; системное (если первое) сохраняется всегда."""
    if not messages:
        return []
    head: list[Message] = []
    rest = messages
    if messages[0].role == "system":
        head = [messages[0]]
        rest = messages[1:]
    return head + rest[-n:]''',
    ),
    (
        "synthetic",
        "CLI-скрипт: точка входа build_index, которая печатает прогресс в stderr, а "
        "не в stdout.",
        r'''"""CLI: сборка RAG-индекса из корпуса."""

from __future__ import annotations

import sys


def main() -> int:
    print("Собираю индекс…", file=sys.stderr)
    count = _build()
    print(f"Готово: {count} чанков.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())''',
    ),
    (
        "synthetic",
        "Функция обновления задачи: сменить интервал у существующей задачи, вернуть "
        "True при успехе. Клампни интервал 15..86400.",
        r'''def update_interval(task_id: str, interval_seconds: int) -> bool:
    init_schema()
    if not _TASK_ID_RE.match(task_id or ""):
        raise ValueError("некорректный task_id")
    iv = max(15, min(86400, int(interval_seconds)))
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE jobs SET interval_seconds = ? WHERE task_id = ?",
            (iv, task_id),
        )
        conn.commit()
        return cur.rowcount > 0''',
    ),
    (
        "synthetic",
        "Роут GET /api/tasks: список задач планировщика, спроецированный только в "
        "поля task_id, task_type и interval_seconds.",
        r'''@router.get("/api/tasks")
async def list_tasks():
    """Список задач планировщика (краткая проекция)."""
    jobs = scheduler_store.list_jobs()
    return [
        {
            "task_id": j["task_id"],
            "task_type": j["task_type"],
            "interval_seconds": j["interval_seconds"],
        }
        for j in jobs
    ]''',
    ),
    (
        "synthetic",
        "Dataclass-конфиг провайдера Ollama: base_url, num_ctx, num_predict с "
        "дефолтами; base_url без хвостового слэша через __post_init__.",
        r'''@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    num_ctx: int = 8192
    num_predict: int = 1024

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")''',
    ),
    (
        "synthetic",
        "Функция: определи по имени модели, эмбеддинг ли это (по набору паттернов в "
        "имени). Сравнивай только хвост после последнего слэша.",
        r'''_EMBEDDING_MODEL_PATTERNS = ("bge-", "nomic-embed", "-embed", "e5-", "gte-")


def _is_embedding_model(model_id: str) -> bool:
    """True, если имя модели похоже на эмбеддинг-модель."""
    mid = model_id.rsplit("/", 1)[-1].lower()
    return any(p in mid for p in _EMBEDDING_MODEL_PATTERNS)''',
    ),
    (
        "synthetic",
        "Роут POST /api/tasks: регистрирует задачу планировщика из тела; 400 на "
        "ValueError валидации, иначе возвращает результат register_job.",
        r'''@router.post("/api/tasks")
async def create_task(request: Request):
    """Зарегистрировать периодическую задачу."""
    body = await request.json()
    try:
        return scheduler_store.register_job(
            task_id=str(body.get("task_id") or "").strip(),
            interval_seconds=int(body.get("interval_seconds") or 60),
            task_type=str(body.get("task_type") or "").strip(),
            payload=str(body.get("payload") or ""),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc''',
    ),
    (
        "synthetic",
        "Функция: нормализуй список сырых сообщений (list[dict]) в list[Message], "
        "пропуская элементы без role/content.",
        r'''def _coerce_messages(raw: list[dict]) -> list[Message]:
    """Сырые dict-сообщения → list[Message]; элементы без role/content отбрасываются."""
    out: list[Message] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if isinstance(role, str) and isinstance(content, str) and content:
            out.append(Message(role=role, content=content))
    return out''',
    ),
    (
        "synthetic",
        "Тест на FSM-переходы: из planning нельзя прыгнуть сразу в execution — "
        "функция допустимости перехода возвращает False.",
        r'''"""Тест допустимости переходов FSM задачи."""

from __future__ import annotations

import unittest

from app.agent_constants import TASK_ALLOWED_EDGES


def _is_allowed(current: str, target: str) -> bool:
    return target in TASK_ALLOWED_EDGES.get(current, ())


class TestTaskEdges(unittest.TestCase):
    def test_planning_cannot_skip_to_execution(self) -> None:
        self.assertFalse(_is_allowed("planning", "execution"))

    def test_planning_to_plan_approved_ok(self) -> None:
        self.assertTrue(_is_allowed("planning", "plan_approved"))

    def test_done_is_terminal(self) -> None:
        self.assertEqual(TASK_ALLOWED_EDGES.get("done"), ())''',
    ),
    (
        "synthetic",
        "Хелпер форматирования usage-меты: собери dict со временем в мс и токенами "
        "из сырого usage провайдера.",
        r'''def _build_usage_meta(usage: dict, elapsed_ms: int) -> dict:
    """Сырой usage провайдера → нормализованная мета для стрима."""
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    total = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
    return {
        "time_ms": elapsed_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total,
    }''',
    ),
    (
        "synthetic",
        "Роут GET /api/rag/status: верни статус RAG-индекса через существующий "
        "билдер ответа.",
        r'''@router.get("/api/rag/status")
async def rag_status():
    """Статус RAG-индекса (наличие, число чанков, путь)."""
    return build_rag_status_response()''',
    ),
    (
        "synthetic",
        "Функция: выбери top_k результатов RAG по убыванию score, отфильтровав те, "
        "что ниже порога min_similarity.",
        r'''def _top_hits(hits: list[RagHit], top_k: int, min_similarity: float) -> list[RagHit]:
    """Отфильтровать по порогу и вернуть top_k по убыванию score."""
    filtered = [h for h in hits if h.score >= min_similarity]
    filtered.sort(key=lambda h: h.score, reverse=True)
    return filtered[: max(0, top_k)]''',
    ),
    (
        "synthetic",
        "Мидлварь-хелпер: извлеки Bearer-токен из заголовка Authorization или верни "
        "None, если его нет/формат не тот.",
        r'''def _bearer_token(auth_header: str | None) -> str | None:
    """Достать токен из 'Authorization: Bearer <token>' или None."""
    if not auth_header:
        return None
    prefix = "Bearer "
    if not auth_header.startswith(prefix):
        return None
    token = auth_header[len(prefix) :].strip()
    return token or None''',
    ),
    (
        "synthetic",
        "Dataclass чекпойнта разговора: id, метка, epoch создания и снимок "
        "состояния; фабрика создания с текущим временем.",
        r'''@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    label: str
    created_at: float
    state: dict

    @classmethod
    def create(cls, checkpoint_id: str, label: str, state: dict) -> Checkpoint:
        return cls(
            checkpoint_id=checkpoint_id,
            label=label.strip(),
            created_at=time.time(),
            state=state,
        )''',
    ),
    (
        "synthetic",
        "Функция: сгенерируй короткий id из 8 hex-символов на основе secrets.",
        r'''def _short_id() -> str:
    """Короткий случайный id из 8 hex-символов."""
    return secrets.token_hex(4)''',
    ),
    (
        "synthetic",
        "Роут PATCH /api/tasks/{task_id}/interval: меняет интервал задачи; 404 если "
        "нет, 400 на ошибку валидации.",
        r'''@router.patch("/api/tasks/{task_id}/interval")
async def patch_interval(task_id: str, request: Request):
    """Изменить интервал периодической задачи."""
    body = await request.json()
    try:
        updated = scheduler_store.update_interval(
            task_id, int(body.get("interval_seconds") or 60)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=404, detail="Задача не найдена.")
    return {"ok": True, "task_id": task_id}''',
    ),
    (
        "synthetic",
        "Функция: посчитай приблизительное число токенов как длину текста в "
        "символах, делённую на 4 (грубая эвристика), минимум 1 для непустого.",
        r'''def _approx_tokens(text: str) -> int:
    """Грубая оценка числа токенов ≈ символы / 4 (минимум 1 для непустого)."""
    t = (text or "").strip()
    if not t:
        return 0
    return max(1, len(t) // 4)''',
    ),
    (
        "synthetic",
        "Тест на sliding_window: системное сообщение сохраняется, а из остальных "
        "остаются только последние n.",
        r'''"""Тест скользящего окна контекста."""

from __future__ import annotations

import unittest

from app.providers import Message
from app.agent.context import sliding_window


class TestSlidingWindow(unittest.TestCase):
    def test_keeps_system_and_last_n(self) -> None:
        msgs = [Message("system", "s")] + [
            Message("user", str(i)) for i in range(5)
        ]
        out = sliding_window(msgs, n=2)
        self.assertEqual(out[0].role, "system")
        self.assertEqual([m.content for m in out[1:]], ["3", "4"])

    def test_empty(self) -> None:
        self.assertEqual(sliding_window([], n=3), [])''',
    ),
    (
        "synthetic",
        "Функция: безопасно достань вложенное значение из dict по цепочке ключей, "
        "верни default, если путь оборвался.",
        r'''def _dig(data: dict, *keys: str, default: object = None) -> object:
    """Достать data[k1][k2]...; вернуть default, если путь оборвался."""
    cur: object = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur''',
    ),
    (
        "synthetic",
        "Роут POST /api/branches/{branch_id}/switch: переключить активную ветку; "
        "LookupError агента → 404.",
        r'''@router.post("/api/branches/{branch_id}/switch")
async def switch_branch(branch_id: str, request: Request):
    """Сделать ветку активной для разговора."""
    body = await request.json()
    conversation_id = str(body.get("conversation_id") or "default").strip()
    try:
        return agent.switch_branch(conversation_id, branch_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip() or "Ветка не найдена") from exc''',
    ),
    (
        "synthetic",
        "Функция хранилища: удали результаты задачи старше given epoch, верни число "
        "удалённых строк.",
        r'''def prune_results(task_id: str, older_than_epoch: float) -> int:
    init_schema()
    if not _TASK_ID_RE.match(task_id or ""):
        raise ValueError("некорректный task_id")
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM results WHERE task_id = ? AND ts < ?",
            (task_id, older_than_epoch),
        )
        conn.commit()
        return cur.rowcount''',
    ),
    (
        "synthetic",
        "Enum-подобные константы стратегий контекста и хелпер валидации значения "
        "стратегии с понятной ошибкой.",
        r'''"""Стратегии сборки контекста разговора."""

from __future__ import annotations

CONTEXT_STRATEGIES = frozenset({"sliding", "full", "summary"})


def validate_strategy(strategy: str) -> str:
    """Проверить стратегию контекста; вернуть нормализованное значение."""
    s = (strategy or "").strip().lower()
    if s not in CONTEXT_STRATEGIES:
        raise ValueError(
            f"context_strategy должен быть одним из: {', '.join(sorted(CONTEXT_STRATEGIES))}",
        )
    return s''',
    ),
    (
        "synthetic",
        "Асинхронный генератор: обёртка над стримом провайдера, которая склеивает "
        "весь текст и в конце отдаёт StreamResult с полным ответом в meta.",
        r'''async def collect_stream(
    provider: AIProvider, messages: list[Message], model: str
) -> AsyncIterator[StreamResult]:
    """Пробросить чанки провайдера и в конце отдать полный текст в meta."""
    parts: list[str] = []
    async for result in provider.stream_chat(messages, model):
        if result.text is not None:
            parts.append(result.text)
        yield result
    yield StreamResult(meta={"full_text": "".join(parts)})''',
    ),
    (
        "synthetic",
        "Функция: приведи температуру к безопасному float в диапазоне 0.0..2.0, "
        "дефолт 0.7 при ошибке разбора.",
        r'''def _safe_temperature(value: object, default: float = 0.7) -> float:
    """Привести температуру к float в [0.0, 2.0]; при ошибке — default."""
    try:
        t = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(0.0, min(2.0, t))''',
    ),
    (
        "synthetic",
        "Тест на delete_job: удаление несуществующей задачи возвращает False, "
        "существующей — True. Изоляция БД через tempfile.",
        r'''"""Тест удаления задач планировщика."""

from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path


class TestDeleteJob(unittest.TestCase):
    def setUp(self) -> None:
        self._fd, self._path = tempfile.mkstemp(suffix=".sqlite")
        os.close(self._fd)
        os.environ["SCHEDULER_SQLITE_PATH"] = self._path
        import app.scheduler_store as ss

        importlib.reload(ss)
        self.ss = ss

    def tearDown(self) -> None:
        os.environ.pop("SCHEDULER_SQLITE_PATH", None)
        Path(self._path).unlink(missing_ok=True)

    def test_delete_missing_returns_false(self) -> None:
        self.assertFalse(self.ss.delete_job("nope"))

    def test_delete_existing_returns_true(self) -> None:
        self.ss.register_job(
            task_id="j1", interval_seconds=60, task_type="reminder", payload="x",
        )
        self.assertTrue(self.ss.delete_job("j1"))''',
    ),
    (
        "synthetic",
        "Функция: собери человекочитаемое превью результата задачи по её kind "
        "(reminder/http_sample/heartbeat_rollup), иначе обрезанный сырой контент.",
        r'''def _result_preview(obj: dict, raw: str) -> str:
    """Короткое превью результата задачи по его kind."""
    kind = str(obj.get("kind", "?"))
    if kind == "reminder":
        return f"напоминание: {(obj.get('note') or '')[:160]}"
    if kind == "http_sample":
        return f"HTTP {obj.get('status_code', obj.get('error', '?'))} {str(obj.get('url', ''))[:80]}"
    if kind == "heartbeat_rollup":
        return f"heartbeat tick={obj.get('tick')}"
    return raw[:160]''',
    ),
    (
        "synthetic",
        "Роут GET /api/branches/{branch_id}/messages: сообщения ветки; LookupError "
        "агента → 404.",
        r'''@router.get("/api/branches/{branch_id}/messages")
async def branch_messages(branch_id: str, conversation_id: str):
    """Сообщения указанной ветки разговора."""
    try:
        return agent.branch_messages(conversation_id, branch_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip() or "Ветка не найдена") from exc''',
    ),
    (
        "synthetic",
        "Функция: дедуплицируй список строк, сохранив порядок первого появления.",
        r'''def _dedup_keep_order(items: list[str]) -> list[str]:
    """Убрать дубли, сохранив порядок первого появления."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out''',
    ),
]


def _to_record(system: str, user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Дедуп по тексту user-запроса.
    seen_users: set[str] = set()
    records: list[dict] = []
    real_count = 0
    for source, user, assistant in EXAMPLES:
        key = user.strip().lower()
        if key in seen_users:
            print(f"Пропущен дубль user-запроса: {user[:50]}…", file=sys.stderr)
            continue
        seen_users.add(key)
        records.append(_to_record(SYSTEM, user.strip(), assistant.strip()))
        if source == "real":
            real_count += 1

    total = len(records)
    if total < 50:
        print(f"ОШИБКА: примеров всего {total}, нужно ≥50.", file=sys.stderr)
        return 1

    # Детерминированный сплит 80/20.
    rng = random.Random(42)
    idx = list(range(total))
    rng.shuffle(idx)
    eval_size = max(10, round(total * 0.2))
    eval_idx = set(idx[:eval_size])

    train = [records[i] for i in range(total) if i not in eval_idx]
    ev = [records[i] for i in range(total) if i in eval_idx]

    _write_jsonl(data_dir / "raw.jsonl", records)
    _write_jsonl(data_dir / "train.jsonl", train)
    _write_jsonl(data_dir / "eval.jsonl", ev)

    real_pct = round(real_count / total * 100)
    print(
        f"Готово: всего {total} (реальных {real_count} ≈ {real_pct}%), "
        f"train {len(train)}, eval {len(ev)}.",
        file=sys.stderr,
    )
    if real_pct < 20:
        print(f"ВНИМАНИЕ: доля реальных {real_pct}% < 20%.", file=sys.stderr)
    return 0


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
