# Baseline (openai/gpt-4o-mini, без файнтюна)

Провайдер: **routerai**. Температура 0.0. Примеров: 10 (из eval).

Точка отсчёта: как базовая модель отвечает на те же запросы ДО дообучения. После файнтюна сравниваем по критериям из `criteria.md`.

## Пример 1 (3976 мс)

**User:**

> Верни список всех задач планировщика из хранилища.

**Baseline-ответ:**

````python
```python
from __future__ import annotations
from dataclasses import dataclass
from fastapi import APIRouter, HTTPException
from typing import AsyncIterator, list
from app.scheduler_store import fetch_all_tasks

router = APIRouter()

@dataclass
class TaskDTO:
    id: int
    name: str
    description: str | None

@router.get("/tasks", response_model=list[TaskDTO])
async def get_all_tasks() -> list[TaskDTO]:
    try:
        tasks = await fetch_all_tasks()
        return [TaskDTO(id=task['id'], name=task['name'], description=task['description']) for task in tasks]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
```
````

**Эталон (reference):**

````python
def list_jobs() -> list[dict]:
    init_schema()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT task_id, interval_seconds, task_type, payload, next_run, "
            "last_run, created_at FROM jobs",
        ).fetchall()
    return [dict(r) for r in rows]
````

---

## Пример 2 (4194 мс)

**User:**

> GET /api/models: обнови модели Ollama, если провайдер локальный, и верни список моделей агента.

**Baseline-ответ:**

````python
```python
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from dataclasses import dataclass
from typing import list, AsyncIterator
from app.scheduler_store import get_local_models, update_local_models

router = APIRouter()

@dataclass
class ModelResponse:
    model_name: str
    is_updated: bool

@router.get("/api/models", response_model=list[ModelResponse])
async def get_models() -> list[ModelResponse]:
    try:
        local_models = await get_local_models()
        updated_models = []

        for model in local_models:
            is_updated = await update_local_models(model)
            updated_models.append(ModelResponse(model_name=model, is_updated=is_updated))

        return updated_models

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
```
````

**Эталон (reference):**

````python
@router.get("/api/models")
async def list_models():
    from ..providers import OllamaProvider

    ollama = providers.get("ollama")
    if isinstance(ollama, OllamaProvider):
        await ollama.refresh_models()
    return agent.list_models()
````

---

## Пример 3 (3656 мс)

**User:**

> GET /api/branches?conversation_id=...: верни список веток разговора у агента.

**Baseline-ответ:**

````python
```python
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query
from dataclasses import dataclass
from typing import AsyncIterator, list
from app.scheduler_store import get_branches_by_conversation_id

router = APIRouter()

@dataclass
class Branch:
    id: str
    name: str

@router.get("/api/branches", response_model=list[Branch])
async def get_branches(conversation_id: str = Query(..., description="ID разговора")) -> list[Branch]:
    try:
        branches = await get_branches_by_conversation_id(conversation_id)
        return [Branch(id=branch['id'], name=branch['name']) for branch in branches]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
```
````

**Эталон (reference):**

````python
@router.get("/api/branches")
async def list_branches(conversation_id: str):
    return agent.list_branches(conversation_id)
````

---

## Пример 4 (4333 мс)

**User:**

> Добавь роут GET /api/health, который возвращает статус сервиса и текущее время в epoch.

**Baseline-ответ:**

````python
```python
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from datetime import datetime
from time import time

router = APIRouter()

# Проверка состояния сервиса
@router.get("/api/health")
async def get_health() -> dict[str, int | str]:
    try:
        return {
            "status": "ok",
            "timestamp": int(time())
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
```
````

**Эталон (reference):**

````python
@router.get("/api/health")
async def health():
    """Простой health-check: сервис жив и его время."""
    return {"ok": True, "time": time.time()}
````

---

## Пример 5 (5071 мс)

**User:**

> Роут POST /api/branches: создаёт ветку через payload, 400 если имя пустое, ошибку значения агента пробрасывает как HTTPException.

**Baseline-ответ:**

````python
```python
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from dataclasses import dataclass
from typing import Any
from app.scheduler_store import create_branch
from app.models import Agent

router = APIRouter()

@dataclass
class BranchPayload:
    name: str

@router.post("/api/branches")
async def create_branch_endpoint(payload: BranchPayload) -> dict[str, Any]:
    if not payload.name:
        raise HTTPException(status_code=400, detail="Имя ветки не может быть пустым")
    
    try:
        branch = await create_branch(payload.name)
        return {"branch": branch}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
```
````

**Эталон (reference):**

````python
@router.post("/api/branches")
async def create_branch(request: Request):
    """Создать ветку разговора."""
    body = await request.json()
    p = CreateBranchPayload.from_body(body if isinstance(body, dict) else {})
    if not p.name:
        raise HTTPException(status_code=400, detail="Нужно непустое поле name.")
    try:
        return agent.create_branch(p.conversation_id, p.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "ValueError") from exc
````

---

## Пример 6 (4293 мс)

**User:**

> Функция хранилища: верни задачу планировщика по task_id или None, если её нет.

**Baseline-ответ:**

````python
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import sqlite3

DATABASE_PATH = "path/to/database.db"

@dataclass
class Task:
    task_id: str
    name: str
    description: str | None

def get_task_by_id(task_id: str) -> Optional[Task]:
    try:
        connection = sqlite3.connect(DATABASE_PATH)
        cursor = connection.cursor()
        cursor.execute("SELECT task_id, name, description FROM tasks WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        if row:
            return Task(task_id=row[0], name=row[1], description=row[2])
        return None
    except sqlite3.Error as exc:
        raise RuntimeError("Ошибка доступа к базе данных") from exc
    finally:
        connection.close()
```
````

**Эталон (reference):**

````python
def get_job(task_id: str) -> dict | None:
    init_schema()
    if not _TASK_ID_RE.match(task_id or ""):
        raise ValueError("некорректный task_id")
    with _connect() as conn:
        row = conn.execute(
            "SELECT task_id, interval_seconds, task_type, payload, next_run, "
            "last_run, created_at FROM jobs WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    return dict(row) if row else None
````

---

## Пример 7 (3639 мс)

**User:**

> Абстрактный интерфейс хранилища памяти разговоров: load и save одного разговора по id.

**Baseline-ответ:**

````python
```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ConversationDTO:
    id: str
    data: dict | None

class ConversationStorageInterface(ABC):
    @abstractmethod
    async def load(self, conversation_id: str) -> ConversationDTO | None:
        ...

    @abstractmethod
    async def save(self, conversation: ConversationDTO) -> None:
        ...
```
````

**Эталон (reference):**

````python
"""Интерфейс хранилища памяти разговоров."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ConversationStore(ABC):
    @abstractmethod
    def load(self, conversation_id: str) -> dict:
        ...

    @abstractmethod
    def save(self, conversation_id: str, state: dict) -> None:
        ...
````

---

## Пример 8 (3731 мс)

**User:**

> Провайдер-заглушка EchoProvider на базе AIProvider: стримит обратно контент последнего пользовательского сообщения.

**Baseline-ответ:**

````python
```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from fastapi import APIRouter, HTTPException
from typing import AsyncIterator

UIMessages = list[dict]

@dataclass
class UserMessage:
    content: str

class AIProvider(ABC):
    @abstractmethod
    async def stream_response(self, message: UserMessage) -> AsyncIterator[str]:
        pass

class EchoProvider(AIProvider):
    async def stream_response(self, message: UserMessage) -> AsyncIterator[str]:
        if not message.content:
            raise HTTPException(status_code=400, detail="Сообщение не может быть пустым")
        yield message.content

router = APIRouter()
```
````

**Эталон (reference):**

````python
class EchoProvider(AIProvider):
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
        yield StreamResult(meta={"model": model, "echo": True})
````

---

## Пример 9 (2557 мс)

**User:**

> DTO настроек RAG-поиска: стратегия, top_k, порог сходства и опциональный путь к индексу, с from_body.

**Baseline-ответ:**

````python
```python
from __future__ import annotations
from dataclasses import dataclass
from fastapi import Body

@dataclass
class RagSearchSettings:
    strategy: str
    top_k: int
    similarity_threshold: float
    index_path: str | None = Body(default=None)
```
````

**Эталон (reference):**

````python
@dataclass(frozen=True)
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
        )
````

---

## Пример 10 (4117 мс)

**User:**

> Роут POST /api/checkpoints: создаёт чекпойнт разговора; conversation_id из тела с дефолтом default.

**Baseline-ответ:**

````python
```python
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from dataclasses import dataclass
from typing import Optional
from app.scheduler_store import create_checkpoint

router = APIRouter()

DEFAULT_CONVERSATION_ID = "default"

@dataclass
class CheckpointCreateDTO:
    conversation_id: Optional[str] = DEFAULT_CONVERSATION_ID

@router.post("/api/checkpoints")
async def create_checkpoint_route(checkpoint: CheckpointCreateDTO) -> dict:
    try:
        checkpoint_id = await create_checkpoint(checkpoint.conversation_id)
        return {"checkpoint_id": checkpoint_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
```
````

**Эталон (reference):**

````python
@router.post("/api/checkpoints")
async def create_checkpoint(request: Request):
    """Зафиксировать текущее состояние разговора как чекпойнт."""
    body = await request.json()
    conversation_id = str(body.get("conversation_id") or "default").strip()
    label = str(body.get("label") or "").strip()
    return agent.create_checkpoint(conversation_id, label)
````

---
