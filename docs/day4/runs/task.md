# Head-to-head задача (одна и та же для локальной и облачной модели)

Задача подобрана как **дискриминатор конвенций**: у неё есть «домашний» правильный
ответ, и есть очевидная ловушка (написать сырой SQL прямо в роутере). Проверяет:
инвариант reuse-first, запрет `sqlite3.connect` в роутах, русский `HTTPException(404)`,
`from __future__`, docstring, type-hints.

## Промпт (user message)

> Добавь в `app/scheduler_routes.py` эндпоинт `GET /api/scheduler/jobs/{task_id}`,
> который возвращает одну джобу планировщика по её `task_id`. Если такой джобы нет —
> отвечай 404. Данные бери из слоя `app/scheduler_store.py`. Верни обновлённый
> `app/scheduler_routes.py` целиком.

## Контекст, который дан обеим моделям

(в IDE Continue это пришло бы через `@file`; в скриптовом прогоне вставлено в промпт)

`app/scheduler_routes.py` — текущий файл целиком.
`app/scheduler_store.py` — релевантная часть: есть `list_jobs()`, `_connect()`,
`init_schema()`; **отдельной `get_job(task_id)` в сторе нет**.

## Эталон «домашнего» ответа (для оценки)

Правильных решения два, оба чистые:
- **A (reuse-first в роутере):** отфильтровать `list_jobs()` по `task_id` в обработчике,
  на пустой результат — `HTTPException(404, detail="…")`. Ноль нового доступа к БД.
- **B (тонкая функция в сторе):** добавить `get_job(task_id)` в `scheduler_store.py`
  через `_connect()` + параметризованный SQL, роутер её вызывает.

Обе — с `from __future__`, docstring роут-функции, аннотацией `-> dict`.

**Антипаттерн (провал):** `sqlite3.connect(...)` или сырой SQL прямо в `scheduler_routes.py`;
`except:`/`Any`; ответ-`{"error": ...}` с кодом 200 вместо `HTTPException(404)`.
