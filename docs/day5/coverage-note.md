# HTTP-покрытие эндпоинтов после Run 1 (задача T17, research)

Профиль `research` (read-only) прошёл роуты `app/routers/hub.py`, `app/scheduler_routes.py`,
`app/mcp_panel.py` и сопоставил их с реальными вызовами `TestClient` в `tests/`.
«Покрыт» = есть HTTP-запрос на путь через `TestClient(app)` (не полнота проверки логики).

**Итог: покрыто 9 из 23 эндпоинтов на HTTP-слое (~39%).**

| Метод | Путь | Покрыт (TestClient) | Тест-файл(ы) / прим. |
|-------|------|---------------------|----------------------|
| GET  | /api/models | Да | tests/test_models_api.py |
| POST | /api/chat | Нет | SSE-стрим, HTTP-теста нет |
| POST | /api/rag/compare | Нет | логика — tests/test_rag_compare.py, но напрямую агента, не через HTTP |
| POST | /api/rag/compare_modes | Нет | — |
| GET  | /api/rag/status | Да | tests/test_rag_status_api.py |
| GET  | /api/branches | Да | tests/test_checkpoint_branch_flow.py |
| POST | /api/branches | Да | tests/test_branch_api.py; tests/test_checkpoint_branch_flow.py |
| POST | /api/checkpoints | Да | tests/test_checkpoint_api.py; tests/test_checkpoint_branch_flow.py |
| GET  | /api/memory | Нет | — |
| GET  | /api/profiles | Нет | есть только POST-тест |
| POST | /api/profiles | Да | tests/test_profiles_api.py (только негативные тела) |
| GET  | /api/task-state | Нет | — |
| GET  | /api/invariants | Нет | — |
| POST | /api/invariants | Нет | — |
| POST | /api/task-state | Нет | — |
| GET  | / | Нет | отдаёт index.html, HTTP-теста нет |
| GET  | /api/scheduler/ping | Нет | — |
| GET  | /api/scheduler/jobs | Нет | стор покрыт test_scheduler_store.py, но не HTTP-роут |
| GET  | /api/scheduler/jobs/{task_id} | Нет | 404-ветка не проверяется |
| GET  | /api/scheduler/stream | Нет | SSE, теста нет |
| GET  | /api/mcp/status | Да | tests/test_mcp_orchestration.py |
| POST | /api/mcp/connect | Да | tests/test_mcp_orchestration.py |
| POST | /api/mcp/disconnect | Да | tests/test_mcp_orchestration.py |

## Приоритет оставшихся дыр (по риску = объём логики × отсутствие теста)

1. **Весь `/api/scheduler/*`** — 4 роута, включая непроверенную 404-ветку `jobs/{task_id}`
   (добавлена в T13) и SSE-`stream`.
2. **`POST /api/chat` и оба `POST /api/rag/compare[_modes]`** — ядро продукта, самая тяжёлая
   логика, ноль HTTP-проверок.
3. **GET-читатели** `/api/memory`, `/api/profiles`, `/api/task-state`, `/api/invariants` +
   `POST /api/invariants`/`POST /api/task-state` — проще, но тоже без покрытия.

## Наблюдения

- `POST /api/profiles` дёргается только с невалидными телами (`[]`, `"hello"`) — happy-path
  создания профиля через HTTP не проверяется.
- Планировщик и RAG-compare покрыты на уровне стора/агента, но не на границе HTTP-роута
  (валидация 400/503 и разбор payload в самих роутах не проверяются).

_Источник: субагент `research`, Run 1 задача T17. Код при исследовании не менялся._
