# AI Chat Hub — архитектура

Кратко: FastAPI-приложение с веб-чатом, RAG-пайплайном, MCP-серверами и планировщиком. Стек: Python 3.10/3.12, FastAPI + Uvicorn, httpx, MCP SDK, numpy, nginx, Docker Compose, Ollama (локальная LLM на Mac).

## Слои

| Слой | Где | Что делает |
|---|---|---|
| HTTP / SSE | `app/main.py`, `app/routers/` | REST + SSE-стрим (`/api/chat`, `/api/models`, `/api/memory/*`, `/api/rag/*`, `/api/scheduler/*`, `/api/mcp/*`) |
| Агент | `app/agent/` (миксины: `streaming`, `context`, `rag_context`, `task_fsm`, `memory_branches`, `facts_wm`, `prompts`) | Сборка промпта, история, факты, ветки, FSM задачи, RAG-контекст |
| Провайдеры LLM | `app/providers.py` | `RouterAIProvider` (OpenAI-совместимый облачный), `OllamaProvider` (локальный, нативный `/api/chat`) |
| RAG | `app/rag/` | Чанкинг (fixed/structural), эмбеддинги (Ollama bge-m3 или RouterAI), retrieval (cosine + keyword), Day 23–24 фильтры/реранк/refusal |
| MCP | `app/mcp_panel.py`, `app/mcp_stdio_client.py`, `scripts/*_mcp_server.py` | stdio-клиент к локальным MCP-серверам, парсинг tool-call в стриме |
| Память | `data/agent_memory.json` (JSON-файл per-conversation: branches, checkpoints, facts, working_memory) |

## Поток запроса

1. Браузер `POST /api/chat` со списком messages, `provider`, `model`, `rag`, `profile_id`, `context_strategy`, `branch_id`.
2. `app/routers/hub.py` валидирует payload, вызывает `agent.stream_reply(...)`.
3. `AgentStreamingMixin.stream_reply` собирает контекст:
   - тянет state по `conversation_id`, выбирает branch
   - применяет стратегию (sliding / branching / facts) к истории
   - если `rag.enabled` — `_rag_context_message` делает retrieval (subqueries, cosine + keyword), генерит system-блок с отрывками
   - подмешивает profile/invariants/help system messages
4. Запрос уходит в провайдера (`OllamaProvider.stream_chat` для локалки или `RouterAIProvider`). Стрим SSE назад клиенту.
5. После [DONE] эмитится финальный `[META]{...}` с токенами, временем, RAG-метаданными, состоянием задачи.

## Развёртывание

- **Локально (dev)**: `uvicorn app.main:app --reload --port 8000`. `.env` с `ROUTERAI_API_KEY` (опционально, для облачного провайдера и для эмбеддингов если не используется bge-m3 локально).
- **VPS**: `docker compose -p aichat up -d --build`. Compose-сеть пиннится на `172.28.0.0/24` (gateway `172.28.0.1`), `nginx` слушает 80, проксирует на `app:8000`.
- **Локальная LLM с VPS**: SSH reverse-туннель с Mac → VPS пробрасывает Mac-овский Ollama (`11434`) на `172.28.0.1:11434` внутри docker-bridge. Контейнер `app` ходит туда по env `OLLAMA_BASE_URL`.

## Day-маркеры

В коммитах и переменных встречается `Day NN` — это шаги учебного плана. Ключевые:

- **Day 21–24** — RAG: индекс, multi-query/гибридный поиск, фильтр+реранк+rewrite, refusal порог + автогенерация Источники/Цитаты.
- **Day 25** — мини-чат с памятью задачи (goal/constraints/terms).
- **Day 27** — локальная LLM через Ollama (Vikhr-Nemo-12B-Instruct Q4_K_M).
- **Day 28** — локальные эмбеддинги (bge-m3) → RAG полностью без облака.
- **Day 29** — оптимизация под кейс (temperature presets, system-prompt ограничения, native /api/chat).
- **Day 30** — приватный сервис: nginx rate-limit (`10r/m` на `/api/chat`), осознанный `num_ctx=8192`, `num_predict=1024`.
- **Day 31** — `/help` команда: assistant отвечает по проекту через RAG + git-ветка как контекст.
