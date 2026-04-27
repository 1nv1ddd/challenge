# AI Chat Hub — API-карта

Базовый префикс — пустой. Все маршруты определены в `app/routers/`.

## Чат и модели

| Метод | Путь | Что |
|---|---|---|
| `GET` | `/` | Статика — SPA из `static/index.html` |
| `GET` | `/api/models` | Список моделей по провайдерам: `{routerai: [...], ollama: [...]}`. Для Ollama тянет `/api/tags` живьём, исключая embedding-модели |
| `POST` | `/api/chat` | SSE-стрим ответа модели. Body: `provider`, `model`, `conversation_id`, `messages`, `temperature`, `profile_id`, `context_strategy` (`sliding`/`branching`/`facts`), `branch_id`, `resume`, `rag` (`{enabled, strategy, top_k, ...}`), `task_workflow`. Стрим: `data: "tok"`, ..., `data: [META]{...}`, `data: [DONE]` |

## Память диалога

| Метод | Путь | Что |
|---|---|---|
| `GET` | `/api/memory/{conversation_id}` | Слои памяти: short_term (sliding window), working_memory (per-conv), long_term (global), invariants |
| `GET` | `/api/memory/{conversation_id}/branches` | Список веток |
| `POST` | `/api/memory/{conversation_id}/checkpoint` | Снапшот текущей ветки → checkpoint |
| `POST` | `/api/memory/{conversation_id}/branch` | Создать ветку из checkpoint'а |
| `GET/POST` | `/api/profiles[/{id}]` | Профили стиля (style/format/constraints) |

## RAG

| Метод | Путь | Что |
|---|---|---|
| `GET` | `/api/rag/status` | Состояние индекса: путь, `indexed`, статистика по стратегиям |
| `POST` | `/api/rag/compare` | Сравнение ответов LLM **без RAG vs с RAG** (Day 22) |
| `POST` | `/api/rag/compare-modes` | Сравнение «базовый RAG» vs «фильтр+реранк+rewrite» (Day 23) |

## Планировщик и MCP

| Метод | Путь | Что |
|---|---|---|
| `GET` | `/api/scheduler/tasks` | Список задач планировщика (SQLite в `data/mcp_scheduler.sqlite`) |
| `POST` | `/api/scheduler/tasks` | Создать задачу |
| `GET` | `/api/scheduler/stream` | SSE-канал тиков планировщика → UI инжектит системные сообщения в активный чат |
| `GET/POST` | `/api/mcp/*` | Подключение/отключение MCP-серверов через stdio, список tools, вызовы |

## Провайдеры

`RouterAIProvider` бьёт в `https://routerai.ru/api/v1/chat/completions` (OpenAI-совместимо). Стрим — нативный SSE или fallback на не-стрим.

`OllamaProvider` бьёт в нативный `${OLLAMA_BASE_URL}/api/chat` с `options.num_ctx` (по умолчанию 8192) и `options.num_predict` (1024). Стрим — line-delimited JSON, парсится в обычный SSE для клиента.

## Конфигурация (env)

| Переменная | Назначение | Дефолт |
|---|---|---|
| `ROUTERAI_API_KEY` | Ключ облачного провайдера | (нет) |
| `OLLAMA_BASE_URL` | Endpoint Ollama | `http://localhost:11434` |
| `OLLAMA_NUM_CTX` | Контекст модели в токенах | `8192` |
| `OLLAMA_NUM_PREDICT` | Лимит на ответ | `1024` |
| `RAG_EMBEDDINGS_URL` | Эндпоинт эмбеддингов (RouterAI или Ollama) | RouterAI |
| `RAG_EMBEDDING_MODEL` | Модель эмбеддингов | `openai/text-embedding-3-small` |
| `RAG_AUTO_BUILD` | Перебилд индекса при старте | `0` (на VPS), любое `1` локально |
| `RAG_MIN_SIMILARITY` | Порог отсечения (Day 23) | `0` |
| `RAG_ANSWER_MIN_SCORE` | Порог refusal (Day 24) | `0.25` |
