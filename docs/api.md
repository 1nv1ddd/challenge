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

## Уверенность и маршрутизация

| Метод | Путь | Что |
|---|---|---|
| `POST` | `/api/triage` | Триаж обращения через гейт уверенности (Day 7). Body: `provider`, `model`, `text`, `samples`, `temperature`. Ответ: решение, статус `OK`/`UNSURE`/`FAIL`, confidence, метрики |
| `POST` | `/api/route` | Ответ через каскад моделей (Day 8). Body: `question`, опционально `provider`, `small_model`, `large_model`, `temperature`. Ответ: `answer`, `tier`, `escalated`, `escalation_reason`, `preroute`, `attempts`, `metrics` |
| `POST` | `/api/intake` | Разбор письма-заявки (Day 9). Body: `letter`, опционально `mode` (`staged` / `mono` / `staged_rules`), `today` (`YYYY-MM-DD`), `provider`, `mono_model`, `stage_models`, `temperature`. Ответ: `fields`, `decision`, `reply`, `stages`, `metrics`, `ok` |
| `POST` | `/api/intent` | Классификация интента обращения через micro-model с fallback на LLM (Day 10). Body: `text`, опционально `strategy` (`micro_embed_first` / `micro_tfidf_first` / `micro_only` / `llm_only`), `provider`, `llm_model`, `temperature`. Ответ: `label`, `source`, `micro` (score, статус, соседи), `llm`, `metrics` |
| `GET` | `/api/intent/bank` | Состав банка примеров уровня 1: сколько примеров на метку, известные метки и стратегии |

## Безопасность промптов

| Метод | Путь | Что |
|---|---|---|
| `POST` | `/api/redteam` | Прогон корпуса prompt injection по версиям system-промпта (Day 11). Body: опционально `versions` (`["v1","v2"]`), `ids`, `target` (`bank` / `support`), `vector` (`direct` / `indirect` / `jailbreak`), `technique`, `model`, `provider`, `temperature`. Ответ: `runs` (по версии: `broken`, `held`, `break_rate`, `by_vector`, вердикты с сигналами) и `diff` (`fixed` / `still_broken` / `regressed`) |
| `GET` | `/api/redteam/corpus` | Корпус инъекций с классификацией: вектор, техника, цель, источник и разбор «почему работает / как защититься» |
| `POST` | `/api/indirect` | Прогон ловушек непрямой инъекции по наборам слоёв защиты (Day 12). Body: опционально `presets` (`none` / `sanitize` / `boundary` / `guard` / `all`), `ids`, `scenario` (`summarize` / `analyze` / `search`), `source` (`email` / `document` / `webpage`), `hiding`, `model`, `provider`, `temperature`. Ответ: `runs` (по пресету: `injected`, `blocked`, `broke_usefulness`, результаты с находками guard) и `effect` (`fixed` / `still_injected` / `usefulness_lost`) |
| `GET` | `/api/indirect/corpus` | Корпус ловушек: носитель, техника сокрытия, сценарий агента, длина документа против видимой части и разбор каждой |

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
