# AI Chat Hub

Веб-интерфейс для общения с нейросетями через бесплатные API.

## Поддерживаемые провайдеры (бесплатно)

| Провайдер | Модели | Лимиты |
|-----------|--------|--------|
| **Google Gemini** | Gemini 2.0 Flash, 1.5 Pro и др. | ~1500 запросов/день (Flash) |
| **Groq** | Llama 3.3 70B, Mixtral 8x7B и др. | ~14 400 запросов/день |
| **OpenRouter** | DeepSeek, Llama 4, Qwen 3 и др. | Зависит от модели |

## Быстрый старт (локально)

```bash
# 1. Клонировать и перейти в папку
cd challenge

# 2. Создать .env файл с ключами
cp .env.example .env
# Заполнить ключи в .env (хотя бы один провайдер)

# 3. Установить зависимости
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Запустить
uvicorn app.main:app --reload
# Открыть http://localhost:8000
```

## MCP (Model Context Protocol) — минимальный клиент

В проекте: **SDK** `mcp` в `requirements.txt`, локальный **сервер** `scripts/minimal_mcp_server.py` (stdio), **клиент** `scripts/mcp_list_tools.py` (соединение + `list_tools`).

```bash
# уже после pip install -r requirements.txt
python scripts/mcp_list_tools.py
```

Ожидается в stderr строка про успешное подключение, в stdout — список инструментов (`ping`, `echo`). JSON: `python scripts/mcp_list_tools.py --json`.

Проверка задания (автоматически):

```bash
python -m unittest tests.test_mcp_connection -v
```

Другой MCP-процесс (пример): `MCP_CMD=npx` и `MCP_ARGS='["-y","@modelcontextprotocol/server-everything"]'` перед запуском клиента (нужен Node.js).

**Tech radar (День 19):** `scripts/tech_radar_mcp_server.py` — инструменты `search` (GitHub последний релиз), `summarize`, `saveToFile`, пайплайн `run_pipeline`. Файлы: `data/tech_radar_outputs/release_watch_<owner>_<repo>.md`. Тесты: `python -m unittest tests.test_tech_radar_mcp -v` (нужен интернет).

## RAG — индексация документов (День 21)

**Откуда берётся объём «20–30+ страниц»:** при `python scripts/build_rag_index.py` в индекс попадает всё из `data/rag_corpus/*.md` **плюс по умолчанию** `README.md` (документация) и `app/agent.py` (код как текст). Основная масса — файл **`data/rag_corpus/00_polarline_handbook.md`** (~110+ КБ текста, по грубой оценке десятки страниц при ~2–3 тыс. знаков на страницу); вместе с README и `agent.py` суммарно получается **существенно больше** порога 20–30 страниц. PDF в репозитории не зашит: при необходимости добавьте `.pdf`, извлечь текст (например `pypdf`) и положите `.md`/`.txt` в `data/rag_corpus/` или расширьте скрипт сборки.

Две стратегии chunking — **fixed** (окно по символам) и **structural** (заголовки Markdown). Эмбеддинги: OpenAI-совместимый `POST /v1/embeddings` на RouterAI (`ROUTERAI_API_KEY`, модель `RAG_EMBEDDING_MODEL`, по умолчанию `openai/text-embedding-3-small`). Индекс: SQLite `data/rag_index/chunks.sqlite` (в `.gitignore`), отчёт сравнения: `data/rag_index/chunking_report.md`.

```bash
pip install -r requirements.txt
python scripts/build_rag_index.py
```

**Автосборка при старте:** если задан `ROUTERAI_API_KEY` и файла индекса нет (или таблица `chunks` пуста), при запуске uvicorn индекс соберётся сам (`RAG_AUTO_BUILD=1` по умолчанию; отключить: `RAG_AUTO_BUILD=0`). Первый старт может занять 1–3 минуты.

В UI в шапке чата выберите **RAG** → `fixed`, `structural` или `fixed vs structural` (модель получит оба набора отрывков и может кратко сравнить их по релевантности). Статус индекса: `GET /api/rag/status`. Тесты без сети: `python -m unittest tests.test_rag_chunking -v`.

**План (фазы задачи):** чекбокс **«План (фазы)»** в шапке и поле `task_workflow` в `POST /api/chat` (`true` / `false`). Пока **выкл** — в промпт не подмешивается машина состояний задачи (planning → …), не создаётся новая задача по ключевым словам и не продвигаются фазы после ответа. Пока **вкл** — прежнее поведение FSM. Если поле не передавать (старые клиенты), по умолчанию считается **включённым**.

## RAG: сравнение с/без контекста (День 22)

Пайплайн: **вопрос → семантический поиск чанков → системное сообщение с отрывками + вопрос → LLM**. Для сравнения с ответом «в лоб» без индекса:

- **Сравнение вручную:** в шапке переключите **RAG** «выкл» / `fixed` / `structural` / `fixed vs structural` и отправьте один и тот же вопрос дважды.
- **API (опционально):** `POST /api/rag/compare` — два ответа в одном запросе; для обычной работы достаточно селектора RAG.

Контрольные вопросы и ожидания: **`data/rag_eval/golden_questions.json`**. После добавления `data/rag_corpus/22_eval_knowledge.md` пересоберите индекс: `python scripts/build_rag_index.py`. Тест без сети: `python -m unittest tests.test_rag_compare -v`.

**Несколько вопросов в одном сообщении:** раньше строился один эмбеддинг на весь текст — в топ попадали в основном фрагменты под «средний» запрос (например, только PL-*). Сейчас текст **режется на подвопросы** (в т.ч. `1.?2.?` в одной строке и несколько `?` в абзаце), плюс **расширяющие запросы** по якорям (README/SQLite, `agent.py` / MCP, A.42 / NODE-Q, `22_eval`). Параллельно работает **гибридный слой**: из запроса извлекаются коды вроде `PL-xxx-RET`, `NODE-Q-*`, `_MCP_MAX_STEPS`, `22_eval` и т.д., и в выдачу подмешиваются чанки из SQLite с **точным вхождением** подстроки (регистронезависимо). Результаты **объединяются** с семантическим поиском (лучший score на `chunk_id`, до ~28 чанков). В метаданных: `rag_subqueries`, `rag_keyword_needles`.

## RAG: реранкинг и фильтрация (День 23)

Второй этап после семантического поиска и гибридных якорей:

1. **Расширенный fetch (`top_k_fetch`)** — в индекс запрашивается больше кандидатов, чем попадёт в финальный контекст (`top_k`).
2. **Порог similarity (`min_similarity`)** — отбрасываются чанки с косинусной оценкой ниже порога; чанки с keyword-boost (точное вхождение кодов) сохраняются.
3. **Лексический реранк (`rerank: lexical`)** — пересчёт score как смесь косинуса и пересечения токенов запроса с текстом чанка.
4. **Query rewrite (эвристика)** — нормализация текста и строка `[retrieval_terms]` с якорями из запроса перед построением подзапросов и эмбеддингов (без второго вызова LLM).

**UI:** чекбокс **«фильтр + rewrite»** (в метаданных ответа: **filtered / rewritten**) в шапке; состояние в `localStorage`. В тело `POST /api/chat` уходит объект `rag` с полями `query_rewrite`, `min_similarity`, `rerank`, `top_k_fetch`.

**Переменная окружения:** `RAG_MIN_SIMILARITY` — значение по умолчанию для порога, если в запросе не передан `min_similarity` (при включённом режиме День 23).

**Сравнение режимов (два ответа с RAG):** `POST /api/rag/compare_modes` — тело как у `/api/rag/compare`, плюс опционально `min_similarity` (по умолчанию `0.28`). Ответ: `with_rag_baseline` (без D23), `with_rag_enhanced` (rewrite + порог + lexical), и метаданные `rag_baseline_meta` / `rag_enhanced_meta`.

Тесты без сети: `python -m unittest tests.test_rag_postprocess -v`.

## RAG: цитаты, источники, анти-галлюцинации (День 24)

Итоговое сообщение пользователю содержит три обязательных раздела:

- `## Ответ` — формулирует модель (только по отрывкам из «Набор отрывков»);
- `## Источники` и `## Цитаты` — **подставляет сервер** из фактически найденных чанков (`chunk_id`, `source`, `section`, фрагмент текста), чтобы источники и цитаты не расходились с ретривом.

В режиме **compare** модель дописывает `## Сравнение chunking` после ответа; сервер вставляет Источники/Цитаты **между** `## Ответ` и сравнением (порядок: Ответ → Источники → Цитаты → Сравнение). Для compare-стрима ответ буферизуется до конца генерации и отдаётся одним фрагментом с правильным порядком разделов.

**Порог уверенности:** если после поиска нет якорных keyword-чанков и **максимальный семантический score** ниже порога, в контекст подставляется инструкция отказа: модель обязана начать с **«Не знаю»**, не выдумывать факты и попросить уточнение; в `## Источники` / `## Цитаты` — честные заглушки.

- Переменная окружения: **`RAG_ANSWER_MIN_SCORE`** (по умолчанию `0.25`).
- В `rag` тела запроса: **`answer_min_score`** (перекрывает env).

Режим **compare** (fixed vs structural): отказ только если **оба** набора отрывков слабые.

Примеры **10 вопросов** для ручной проверки в чате: **`data/rag_eval/day24_questions.json`**.

## Деплой на VPS (Docker)

```bash
# 1. На VPS установить Docker и Docker Compose
# Ubuntu/Debian:
sudo apt update && sudo apt install -y docker.io docker-compose-plugin

# 2. Скопировать проект на VPS
scp -r . user@your-vps-ip:/opt/aichat

# 3. На VPS:
cd /opt/aichat
cp .env.example .env
nano .env  # заполнить ключи

# 4. Запуск
docker compose up -d --build

# Сайт будет доступен на http://your-vps-ip
```

## Получение API ключей

### Google Gemini (рекомендую начать с него)
1. Перейти на https://aistudio.google.com/apikey
2. Нажать "Create API Key"
3. Скопировать ключ в `.env`

### Groq
1. Перейти на https://console.groq.com/keys
2. Создать ключ
3. Скопировать в `.env`

### OpenRouter
1. Перейти на https://openrouter.ai/keys
2. Создать ключ
3. Скопировать в `.env`

## Добавление HTTPS (опционально)

Для HTTPS на VPS можно использовать Caddy вместо Nginx. Заменить в `docker-compose.yml`:

```yaml
services:
  app:
    build: .
    env_file: .env
    restart: unless-stopped
    expose:
      - "8000"

  caddy:
    image: caddy:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
    depends_on:
      - app
    restart: unless-stopped

volumes:
  caddy_data:
```

Создать `Caddyfile`:
```
your-domain.com {
    reverse_proxy app:8000
}
```

<-claude проверка AI-ревью Day 32 -->
