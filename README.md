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
