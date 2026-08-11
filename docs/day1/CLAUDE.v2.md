# Проект: AI Chat Hub (challenge)

FastAPI-приложение с агентом, RAG, MCP-серверами и планировщиком.
Стек: Python 3.12, FastAPI + Uvicorn, httpx, MCP SDK, numpy, nginx (reverse proxy), Docker Compose.

## Структура

- `app/` — FastAPI-приложение: `main.py`, `agent/`, `rag/`, `routers/`, `scheduler_*`, `mcp_*`.
- `scripts/` — standalone MCP-серверы и `build_rag_index.py`.
- `static/` — фронт.
- `data/` — runtime-файлы (память, SQLite, RAG-индекс). Почти всё в `.gitignore`.
- `tests/` — тесты.
- `Dockerfile`, `docker-compose.yml`, `nginx.conf` — деплой.

## Локальный запуск

Проект использует `.venv` в корне (Python 3.12) и `.env` с ключами.

```bash
# активировать venv (зависимости уже стоят; если нет — pip install -r requirements.txt)
source .venv/bin/activate

# запустить dev-сервер
uvicorn app.main:app --reload --port 8000
```

Открыть [http://localhost:8000/](http://localhost:8000/).

`.env` нужен минимум с `ROUTERAI_API_KEY`. Образец — `.env.example`. В git `.env` не коммитим (есть в `.gitignore`).

### Правило: перезапуск после правок

**После каждой правки кода локальный сервер должен работать на свежей версии.**

- Запускаем uvicorn с флагом `--reload` — он сам пересобирает процесс, когда меняется любой `.py` в `app/` или `scripts/`. Этого достаточно для большинства правок (Python, Jinja-шаблоны при их появлении).
- Правки в `static/` (HTML/JS/CSS) — `--reload` их не подхватывает, достаточно reload страницы в браузере (Cmd+R). Если кэш — Shift+Cmd+R.
- Правки в `.env` или `requirements.txt` — `--reload` **не** перечитывает окружение и не ставит зависимости. Нужно остановить сервер (Ctrl+C в терминале или `kill` процесса uvicorn) и запустить заново; при смене зависимостей — сначала `pip install -r requirements.txt`.
- Правки в корпусе RAG (`data/rag_corpus/*`) — нужно пересобрать индекс: `python scripts/build_rag_index.py`, сервер подхватит новый `chunks.sqlite` на следующем запросе.

Рабочий процесс Claude по этому проекту: после каждой правки — убедиться, что dev-сервер запущен с `--reload`; если был остановлен, поднять заново (`uvicorn app.main:app --reload --port 8000`, желательно в background); при правках вне зоны `--reload` — полный рестарт. В конце — выдать пользователю короткий промпт для проверки в веб-чате `http://localhost:8000/`.

Проверка, что сервер поднят и обслуживает запросы:

```bash
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8000/
```

Вариант через Docker локально (используем только для проверки деплой-конфига, не для итеративной разработки):

```bash
docker compose up --build
```

Откроется на `http://localhost/` (через nginx на 80-м).

## Git workflow

Remote: `git@github.com:1nv1ddd/challenge.git`, основная ветка — `main`.

**Коммит и push делаются ТОЛЬКО по явной команде пользователя** ("закоммить", "запушь", "коммит+пуш"). Никогда не коммитить проактивно.

Стиль коммит-сообщений (из истории):
- Учебные шаги: `Day NN: <тема>` или `feat(<scope>): Day NN <...>` (например, `Day 24: server-side sources/quotes, refusal threshold, compare stream order`).
- Фичи: `feat(scope): ...`
- Рефакторы: `Refactor <area>, ...`
- Правки: `Fix ...`

Номер дня — спросить у пользователя, если неочевидно по последнему коммиту (`git log --oneline -1`).

Формат коммита — через HEREDOC с `Co-Authored-By: Claude ...` (см. общие правила).

## Деплой на VPS

**VPS:** `root@138.124.86.111` (SSH по ключу, без пароля).
**Путь на сервере:** `/root/challenge`.
**Docker Compose project name:** `aichat` (не `challenge`). Контейнеры: `aichat-app-1`, `aichat-nginx-1`.
**Публичный URL:** `http://138.124.86.111/`.

Деплой запускается **только по команде пользователя** ("задеплой", "деплой"). Порядок:

1. Убедиться что последние изменения запушены в `main` на GitHub.
2. Зайти по SSH и обновить репозиторий + пересобрать контейнер:

```bash
ssh root@138.124.86.111 'git config --global --add safe.directory /root/challenge && \
  cd /root/challenge && \
  git fetch origin && \
  git reset --hard origin/main && \
  docker compose -p aichat up -d --build'
```

Пояснения:
- `safe.directory` — нужно из-за того что рабочая копия на VPS имеет ownership `501:staff`, а git запускается из-под `root`. Без этого git ругается "dubious ownership".
- `git reset --hard origin/main` — на VPS исторически есть локальные правки и untracked-файлы, `git pull` на них падает. Сбрасываем рабочую копию жёстко к удалённому `main`. **Перед этим убедиться, что пользователь подтвердил деплой** — это деструктивная операция для серверной рабочей копии (но не для кода в репо и не для `.env`, т.к. он в `.gitignore`).
- `-p aichat` — сохраняем существующие имена контейнеров/сети. Без флага compose возьмёт имя по папке (`challenge`) и создаст параллельный стек.
- `--build` — пересобираем image `aichat-app` из нового кода.

### Git remote на VPS: HTTPS, не SSH

На VPS **нет** SSH-ключа, привязанного к GitHub (`~/.ssh/` содержит только `authorized_keys` и `known_hosts`). Поэтому `git fetch` через `git@github.com:...` падает с `Permission denied (publickey)`.

Репозиторий `1nv1ddd/challenge` — публичный, поэтому используем HTTPS-remote, он работает без аутентификации на чтение. Один раз уже переключено командой:

```bash
ssh root@138.124.86.111 'cd /root/challenge && git remote set-url origin https://github.com/1nv1ddd/challenge.git'
```

Проверка перед деплоем (если вдруг кто-то снова переключил на SSH):

```bash
ssh root@138.124.86.111 'cd /root/challenge && git remote -v'
# должно быть: origin https://github.com/1nv1ddd/challenge.git
```

Если видишь `git@github.com:...` — сначала переключи на HTTPS, потом деплой. НЕ пытаться завести SSH-ключ на VPS без явной просьбы пользователя.

Дополнительно: при первом деплое после сброса `known_hosts` на VPS может потребоваться `ssh-keyscan` для github.com, но при HTTPS-remote это уже не нужно.

### Health-check после деплоя

```bash
ssh root@138.124.86.111 'cd /root/challenge && docker compose -p aichat ps && \
  curl -s -o /dev/null -w "HTTP %{http_code} t=%{time_total}s\n" http://localhost/'
curl -s -o /dev/null -w "external HTTP %{http_code}\n" http://138.124.86.111/
```

Ожидаем: оба контейнера `Up`, HTTP 200 внутри и снаружи.

### Что НЕ трогать на VPS

- `.env` в `/root/challenge/.env` — лежит отдельно, ключи настоящие. В git его нет (`.gitignore`), `git reset --hard` его не тронет. Не перезаписывать, не удалять.
- `data/` — runtime-данные (память, RAG-индекс, SQLite планировщика). Смонтирована в контейнер как volume (`./data:/app/data`). Не чистить без прямой команды.

### Откат

Если задеплоили что-то поломанное:

```bash
ssh root@138.124.86.111 'cd /root/challenge && \
  git reset --hard <предыдущий-sha> && \
  docker compose -p aichat up -d --build'
```

SHA предыдущего рабочего коммита взять из `git log` перед деплоем (сохранить в уме/заметке).

## Важные отличия local ↔ VPS

- Локально — venv + `uvicorn --reload`. На VPS — Docker Compose + nginx на 80.
- Имя compose-проекта на VPS: `aichat`. Локально можно любое, но если запускаешь `docker compose` — имена контейнеров будут другие.
- На VPS git-директория требует `safe.directory` из-за mismatched ownership.

---

# Стиль кода (как ХОЧУ, чтобы писалось)

> Эта часть — про то, как выглядит код. Всё ниже вытащено из реальных модулей проекта.
> Пиши так, будто дописываешь в уже существующий файл, а не создаёшь проект с нуля.

## Инвариант №1 — сначала переиспользуй, потом пиши

**Перед тем как написать новую функцию/хелпер — проверь, нет ли уже такого в модуле.**
Не плоди дубликаты. Пример реальной ошибки: писать `list_scheduler_jobs()`, когда в
`app/scheduler_store.py` уже есть `list_jobs()`. Если нужен подмножество полей —
переиспользуй существующую функцию и спроецируй результат в обработчике, а не дублируй
доступ к БД.

Порядок действий для любой правки:
1. `Grep`/`Read` по модулю, куда пишешь — есть ли похожий хелпер/паттерн?
2. Есть — переиспользуй или расширь его.
3. Нет — пиши новый, повторяя стиль соседних функций в этом же файле.

## Naming conventions

- Модули, функции, переменные — `snake_case`. Классы — `PascalCase`.
- Приватные функции/атрибуты — с `_` в начале: `_connect()`, `_db_file()`, `_normalize_stream_content()`.
- Константы модуля — `UPPER_SNAKE`, приватные — с `_`: `_TASK_ID_RE`, `_ALLOWED_TYPES`, `TASK_PHASES`.
- Классы-миксины агента — суффикс `Mixin`: `AgentTaskFsmMixin`.
- Провайдеры/сервисы — суффикс по роли: `RouterAIProvider`, `OllamaProvider` (базовый `AIProvider(ABC)`).
- Роутеры именуем `router`, эндпоинт-функции — по действию: `list_models`, `create_branch`, `rag_status`.

## Обязательные паттерны

- **Всегда** первой строкой кода модуля (после докстринга) — `from __future__ import annotations`.
- **Однострочный docstring на русском** в начале каждого модуля: что он делает.
- Современные type hints: `str | None`, `list[dict]`, `AsyncIterator[StreamResult]`. Возвраты аннотируем.
- DTO — через `@dataclass` (`Message`, `StreamResult`), а не голые dict в сигнатурах.
- Интерфейсы — `ABC` + `@abstractmethod`.
- FastAPI-роут: `APIRouter(prefix=..., tags=[...])`, `async def`, ошибки через
  `HTTPException(status_code=..., detail="...")` с **русским** сообщением.
- Разбор тела запроса — через payload-класс с `.from_body(body)`, а не ручной разбор dict в обработчике
  (см. `app/payloads.py`).
- Исключения ловим **точечно** и пробрасываем причину:
  `except ValueError as exc: raise HTTPException(..., detail=str(exc)) from exc`.
- Доступ к SQLite планировщика — только через функции `app/scheduler_store.py`
  (`_connect()`, `init_schema()`, `list_jobs()` …), напрямую `sqlite3.connect` в роутах/агенте не лезем.
- Константы (фазы, edges, типы задач) — в `app/agent_constants.py`, не хардкодим по месту.

## 3–5 примеров хорошего кода (эталон из проекта)

1. **DTO + интерфейс провайдера** — `app/providers.py:13-33`: `@dataclass Message/StreamResult`,
   `AIProvider(ABC)` с `@abstractmethod async def stream_chat(...) -> AsyncIterator[StreamResult]`.
2. **FastAPI-роут с обработкой ошибок** — `app/routers/hub.py:59-85` (`rag_compare`): payload через
   `.from_body`, ранняя валидация → `HTTPException(400/503, detail="русский текст")`,
   `except ValueError as exc: ... from exc`.
3. **SSE-стрим** — `app/scheduler_routes.py:18-32`: `StreamingResponse` + async-генератор,
   `media_type="text/event-stream"`, заголовки `Cache-Control: no-store`, `X-Accel-Buffering: no`.
4. **Явный FSM без магии** — `app/agent/task_fsm.py:37-47`, `_illegal_transition_message`: переходы
   валидируются по `TASK_ALLOWED_EDGES` из констант, ошибка — человекочитаемая строка.
5. **Слой хранилища** — `app/scheduler_store.py:25-56`: `_connect()` c `row_factory = sqlite3.Row`,
   идемпотентный `init_schema()`, параметризованный SQL (никаких f-string в запросах).
6. **Тест** — `tests/test_scheduler_store.py`: `unittest.TestCase`, изоляция БД через
   `tempfile` + `SCHEDULER_SQLITE_PATH` + `importlib.reload`, `setUp/tearDown`.

## 3–5 антипаттернов (ЗАПРЕЩЕНО)

1. **Дубликат существующего** — новая функция, повторяющая уже имеющуюся (см. Инвариант №1). Сначала ищи.
2. **Голый `except:`** или `except Exception` без нужды. Ловим конкретный тип
   (`json.JSONDecodeError`, `ValueError`, `httpx.HTTPError`). В проекте голого `except:` нет — не заводи.
3. **`print()` в request-пути / коде приложения.** Вывод в stdout запрещён; в CLI-скриптах — только
   `print(..., file=sys.stderr)` (как `app/rag/build_index.py`).
4. **`Any` в аннотациях.** Уточняй тип (`dict`, `list[dict]`, конкретный класс). `Any` — только когда
   тип реально приходит извне и иначе никак.
5. **Прямой `sqlite3.connect` мимо `scheduler_store`** и **f-string/конкатенация в SQL** (риск инъекции) —
   только параметризованные запросы через существующие функции стора.
6. **Забыть `from __future__ import annotations`** или русский docstring модуля.

## Шаблон типичного модуля

```python
"""Однострочное описание модуля на русском: что он делает."""

from __future__ import annotations

# 1) stdlib
import json
import time

# 2) сторонние
import httpx

# 3) внутренние (относительные импорты)
from ..agent_constants import TASK_PHASES

# 4) приватные константы модуля
_SOME_RE = re.compile(r"...")

# 5) публичный API: dataclass'ы / функции / классы
@dataclass
class Result:
    text: str | None = None


def do_something(arg: str) -> Result:
    """Что делает функция (если неочевидно из имени)."""
    ...
```

Порядок внутри файла: docstring → `from __future__` → импорты (stdlib → сторонние → внутренние) →
приватные константы/хелперы (`_name`) → публичные функции/классы.
