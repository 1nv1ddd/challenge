# Прогон профиля `testgen` — 22.07.2026

**Профиль:** [`.claude/agents/testgen.md`](../../.claude/agents/testgen.md)
**Вход:** «найди сам непокрытое в `app/` и закрой, минимум 3 файла» + две находки других профилей
(см. ниже). Что именно покрывать — решал агент.

## Итог

| | было | стало |
|---|---|---|
| тестов в сьюте | 50 | **121** |
| файлов тестов | 6 | 9 (+1 усилен) |
| упало | — | **0** |

Проверено main-агентом независимо: `python -m unittest discover -s tests` → `Ran 121 tests ... OK
(skipped=1)`; `git diff --stat -- app/` показывает **только** предсуществующую правку Day 2
(`hub.py`, 1 строка) — все диверсии откачены.

## Что выбрал покрывать

Инвентаризация показала: тесты дёргали `app.rag.*`, `app.mcp_*`, `scheduler_store` и
`SimpleChatAgent.stream_reply`, а вне покрытия целиком оставались ядро агента, слой разбора тел
запросов и валидация роутов.

| Файл | Тестов | Цель и почему она |
|---|---|---|
| [`tests/test_task_fsm.py`](../../tests/test_task_fsm.py) | 26 | `app/agent/task_fsm.py` — 255 строк, 0 упоминаний в тестах. Правила «без скачков по фазам», «`done` терминальна», «не аппрувить план самому» держались только на договорённостях в коде и ломались бы молча |
| [`tests/test_payloads_api.py`](../../tests/test_payloads_api.py) | 19 | `app/payloads.py` + валидация `hub.py` — через `from_body` проходит каждый запрос. Хрупкое место: `task_workflow=None if raw_tw is None else bool(...)` — различие «не задано» и `False`, которое легко «упростить» до `bool()` |
| [`tests/test_agent_memory_layers.py`](../../tests/test_agent_memory_layers.py) | 23 | чекпойнты/ветки/инварианты/профили + санитайзинг long-term памяти (фильтр, выбрасывающий случайно затянутые секреты) |
| [`tests/test_branch_api.py`](../../tests/test_branch_api.py) | 1 → 4 | усилен по заданию, см. ниже |

Отбросил осознанно: `streaming.py` (633 строки, нужен связный набор фейков провайдера + MCP + RAG),
`scheduler_notify.py` (SSE + глобальный луп — тест вышел бы про моки, а не про поведение),
`providers.py` (только сеть).

## Кросс-вход от других профилей

Задание собрано не мной с нуля, а из находок двух других прогонов:

1. **От профиля `review`** (Day 2): ассерт `assertNotIn("branch_id", r.json())` в
   `tests/test_branch_api.py:27` при ответе 400 **тавтологически истинен** — тело ошибки и так не
   содержит этого ключа, тест зелёный при любом поведении кода. Профиль `testgen` по своему
   контракту не имеет права трогать существующие тесты, поэтому разрешение на правку именно этого
   файла выдано во входе явно, как исключение.
2. **От профиля `smoke`** (сегодня): негативный путь создания ветки без чекпойнта **недостижим
   через UI** — фронт сам шлёт `POST /api/checkpoints` первым. Значит контракт
   `POST /api/branches` без `checkpoint_id` → 400 может защитить **только** тест уровня API.

## Доказательство, что тесты живые

Требование профиля: для каждого файла внести временную диверсию в код, показать красный тест,
вернуть как было. 11 диверсий, все откачены (md5 сверен с бэкапом).

| Диверсия | Реакция |
|---|---|
| `task_fsm.py:166` — проверку легальности перехода в `if False:` | 3 FAIL, в т.ч. `test_phase_skip_is_rejected_and_state_unchanged` |
| `task_fsm.py:241` — убрана ветка «остаёмся в planning, пока план не утверждён» | `test_turn_does_not_auto_approve_plan`: `'plan_approved' != 'planning'` |
| `task_fsm.py:55` — убрано `"не утверждаю"` из списка отказов | `test_rejection_wording_does_not_promote` |
| `payloads.py:41` → `task_workflow=bool(raw_tw)` | `test_task_workflow_false_is_not_confused_with_unset`: `False is not None` |
| `hub.py:66` — убрано `or not rc.message` | `test_empty_message_returns_400_and_skips_agent`: `200 != 400` |
| `hub.py:82` — `400` → `503` для `LookupError` | `test_missing_index_is_reported_as_400` |
| `normalize.py:271` — снят фильтр `LONG_TERM_ALLOWED_KEYS` | `test_long_term_memory_is_sanitized_on_load`: `'api_key' unexpectedly found` |
| `normalize.py:307` — убран срез `[-INVARIANTS_MAX_ITEMS:]` | `test_item_limit_keeps_last_items`: `35 != 30` |
| `memory_branches.py:141` — вместо `raise ValueError` пустой чекпойнт | `test_branch_from_unknown_checkpoint_raises` |
| `hub.py:141` — возврат к историческому `return {"error": ...}` c HTTP 200 | оба теста на 400 краснеют |
| **тонкая:** валидация после мутации — перед `raise HTTPException(400)` создать чекпойнт и ветку | см. ниже |

**Последняя диверсия — главное доказательство дня.** Она имитирует правдоподобную регрессию:
статус остаётся 400, тело ошибки по-прежнему без `branch_id` — то есть **старый тавтологичный
ассерт остался бы зелёным**, пропустив реальную порчу данных. Новый тест краснеет:
`Lists differ: ['main', 'branch-1'] != ['main']`. Это и показывает, что усиление ассерта
(проверять отсутствие побочного эффекта, а не отсутствие ключа в теле ошибки) — не косметика.

## Найденные дефекты (не чинил)

1. **`ValueError` из агента не мапится в HTTP-ошибку.** `hub.py:143` вызывает `create_branch`,
   который при несуществующем `checkpoint_id` кидает `ValueError` (`memory_branches.py:141`) —
   ветки `except ValueError` нет, клиент получит **500 вместо 400/404**. То же для
   `POST /api/checkpoints` с чужим `branch_id` и `POST /api/profiles` с пустым `profile_id`.
   CLAUDE.md прямо предписывает `except ValueError as exc: raise HTTPException(...) from exc`.
   Тестом на уровне HTTP не закреплял — «не цементировать баг».
2. **Мёртвая ветка 503:** `if not providers` в `hub.py:62` и `:91` недостижима — `_build_providers()`
   всегда кладёт `ollama` (`bootstrap.py:26`).
3. **Голый `except Exception`** в `scheduler_notify.py:39` и `scheduler_store.py:151` —
   антипаттерн №2 из CLAUDE.md.
4. Наблюдение: `POST /api/task-state` при нелегальном переходе отдаёт **HTTP 200** с
   `{"ok": false}`, в отличие от остальных роутов. Похоже на осознанный выбор ради UI —
   закрепил как есть.

## Чего не покрыл

`streaming.py`, SSE-эндпоинты (`/api/chat`, `/api/scheduler/stream` — нужен рефакторинг с
инъекцией брокера), `providers.py` и `rag/embeddings.py` (только сеть).
Реальный незакрытый кандидат на следующий шаг — валидация `scheduler_store.register_job/delete_job`
(некорректный `task_id`, неизвестный `task_type`, клэмп интервала): логика изолирована и легко
тестируется по образцу `tests/test_scheduler_store.py`.
