# Day 5 — Пул задач execution loop

**Дата:** 2026-07-24. Челлендж advance, неделя 8. Base-SHA: `c4c0234`.

Трекер — реальные GitHub Issues в `1nv1ddd/challenge` (+ машиночитаемое зеркало [`backlog.json`](backlog.json)).
Профиль выбирается автоматически по типу задачи (см. [`loop-protocol.md`](loop-protocol.md)):

- `bug` → **bugfix**
- `test` → **testgen**
- `research` → **research**
- `refactor` → **orchestrator-inline**
- `feature` → **orchestrator-inline**
- `docs` → **orchestrator-inline**

**Состав (18):** 2×bug, 3×docs, 2×feature, 4×refactor, 1×research, 6×test.

| # | ID | Тип | Профиль | Задача | Критерий «сделано» | Issue |
|---|----|-----|---------|--------|--------------------|-------|
| 1 | T01 | test | testgen | Покрыть GET /api/models через TestClient | Новый тест-файл с TestClient на /api/models; весь сьют python -m unittest discover -s tests зелёный. | [#3](https://github.com/1nv1ddd/challenge/issues/3) |
| 2 | T02 | test | testgen | Регрессионный тест: POST /api/branches → 400 без checkpoint_id | Тест TestClient утверждает status_code==400 и русский detail; сьют зелёный. | [#4](https://github.com/1nv1ddd/challenge/issues/4) |
| 3 | T03 | bug | bugfix | POST /api/checkpoints падает 500 на не-объектном JSON теле | Эндпоинт не отдаёт 500 на не-объектном теле (мягкая деградация к {} по образцу chat/rag_compare); добавлен регрессионный тест, показан красным до фикса; весь сьют зелёный. | [#5](https://github.com/1nv1ddd/challenge/issues/5) |
| 4 | T04 | refactor | orchestrator-inline | create_checkpoint/create_branch → payload-класс .from_body | Новые payload-классы в app/payloads.py; оба эндпоинта используют .from_body; поведение (400 без checkpoint_id) сохранено; сьют зелёный. | [#6](https://github.com/1nv1ddd/challenge/issues/6) |
| 5 | T05 | test | testgen | task_fsm: нелегальный переход planning→execution | Тест утверждает ok==False и наличие 'Illegal transition' в error; сьют зелёный. | [#7](https://github.com/1nv1ddd/challenge/issues/7) |
| 6 | T06 | feature | orchestrator-inline | GET /api/scheduler/jobs — список задач планировщика | Эндпоинт возвращает 200 и список задач из list_jobs(); прямого sqlite3 в роуте нет; сьют зелёный; ручная проверка TestClient/curl 200. | [#8](https://github.com/1nv1ddd/challenge/issues/8) |
| 7 | T07 | test | testgen | Покрыть GET /api/rag/status через TestClient | Тест TestClient на /api/rag/status; сьют зелёный. | [#9](https://github.com/1nv1ddd/challenge/issues/9) |
| 8 | T08 | bug | bugfix | POST /api/profiles падает 500 на не-объектном JSON теле | Эндпоинт не отдаёт 500 на не-объектном теле; регрессионный тест показан красным до фикса; весь сьют зелёный. | [#10](https://github.com/1nv1ddd/challenge/issues/10) |
| 9 | T09 | refactor | orchestrator-inline | upsert_profile → payload-класс .from_body | ProfilePayload в app/payloads.py; эндпоинт через .from_body; сьют зелёный. | [#11](https://github.com/1nv1ddd/challenge/issues/11) |
| 10 | T10 | docs | orchestrator-inline | Актуализировать docs/api.md под реальные маршруты | Каждый путь в docs/api.md соответствует реально существующему роуту (проверка grep путей против app/); нет упоминаний несуществующих /api/scheduler/tasks и /api/rag/compare-modes. | [#12](https://github.com/1nv1ddd/challenge/issues/12) |
| 11 | T11 | test | testgen | HTTP happy-path: POST /api/checkpoints → GET /api/branches | Тест TestClient проходит связку и утверждает наличие данных; сьют зелёный. | [#13](https://github.com/1nv1ddd/challenge/issues/13) |
| 12 | T12 | refactor | orchestrator-inline | set_invariants/update_task_state → payload-классы .from_body | Новые payload-классы; оба эндпоинта через .from_body; сьют зелёный. | [#14](https://github.com/1nv1ddd/challenge/issues/14) |
| 13 | T13 | feature | orchestrator-inline | GET /api/scheduler/jobs/{task_id} — одна задача или 404 | 200 для существующего task_id, 404 для отсутствующего; прямого sqlite3 нет; сьют зелёный. | [#15](https://github.com/1nv1ddd/challenge/issues/15) |
| 14 | T14 | refactor | orchestrator-inline | Reuse-first: вынести guard `as_dict(body)` в payloads | as_dict определён один раз в payloads.py и используется в ≥3 местах; ручных повторов guard в hub.py не осталось; сьют зелёный. | [#16](https://github.com/1nv1ddd/challenge/issues/16) |
| 15 | T15 | docs | orchestrator-inline | Добавить русские модульные докстринги в 12 модулей | Каждый app/*.py (кроме __init__) начинается с русского docstring; контрольный grep пустой; сьют зелёный. | [#17](https://github.com/1nv1ddd/challenge/issues/17) |
| 16 | T16 | docs | orchestrator-inline | ADR 0002: конвенция payload-классов для тел запросов | docs/adr/0002-payload-classes.md существует с секциями Status/Context/Decision/Consequences и ссылкой на app/payloads.py. | [#18](https://github.com/1nv1ddd/challenge/issues/18) |
| 17 | T17 | research | research | Какие эндпоинты всё ещё без HTTP-тестов | docs/day5/coverage-note.md с таблицей эндпоинт → покрыт/не покрыт и ссылками на тесты; код не менялся. | [#19](https://github.com/1nv1ddd/challenge/issues/19) |
| 18 | T18 | test | testgen | Покрыть GET /api/mcp/status через TestClient | Тест TestClient на /api/mcp/status; сьют зелёный. | [#20](https://github.com/1nv1ddd/challenge/issues/20) |
