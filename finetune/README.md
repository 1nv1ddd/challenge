# Fine-tuning: код-ассистент в стиле проекта challenge

Дообучаем `gpt-4o-mini` генерировать код в конвенциях проекта (`CLAUDE.md`):
`from __future__ import annotations`, русские docstring'и, `@dataclass`-DTO,
`HTTPException` с русским detail, параметризованный SQL через `scheduler_store`,
без `print`/`Any`/голого `except`.

**Задача типа generation** (код в нашем стеке). assistant-ответ в датасете — эталонный
код в стиле проекта.

## Структура

```
finetune/
  build_dataset.py      # курируемый источник → data/raw|train|eval.jsonl
  validate.py           # валидация JSONL (JSON, 3 роли, непустой content, дубли, длины)
  baseline.py           # 10 примеров eval → базовый gpt-4o-mini без файнтюна
  run_finetune.py       # OpenAI: upload → create job → poll (dry-run по умолчанию)
  criteria.md           # критерии «стало ли лучше»
  baseline_outputs.md   # 10 baseline-ответов (человекочитаемо) + reference
  baseline_outputs.json # они же в JSON
  data/
    raw.jsonl           # все 69 примеров
    train.jsonl         # 55 (80%)
    eval.jsonl          # 14 (20%)
```

## Датасет

- **69 примеров**, формат OpenAI chat: `{"messages":[system, user, assistant]}`.
- **Реальных ≈23%** (16 шт.) — извлечены из модулей проекта (`providers.py`,
  `payloads.py`, `scheduler_store.py`, `routers/hub.py`, `agent/task_fsm.py`, тесты).
  Остальные — синтетические в том же стиле.
- Единый `system`-промпт во всех примерах закрепляет конвенции.
- Сплит 80/20 детерминированный (`random.Random(42)`).

Пересобрать датасет:

```bash
source .venv/bin/activate
python finetune/build_dataset.py
```

## Валидация

```bash
python finetune/validate.py
# ✓ train.jsonl: 55 строк, ошибок нет.
# ✓ eval.jsonl: 14 строк, ошибок нет.
```

## Baseline

10 примеров из eval через базовый `gpt-4o-mini` **без файнтюна** — точка отсчёта.
Провайдер по умолчанию `openai` (нужен `OPENAI_API_KEY`); `routerai` проксирует ту же
модель и работает на ключе из `.env` проекта.

```bash
python finetune/baseline.py --provider=routerai   # уже прогнано → baseline_outputs.md
# или официально:
OPENAI_API_KEY=sk-... python finetune/baseline.py
```

## Запуск файнтюна (подготовлено, НЕ запущено)

По умолчанию **dry-run** — печатает план, ничего не отправляет. Реальный запуск (тратит
деньги на OpenAI API) — только с `--go` и валидным `OPENAI_API_KEY`:

```bash
python finetune/run_finetune.py                    # dry-run
OPENAI_API_KEY=sk-... python finetune/run_finetune.py --go   # реальный upload+job+poll
```

Порядок в `--go`: upload `train.jsonl` и `eval.jsonl` (purpose=fine-tune) → create
fine-tuning job (base `gpt-4o-mini-2024-07-18`, eval как validation_file) → poll статуса
до `succeeded`/`failed`/`cancelled`.
