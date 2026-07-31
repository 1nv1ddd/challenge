"""День 9 (advance): команда `/intake` — разбор письма-заявки с карточкой этапов инференса.

Работает как `/triage` и `/route`: перехватываем префикс в последней реплике пользователя.
В чат уходит готовый ответ клиенту и под ним разбор — что извлекли, как решили и во что это обошлось.
"""

from __future__ import annotations

import re

from ..agent_constants import INTAKE_FIELD_KEYS
from ..staged import IntakeResult

_PREFIX = "/intake"
_MODE_ALIASES = {"mono": "mono", "staged": "staged", "rules": "staged_rules"}
_TODAY_RE = re.compile(r"^today\s*=\s*(\d{4}-\d{2}-\d{2})$", re.I)
_DECISION_MARK = {"accept": "🟢", "clarify": "🟡", "reject": "🔴"}
_FIELD_TITLE = {
    "company": "Организация",
    "product": "Товар",
    "qty_kg": "Объём, кг",
    "budget_rub": "Бюджет, ₽",
    "deadline": "Срок",
    "region": "Регион",
    "contact": "Контакт",
    "payment": "Оплата",
}
_STAGE_TITLE = {
    "normalize": "1. Нормализация входа",
    "decide": "2. Решение по политике",
    "compose": "3. Формирование ответа",
    "monolithic": "Один большой запрос",
}
# Решение, принятое без вызова модели: в таблице этапов это отдельная строка с нулевой ценой.
_SOURCE_NOTE = {
    "rules": "политика применена кодом",
    "rules_fallback": "этап 2 сорвался, политика применена кодом",
}
_USAGE = (
    "### `/intake` — разбор письма-заявки\n\n"
    "Вставьте письмо клиента после команды:\n\n"
    "```\n/intake Нужны трубы 1,5 т в Екатеринбург до 20 августа, бюджет 400к, +7 999 123-45-67\n```\n\n"
    "Режим задаётся первым словом: `staged` (по умолчанию, три этапа), `mono` "
    "(один большой запрос), `rules` (этап решения считается кодом). "
    "Дату обращения можно зафиксировать через `today=2026-07-30` — от неё считаются сроки.\n\n"
    "```\n/intake mono today=2026-07-30 Добрый день! Нужен лист...\n```"
)


def detect_intake_command(text: str) -> tuple[bool, str, str, str]:
    """Возвращает (is_intake, режим, дата обращения, текст письма)."""
    s = (text or "").lstrip()
    if not s.lower().startswith(_PREFIX):
        return False, "", "", text
    rest = s[len(_PREFIX):].lstrip(" :-—")
    mode, today = "staged", ""
    # Режим и дату разрешено указать перед письмом — в любом порядке, но только в начале.
    for _ in range(2):
        head, _, tail = rest.partition(" ")
        lowered = head.strip().lower()
        day = _TODAY_RE.match(lowered)
        if lowered in _MODE_ALIASES:
            mode = _MODE_ALIASES[lowered]
        elif day:
            today = day.group(1)
        else:
            break
        rest = tail.lstrip()
    return True, mode, today, rest.strip()


def usage_markdown() -> str:
    """Подсказка, когда `/intake` вызвали без письма."""
    return _USAGE


def _fields_table(result: IntakeResult) -> list[str]:
    lines = ["| Поле | Значение |", "|---|---|"]
    for key in INTAKE_FIELD_KEYS:
        value = result.fields.value(key)
        shown = "—" if value == "unknown" else value
        lines.append(f"| {_FIELD_TITLE[key]} | {shown} |")
    return lines


def _stages_table(result: IntakeResult) -> list[str]:
    lines = ["| Этап | Модель | Вызовов | Формат | Время | Цена |", "|---|---|---|---|---|---|"]
    for stage in result.stages:
        if stage.error:
            fmt = f"❌ {stage.error}"
        elif stage.repaired:
            fmt = f"⚠️ починен ({stage.first_error})"
        else:
            fmt = "✅"
        lines.append(
            f"| {_STAGE_TITLE.get(stage.stage, stage.stage)} | `{stage.model}` | {stage.calls} | "
            f"{fmt} | {stage.time_ms / 1000:.1f} с | {stage.cost_rub} ₽ |"
        )
    if result.decision.source in _SOURCE_NOTE:
        lines.append(
            f"| 2. Решение по политике | — (код) | 0 | ✅ {_SOURCE_NOTE[result.decision.source]} "
            f"| 0.0 с | 0 ₽ |"
        )
    return lines


def render_intake_card(result: IntakeResult) -> str:
    """Ответ клиенту плюс разбор инференса в markdown."""
    decision = result.decision
    mark = _DECISION_MARK.get(decision.decision, "•")
    missing = ", ".join(decision.missing) if decision.missing else "—"
    lines: list[str] = []
    if result.reply_body:
        lines += [f"**Тема:** {result.reply_subject}", "", result.reply_body]
    else:
        lines.append("_Письмо клиенту не сформировано: этап не прошёл проверку формата._")
    lines += [
        "",
        "---",
        "",
        f"### {mark} Решение: `{decision.decision}` / `{decision.reason}`",
        "",
        f"**Режим:** {result.mode} · **дата обращения:** {result.today} · "
        f"**не хватает полей:** {missing}",
        "",
    ]
    lines += _fields_table(result)
    lines += ["", "### Этапы инференса", ""]
    lines += _stages_table(result)

    m = result.metrics
    repairs = f" (из них {m['repair_calls']} на ремонт формата)" if m["repair_calls"] else ""
    lines += [
        "",
        f"**Цена разбора:** {m['llm_calls']} вызов(ов){repairs} · {m['time_ms'] / 1000:.1f} с · "
        f"{m['prompt_tokens'] + m['completion_tokens']} токенов · {m['cost_rub']} ₽",
    ]
    return "\n".join(lines)
