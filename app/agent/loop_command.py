"""День 14 (advance): команда `/loop` — execution loop с security step и вызовами через шлюз.

Работает как `/gateway`: перехватываем префикс в последней реплике пользователя. В чат уходит
карточка по каждой задаче — попытки, находки ревью, что перехватил шлюз — и общий свод.
"""

from __future__ import annotations

from ..agent_constants import (
    LOOP_GEN_MODEL,
    LOOP_MAX_ATTEMPTS,
    LOOP_REVIEW_MODEL,
)
from ..loop import LoopRun, load_tasks, loop_summary

_PREFIX = "/loop"
_SUBCOMMANDS = ("tasks",)
_OUTCOME_TITLE = {
    "generated": "сгенерировано",
    "gateway_blocked": "🛑 шлюз не пропустил вызов",
    "no_code": "⚠️ в ответе нет кода",
    "checks_failed": "❌ проверки не прошли",
    "security_blocked": "🔒 security review вернул на доработку",
    "accepted_with_warnings": "⚠️ принято с warning",
    "accepted": "✅ принято",
}
_STATUS_TITLE = {
    "committed": "✅ закоммичено",
    "committed_with_warnings": "⚠️ закоммичено с warning",
    "failed": "❌ не принято за отведённые попытки",
}
_USAGE = (
    "### `/loop` — execution loop с security step\n\n"
    "Цикл: генерация кода → синтаксис и тесты в песочнице → security review вторым вызовом LLM → "
    "«коммит». Все вызовы модели идут через шлюз Дня 13.\n\n"
    "```\n/loop\n```\n\n"
    "- `/loop tasks` — список задач корпуса и их ловушек;\n"
    "- `/loop save-token log-requests` — только указанные задачи;\n"
    f"- генератор `{LOOP_GEN_MODEL}`, ревьюер `{LOOP_REVIEW_MODEL}`, до {LOOP_MAX_ATTEMPTS} "
    "попыток на задачу.\n\n"
    "Прогон трёх задач — это до шести вызовов модели, занимает около минуты."
)


def detect_loop_command(text: str) -> tuple[bool, str, tuple[str, ...]]:
    """Возвращает (is_loop, подкоманда, id задач)."""
    s = (text or "").lstrip()
    if not s.lower().startswith(_PREFIX):
        return False, "", ()
    rest = s[len(_PREFIX):].lstrip(" :-—")
    words = [w.strip() for w in rest.split() if w.strip()]
    if words and words[0].lower() in _SUBCOMMANDS:
        return True, words[0].lower(), tuple(words[1:])
    return True, "", tuple(words)


def usage_markdown() -> str:
    """Подсказка по `/loop` — она же ответ на команду с непонятными аргументами."""
    return _USAGE


def render_tasks_card() -> str:
    """Список задач корпуса: что просим сгенерировать и какие небезопасные решения провоцируем."""
    try:
        tasks = load_tasks()
    except ValueError as exc:
        return f"### Задачи цикла\n\n{exc}"
    lines = [
        "### Задачи цикла",
        "",
        "| Задача | Что просим | Ловушки | Почему провоцирует |",
        "|---|---|---|---|",
    ]
    for task in tasks:
        traps = ", ".join(f"`{t}`" for t in task.traps) or "—"
        lines.append(f"| `{task.id}` | {task.title} | {traps} | {task.note or '—'} |")
    return "\n".join(lines)


def _gateway_line(run: LoopRun) -> str:
    events = [e for attempt in run.attempts for e in attempt.gateway]
    if not events:
        return "вызовов не было"
    dirty = [e for e in events if not e.clean]
    if not dirty:
        return f"{len(events)} вызовов, все чистые"
    parts = []
    for event in dirty:
        found = ", ".join(sorted(set(event.input_kinds) | set(event.output_kinds))) or event.status
        parts.append(f"`{event.stage}`: {found}")
    return f"{len(events)} вызовов, из них {len(dirty)} с находками — " + "; ".join(parts)


def _attempt_rows(run: LoopRun) -> list[str]:
    rows = []
    for attempt in run.attempts:
        checks = ", ".join(f"{c.stage}:{'ok' if c.ok else 'fail'}" for c in attempt.checks) or "—"
        if attempt.security is None:
            security = "—"
        elif attempt.security.error:
            security = f"ошибка формата: {attempt.security.error}"
        else:
            blocking = ", ".join(
                f"{f.rule}@{f.line}" if f.line else f.rule for f in attempt.security.blocking
            )
            warnings = ", ".join(f.rule for f in attempt.security.warnings)
            security = " · ".join(
                part for part in (
                    f"**блок:** {blocking}" if blocking else "",
                    f"warning: {warnings}" if warnings else "",
                ) if part
            ) or "чисто"
        rows.append(
            f"| {attempt.number} | {_OUTCOME_TITLE.get(attempt.outcome, attempt.outcome)} | "
            f"{checks} | {security} | {attempt.cost_rub} ₽ |"
        )
    return rows


def render_loop_card(runs: list[LoopRun]) -> str:
    """Карточка прогона: по задаче — попытки и находки, снизу — общий свод."""
    lines = ["### Execution loop с security step", ""]
    for run in runs:
        status = _STATUS_TITLE.get(run.status, run.status)
        lines += [
            f"#### `{run.task_id}` — {run.title}",
            "",
            f"**{status}** · попыток: {len(run.attempts)} · вызовов модели: {run.llm_calls} · "
            f"{run.cost_rub} ₽" + (f" · артефакт `{run.artifact}`" if run.artifact else ""),
            "",
            "| Попытка | Итог | Проверки | Security review | Цена |",
            "|---|---|---|---|---|",
        ]
        lines += _attempt_rows(run)
        lines += [
            "",
            f"- **шлюз:** {_gateway_line(run)}",
            f"- **security step поймал:** "
            f"{', '.join(f'`{r}`' for r in run.caught_by_security()) or '—'}",
            f"- **пропущено с warning:** "
            f"{', '.join(f'`{r}`' for r in run.warned_by_security()) or '—'}",
            f"- **мимо обоих:** {', '.join(f'`{r}`' for r in run.missed_traps()) or '—'}",
            "",
        ]

    summary = loop_summary(runs)
    lines += [
        "---",
        "",
        f"**Итого:** задач {summary['tasks']} · закоммичено {summary['committed']} · "
        f"с warning {summary['committed_with_warnings']} · не принято {summary['failed']} · "
        f"попыток {summary['attempts']} · вызовов {summary['llm_calls']} · "
        f"{summary['cost_rub']} ₽",
        "",
        f"- security step вернул на доработку: "
        f"{', '.join(f'`{k}` × {v}' for k, v in summary['caught_by_security'].items()) or '—'}",
        f"- шлюз перехватил: "
        f"{', '.join(f'`{k}` × {v}' for k, v in summary['caught_by_gateway'].items()) or '—'}",
        f"- прошло мимо обоих: "
        f"{', '.join(f'`{k}` × {v}' for k, v in summary['missed_by_both'].items()) or '—'}",
    ]
    return "\n".join(lines)
