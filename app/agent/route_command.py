"""День 8 (advance): команда `/route` — ответ через каскад моделей плюс карточка маршрута.

Работает как `/triage`: перехватываем префикс в последней реплике пользователя. В чат уходит
сам ответ и под ним разбор — кто отвечал, почему эскалировали (или почему нет) и во что это обошлось.
"""

from __future__ import annotations

from ..routing import RouteResult

_PREFIX = "/route"
_TIER_MARK = {"small": "🟢", "large": "🔴"}
_USAGE = (
    "### `/route` — ответ с маршрутизацией между моделями\n\n"
    "Задайте вопрос после команды, например:\n\n"
    "```\n/route Сколько будет 17 × 23?\n```\n\n"
    "Сначала отвечает дешёвая модель. Если уверенности не хватает — запрос уходит "
    "на сильную. В ответе видно, кто отвечал, по какой причине и сколько это стоило."
)


def detect_route_command(text: str) -> tuple[bool, str]:
    """Возвращает (is_route, текст вопроса без префикса)."""
    s = (text or "").lstrip()
    if not s.lower().startswith(_PREFIX):
        return False, text
    return True, s[len(_PREFIX):].lstrip(" :-—").strip()


def usage_markdown() -> str:
    """Подсказка, когда `/route` вызвали без вопроса."""
    return _USAGE


def _path_line(result: RouteResult) -> str:
    if result.metrics.get("path") == "large":
        return "сразу на сильную модель (pre-routing по тексту запроса)"
    if result.escalated:
        return "дешёвая → **эскалация** → сильная"
    return "осталось на дешёвой модели"


def render_route_card(result: RouteResult) -> str:
    """Ответ модели плюс разбор маршрута в markdown."""
    mark = _TIER_MARK.get(result.tier, "•")
    lines = [
        result.answer,
        "",
        "---",
        "",
        f"### {mark} Маршрут: {_path_line(result)}",
        "",
        "| Шаг | Модель | Уверенность | Время | Цена |",
        "|---|---|---|---|---|",
    ]
    for attempt in result.attempts:
        conf = f"{attempt.assessment.confidence}" if attempt.assessment else "—"
        if attempt.error:
            conf = "ошибка вызова"
        lines.append(
            f"| {attempt.tier} | `{attempt.model}` | {conf} | "
            f"{attempt.time_ms / 1000:.1f} с | {attempt.cost_rub} ₽ |"
        )
    lines.append("")

    pre = result.preroute
    if pre.reasons:
        lines.append(
            f"**Pre-routing:** сложность {pre.score} → `{pre.tier}` "
            f"({', '.join(pre.reasons)})"
        )
    else:
        lines.append(f"**Pre-routing:** маркеров сложности нет → `{pre.tier}`")

    small = next((a for a in result.attempts if a.assessment is not None), None)
    if small is not None and small.assessment is not None:
        a = small.assessment
        self_rep = a.self_reported if a.self_reported is not None else "не выдана"
        lines.append(f"**Самооценка дешёвой модели:** {self_rep} → итог {a.confidence}")
        if a.signals:
            lines.append("**Сигналы:**")
            lines += [f"- {s}" for s in a.signals]
        else:
            lines.append("**Сигналы:** ничего подозрительного")
    lines.append(f"**Решение:** {result.escalation_reason}")

    m = result.metrics
    wasted = f" (из них {m['wasted_rub']} ₽ на неиспользованный черновик)" if m["wasted_rub"] else ""
    lines += [
        "",
        f"**Цена ответа:** {m['llm_calls']} вызов(ов) · {m['time_ms'] / 1000:.1f} с · "
        f"{m['prompt_tokens'] + m['completion_tokens']} токенов · {m['cost_rub']} ₽{wasted}",
    ]
    return "\n".join(lines)
