"""День 7 (advance): команда `/triage` — прогон обращения через гейт уверенности.

Работает как `/help` и `/support`: ловим префикс в последней реплике пользователя,
но здесь ответ модели в чат не стримится — вместо него печатается карточка вердикта
(решение, статус приёмки, голоса сэмплов, нарушенные инварианты, latency и цена).
"""

from __future__ import annotations

from ..confidence import Verdict

_PREFIX = "/triage"
_STATUS_MARK = {"OK": "✅", "UNSURE": "🟡", "FAIL": "⛔"}
_USAGE = (
    "### `/triage` — триаж обращения с оценкой уверенности\n\n"
    "Пришлите текст обращения после команды, например:\n\n"
    "```\n/triage С карты дважды списали 2400 ₽ за один заказ, верните деньги\n```\n\n"
    "Ответ: категория, приоритет и действие, статус приёмки (OK / UNSURE / FAIL) "
    "и то, какое действие реально применил гейт."
)


def detect_triage_command(text: str) -> tuple[bool, str]:
    """Возвращает (is_triage, текст обращения без префикса)."""
    s = (text or "").lstrip()
    if not s.lower().startswith(_PREFIX):
        return False, text
    return True, s[len(_PREFIX):].lstrip(" :-—").strip()


def usage_markdown() -> str:
    """Подсказка, когда `/triage` вызвали без текста обращения."""
    return _USAGE


def _votes_line(votes: dict[str, int]) -> str:
    if not votes:
        return "—"
    ordered = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"`{key}` ×{count}" for key, count in ordered)


def render_triage_card(verdict: Verdict, text: str) -> str:
    """Карточка вердикта в markdown для вывода в чат."""
    mark = _STATUS_MARK.get(verdict.status, "•")
    lines = [
        f"### {mark} Триаж — статус **{verdict.status}**, confidence **{verdict.confidence}**",
        "",
        f"> {text[:300]}" + ("…" if len(text) > 300 else ""),
        "",
    ]
    if verdict.decision is not None:
        d = verdict.decision
        lines += [
            "| Поле | Значение |",
            "|---|---|",
            f"| Категория | `{d.category}` |",
            f"| Приоритет | `{d.priority}` |",
            f"| Действие модели | `{d.action}` |",
            f"| **Применено гейтом** | **`{verdict.applied_action}`** |",
            "",
            f"**Обоснование:** {d.reason}",
        ]
        if d.missing:
            lines.append("**Не хватает данных:** " + ", ".join(f"«{m}»" for m in d.missing))
    else:
        lines += [
            f"Решение не принято — гейт отдал обращение человеку (`{verdict.applied_action}`).",
        ]
    lines.append("")

    valid = len([s for s in verdict.samples if s.decision is not None])
    lines.append(
        f"**Redundancy:** {valid}/{len(verdict.samples)} сэмплов прошли проверки, "
        f"agreement {verdict.agreement} · голоса: {_votes_line(verdict.votes)}"
    )
    if verdict.selfcheck is not None:
        reason = f" — {verdict.selfcheck.reason}" if verdict.selfcheck.reason else ""
        lines.append(f"**Self-check:** `{verdict.selfcheck.verdict}`{reason}")
    else:
        lines.append("**Self-check:** не потребовался (согласие полное, действие не рискованное)")
    if verdict.violations:
        lines.append("**Нарушенные проверки:**")
        lines += [f"- {v}" for v in verdict.violations]
    else:
        lines.append("**Нарушенные проверки:** нет")

    m = verdict.metrics
    lines += [
        "",
        f"**Цена решения:** {m['llm_calls']} вызов(ов) "
        f"(из них ремонтных {m['repair_calls']}, self-check {m['selfcheck_calls']}) · "
        f"{m['time_ms'] / 1000:.1f} с · "
        f"{m['prompt_tokens'] + m['completion_tokens']} токенов · {m['cost_rub']} ₽",
    ]
    return "\n".join(lines)
