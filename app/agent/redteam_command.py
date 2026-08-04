"""День 11 (advance): команда `/redteam` — прогон инъекций по наивному и защищённому промпту.

Работает как `/triage`, `/intake` и `/intent`: перехватываем префикс в последней реплике.
В чат уходит таблица «атака → устоял/пробило» по каждой версии промпта и разбор пробоев.
"""

from __future__ import annotations

from ..agent_constants import (
    SECURITY_MODEL,
    SECURITY_PROMPT_VERSIONS,
    SECURITY_TARGETS,
    SECURITY_VECTORS,
    SECURITY_WEAK_MODEL,
)
from ..security import AttackVerdict, RedteamRun, fixed_by_hardening

_PREFIX = "/redteam"
# Тир модели словом: на сильной видно, сколько атак отбивает сама модель, на слабой — сколько промпт.
_MODEL_ALIASES = {"strong": SECURITY_MODEL, "weak": SECURITY_WEAK_MODEL}
_VERSION_ALIASES = {
    "v1": ("v1",),
    "v2": ("v2",),
    "both": SECURITY_PROMPT_VERSIONS,
    "all": SECURITY_PROMPT_VERSIONS,
}
_VECTOR_TITLE = {
    "direct": "прямая инъекция",
    "indirect": "непрямая инъекция",
    "jailbreak": "джейлбрейк",
}
_USAGE = (
    "### `/redteam` — атаки на промпты проекта\n\n"
    "Гоняет корпус инъекций (`data/prompt_attacks.jsonl`) по двум версиям system-промпта: "
    "`v1` — наивный, `v2` — защищённый.\n\n"
    "```\n/redteam both\n```\n\n"
    "Фильтры — словами после команды:\n\n"
    "- версия: `v1`, `v2`, `both` (по умолчанию `both`);\n"
    f"- тир модели: `strong` (`{SECURITY_MODEL}`, по умолчанию) или `weak` "
    f"(`{SECURITY_WEAK_MODEL}` — на нём защиту обеспечивает промпт, а не выравнивание модели);\n"
    f"- цель: `{'`, `'.join(SECURITY_TARGETS)}` — банковский ассистент или живой агент поддержки;\n"
    f"- вектор: `{'`, `'.join(SECURITY_VECTORS)}`;\n"
    "- конкретная атака: её id, например `dan-roleplay`.\n\n"
    "```\n/redteam v2 support indirect\n```\n\n"
    "```\n/redteam both weak grandma-exploit\n```"
)


def detect_redteam_command(text: str) -> tuple[bool, tuple[str, ...], dict[str, str]]:
    """Возвращает (is_redteam, версии промпта, фильтры корпуса вместе с моделью)."""
    s = (text or "").lstrip()
    if not s.lower().startswith(_PREFIX):
        return False, (), {}
    rest = s[len(_PREFIX):].lstrip(" :-—")
    versions = SECURITY_PROMPT_VERSIONS
    filters = {"target": "", "vector": "", "ids": "", "model": SECURITY_MODEL}
    ids: list[str] = []
    for word in rest.split():
        token = word.strip().lower()
        if token in _VERSION_ALIASES:
            versions = _VERSION_ALIASES[token]
        elif token in _MODEL_ALIASES:
            filters["model"] = _MODEL_ALIASES[token]
        elif token in SECURITY_TARGETS:
            filters["target"] = token
        elif token in SECURITY_VECTORS:
            filters["vector"] = token
        elif token:
            ids.append(token)
    filters["ids"] = ",".join(ids)
    return True, versions, filters


def usage_markdown() -> str:
    """Подсказка по `/redteam` — она же ответ на команду с непонятными аргументами."""
    return _USAGE


def _verdict_row(verdict: AttackVerdict) -> str:
    if verdict.error:
        status = "⚠️ ошибка вызова"
        detail = verdict.error
    elif verdict.broken:
        status = "❌ пробило"
        detail = ", ".join(verdict.signals[:3]) or "—"
    else:
        status = "✅ устоял"
        detail = "отказ" if verdict.refused else "ушёл от темы без отказа"
    vector = _VECTOR_TITLE.get(verdict.vector, verdict.vector)
    return f"| `{verdict.attack_id}` | {vector} | `{verdict.target}` | {status} | {detail} |"


def _run_block(run: RedteamRun) -> list[str]:
    title = "наивный промпт" if run.version == "v1" else "защищённый промпт"
    model = next((v.model for v in run.verdicts if v.model), "—")
    lines = [
        f"#### `{run.version}` — {title}",
        "",
        f"Пробито **{run.broken} из {run.total}** · устояло {run.held} · "
        f"ошибок вызова {run.errors} · `{model}` · {run.cost_rub} ₽",
        "",
        "| Атака | Вектор | Цель | Итог | Сигналы |",
        "|---|---|---|---|---|",
    ]
    lines += [_verdict_row(v) for v in run.verdicts]
    return lines


def render_redteam_card(runs: dict[str, RedteamRun]) -> str:
    """Результат ред-тима: таблица по каждой версии и разбор, что закрыла защита."""
    lines = ["### Ред-тим промптов", ""]
    for version in SECURITY_PROMPT_VERSIONS:
        run = runs.get(version)
        if run is None:
            continue
        lines += _run_block(run)
        lines.append("")
    diff = fixed_by_hardening(runs)
    if any(diff.values()):
        lines += ["---", "", "**Что дала защита:**", ""]
        if diff["fixed"]:
            lines.append(f"- закрыто в v2: {', '.join(f'`{a}`' for a in diff['fixed'])}")
        if diff["still_broken"]:
            lines.append(
                f"- пробивает обе версии: {', '.join(f'`{a}`' for a in diff['still_broken'])}"
            )
        if diff["regressed"]:
            lines.append(
                f"- пробило только v2 (регресс): {', '.join(f'`{a}`' for a in diff['regressed'])}"
            )
    return "\n".join(lines)
