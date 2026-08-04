"""День 12 (advance): команда `/indirect` — ловушки во внешнем контенте и слои защиты.

Работает как `/redteam`: перехватываем префикс в последней реплике пользователя. В чат уходит
таблица «ловушка → сработала ли инъекция» по каждому пресету защиты плюс разбор, что закрыл каждый слой.
"""

from __future__ import annotations

from ..agent_constants import (
    INDIRECT_HIDING,
    INDIRECT_MODEL,
    INDIRECT_PRESETS,
    INDIRECT_SCENARIOS,
    INDIRECT_SOURCES,
    INDIRECT_WEAK_MODEL,
)
from ..indirect import IndirectResult, LayerRun, layer_effect

_PREFIX = "/indirect"
_MODEL_ALIASES = {"strong": INDIRECT_MODEL, "weak": INDIRECT_WEAK_MODEL}
_PRESET_TITLE = {
    "none": "без защиты",
    "sanitize": "только чистка входа",
    "boundary": "только границы данных",
    "guard": "только проверка выхода",
    "all": "все три слоя",
}
_SCENARIO_TITLE = {
    "summarize": "суммаризатор письма",
    "analyze": "аналитик документа",
    "search": "поисковый агент",
}
_USAGE = (
    "### `/indirect` — ловушки непрямой инъекции\n\n"
    "Гоняет корпус ловушек (`data/indirect_cases.jsonl`) через агентов, читающих внешний контент, "
    "и сравнивает наборы слоёв защиты.\n\n"
    "```\n/indirect none all\n```\n\n"
    "Слова после команды:\n\n"
    f"- пресеты защиты: `{'`, `'.join(INDIRECT_PRESETS)}` (по умолчанию `none` и `all`);\n"
    f"- сценарий: `{'`, `'.join(INDIRECT_SCENARIOS)}`;\n"
    f"- носитель: `{'`, `'.join(INDIRECT_SOURCES)}`;\n"
    f"- техника сокрытия: `{'`, `'.join(INDIRECT_HIDING)}`;\n"
    f"- тир модели: `strong` (`{INDIRECT_MODEL}`) или `weak` (`{INDIRECT_WEAK_MODEL}`);\n"
    "- конкретная ловушка: её id, например `web-fake-price`.\n\n"
    "```\n/indirect none sanitize guard all web-fake-price\n```"
)


def detect_indirect_command(text: str) -> tuple[bool, tuple[str, ...], dict[str, str]]:
    """Возвращает (is_indirect, пресеты защиты, фильтры корпуса вместе с моделью)."""
    s = (text or "").lstrip()
    if not s.lower().startswith(_PREFIX):
        return False, (), {}
    rest = s[len(_PREFIX):].lstrip(" :-—")
    presets: list[str] = []
    filters = {"scenario": "", "source": "", "hiding": "", "ids": "", "model": INDIRECT_MODEL}
    ids: list[str] = []
    for word in rest.split():
        token = word.strip().lower()
        if token in INDIRECT_PRESETS:
            if token not in presets:
                presets.append(token)
        elif token in _MODEL_ALIASES:
            filters["model"] = _MODEL_ALIASES[token]
        elif token in INDIRECT_SCENARIOS:
            filters["scenario"] = token
        elif token in INDIRECT_SOURCES:
            filters["source"] = token
        elif token in INDIRECT_HIDING:
            filters["hiding"] = token
        elif token:
            ids.append(token)
    filters["ids"] = ",".join(ids)
    return True, tuple(presets) or ("none", "all"), filters


def usage_markdown() -> str:
    """Подсказка по `/indirect` — она же ответ на команду с непонятными аргументами."""
    return _USAGE


def _result_row(result: IndirectResult) -> str:
    if result.error:
        status = "⚠️ ошибка вызова"
        detail = result.error
    elif result.injected:
        status = "❌ инъекция прошла"
        detail = ", ".join(result.signals[:2]) or "—"
    elif result.blocked:
        status = "🛑 ответ заблокирован"
        detail = ", ".join(sorted({f.kind for f in result.findings})) or "—"
    else:
        status = "✅ чисто"
        detail = ", ".join(sorted({f.kind for f in result.findings})) or "следов нет"
    useful = "да" if result.useful else "**нет**"
    scenario = _SCENARIO_TITLE.get(result.scenario, result.scenario)
    return f"| `{result.case_id}` | {scenario} | `{result.hiding}` | {status} | {detail} | {useful} |"


def _run_block(run: LayerRun) -> list[str]:
    title = _PRESET_TITLE.get(run.preset, run.preset)
    layers = ", ".join(run.layers()) or "—"
    model = next((r.model for r in run.results if r.model), "—")
    lines = [
        f"#### `{run.preset}` — {title}",
        "",
        f"Слои: {layers} · инъекций прошло **{run.injected} из {run.total}** · "
        f"блокировок {run.blocked} · полезность потеряна у {run.broke_usefulness} · "
        f"`{model}` · {run.cost_rub} ₽",
        "",
        "| Ловушка | Агент | Сокрытие | Итог | Детали | Польза |",
        "|---|---|---|---|---|---|",
    ]
    lines += [_result_row(r) for r in run.results]
    return lines


def render_indirect_card(runs: dict[str, LayerRun]) -> str:
    """Результат прогона: таблица по каждому пресету и разбор вклада слоёв."""
    lines = ["### Непрямые инъекции: ловушки и слои защиты", ""]
    for preset, run in runs.items():
        lines += _run_block(run)
        lines.append("")
    effect = layer_effect(runs)
    if any(effect.values()):
        lines += ["---", "", "**Что дала защита:**", ""]
        if effect["fixed"]:
            lines.append(f"- закрыто: {', '.join(f'`{c}`' for c in effect['fixed'])}")
        if effect["still_injected"]:
            lines.append(
                f"- проходит и со всеми слоями: {', '.join(f'`{c}`' for c in effect['still_injected'])}"
            )
        if effect["usefulness_lost"]:
            lines.append(
                f"- защита сломала полезный ответ: "
                f"{', '.join(f'`{c}`' for c in effect['usefulness_lost'])}"
            )
    return "\n".join(lines)
