"""День 10 (advance): команда `/intent` — классификация обращения с карточкой двух уровней.

Работает как `/triage`, `/route` и `/intake`: перехватываем префикс в последней реплике пользователя.
В чат уходит метка интента и разбор — решила ли micro-model сама и во что обошёлся уровень 2.
"""

from __future__ import annotations

from ..micro import IntentResult

_PREFIX = "/intent"
_STRATEGY_ALIASES = {
    "embed": "micro_embed_first",
    "tfidf": "micro_tfidf_first",
    "llm": "llm_only",
    "micro": "micro_only",
}
_LABEL_TITLE = {
    "billing": "💳 деньги и счета",
    "technical": "🛠 продукт не работает",
    "account": "🔑 доступ и учётная запись",
    "data_loss": "🗂 потеря данных",
    "feedback": "💬 отзыв или пожелание",
    "other": "📨 не про поддержку продукта",
}
_SOURCE_TITLE = {
    "micro": "уровень 1 — micro-model",
    "llm": "уровень 2 — большая модель",
    "micro_after_llm_error": "уровень 2 сорвался, метка от micro-model",
    "failed": "метку получить не удалось",
}
_ESCALATE_TITLE = {
    "short_input": "слишком короткое обращение",
    "no_close_neighbor": "нет похожего примера в банке",
    "fallback_label": "победил класс-помойка other",
    "low_margin": "два класса рядом, отрыва нет",
    "low_score": "уверенность ниже порога",
}
_USAGE = (
    "### `/intent` — классификация обращения через micro-model\n\n"
    "Вставьте текст обращения после команды:\n\n"
    "```\n/intent Списали деньги дважды за один месяц, верните переплату\n```\n\n"
    "Стратегия задаётся первым словом: `embed` (по умолчанию — kNN по эмбеддингам с fallback), "
    "`tfidf` (офлайн-классификатор на n-граммах с fallback), `micro` (только уровень 1, без "
    "большой модели), `llm` (сразу большая модель — базовая линия).\n\n"
    "```\n/intent tfidf Не приходит код подтверждения при входе\n```"
)


def detect_intent_command(text: str) -> tuple[bool, str, str]:
    """Возвращает (is_intent, стратегия, текст обращения)."""
    s = (text or "").lstrip()
    if not s.lower().startswith(_PREFIX):
        return False, "", text
    rest = s[len(_PREFIX):].lstrip(" :-—")
    strategy = _STRATEGY_ALIASES["embed"]
    head, _, tail = rest.partition(" ")
    if head.strip().lower() in _STRATEGY_ALIASES:
        strategy = _STRATEGY_ALIASES[head.strip().lower()]
        rest = tail.lstrip()
    return True, strategy, rest.strip()


def usage_markdown() -> str:
    """Подсказка, когда `/intent` вызвали без текста обращения."""
    return _USAGE


def _micro_block(result: IntentResult) -> list[str]:
    micro = result.micro
    if micro is None:
        return ["_Уровень 1 не использовался: стратегия `llm_only` идёт сразу в большую модель._"]
    mark = "✅ OK" if micro.ok else "⚠️ UNSURE"
    reason = _ESCALATE_TITLE.get(micro.escalate_reason or "", micro.escalate_reason or "—")
    lines = [
        f"**Уровень 1 ({micro.backend}):** {mark} · метка `{micro.label}` · "
        f"score {micro.score:.2f} · {micro.time_ms} мс",
        "",
        f"Ближайший пример: {micro.top_similarity:.3f} · отрыв от второго класса: "
        f"{micro.margin:.3f} · согласие соседей: {micro.votes:.0%}",
    ]
    if not micro.ok:
        lines += ["", f"Причина эскалации: **{reason}**"]
    lines += ["", "| Сосед | Метка | Близость |", "|---|---|---|"]
    for neighbor in micro.neighbors:
        preview = neighbor.text[:60] + ("…" if len(neighbor.text) > 60 else "")
        lines.append(f"| {preview} | `{neighbor.label}` | {neighbor.similarity:.3f} |")
    return lines


def _llm_block(result: IntentResult) -> list[str]:
    call = result.llm
    if call is None:
        return ["**Уровень 2:** не понадобился — большая модель не вызывалась. 🎉"]
    if call.error:
        fmt = f"❌ {call.error}"
    elif call.repaired:
        fmt = f"⚠️ формат починен ({call.first_error})"
    else:
        fmt = "✅"
    lines = [
        f"**Уровень 2:** `{call.model}` · {call.calls} вызов(ов) · {fmt} · "
        f"{call.time_ms / 1000:.1f} с · {call.cost_rub} ₽"
    ]
    if result.llm_answer is not None:
        answer = result.llm_answer
        lines.append(f"Модель: `{answer.label}` (confidence {answer.confidence:.2f}) — {answer.reason}")
    return lines


def render_intent_card(result: IntentResult) -> str:
    """Метка интента плюс разбор обоих уровней в markdown."""
    title = _LABEL_TITLE.get(result.label, result.label)
    source = _SOURCE_TITLE.get(result.source, result.source)
    m = result.metrics
    lines = [
        f"### Интент: `{result.label}` — {title}",
        "",
        f"**Решил:** {source} · **стратегия:** `{result.strategy}`",
        "",
    ]
    lines += _micro_block(result)
    lines += ["", *_llm_block(result)]
    lines += [
        "",
        "---",
        "",
        f"**Цена запроса:** {m['llm_calls']} вызов(ов) большой модели · "
        f"{m['time_ms'] / 1000:.1f} с · {m['cost_rub']} ₽",
    ]
    return "\n".join(lines)
