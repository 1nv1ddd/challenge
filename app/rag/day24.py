"""День 24: обязательные источники и цитаты, режим «недостаточно доказательств»."""

from __future__ import annotations

import os
import re
from typing import Any

from .post_retrieval import is_keyword_boosted_hit

_DEFAULT_ANSWER_MIN = float(os.getenv("RAG_ANSWER_MIN_SCORE", "0.25") or "0.25")

_EXPLICIT_ANCHOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)PL-\d{3}-RET"),
    re.compile(r"(?i)NODE-Q-\d+"),
)


def explicit_anchors(user_text: str) -> list[str]:
    """Коды вида PL-###-RET / NODE-Q-###, явно упомянутые в запросе."""
    out: list[str] = []
    for pat in _EXPLICIT_ANCHOR_PATTERNS:
        for m in pat.finditer(user_text or ""):
            val = m.group(0).upper()
            if val not in out:
                out.append(val)
    return out


def filter_hits_by_anchors(
    hits: list[dict[str, Any]], user_text: str
) -> list[dict[str, Any]]:
    """
    Для показа пользователю оставить только чанки, в тексте которых есть хотя бы один якорь
    из запроса. Если якорей в запросе нет — возвращаем hits без изменений. Если ни один чанк
    не совпал — тоже возвращаем исходный список (fallback, чтобы не выдать пустой appendix).
    """
    anchors = explicit_anchors(user_text)
    if not anchors or not hits:
        return list(hits or [])
    lowered = [a.lower() for a in anchors]
    keep = [
        h
        for h in hits
        if any(a in (h.get("text") or "").lower() for a in lowered)
    ]
    return keep or list(hits)


def unmatched_anchors(user_text: str, hits: list[dict[str, Any]]) -> list[str]:
    """
    Якоря из запроса, которых буквально нет ни в одном чанке.
    Эмбеддинги PL-###-RET между собой очень близки, поэтому высокий score без exact-match
    — не доказательство. Если юзер спросил про несуществующий код, hits всё равно будут
    «похожи» на другие PL-коды, но ответ по корпусу невозможен.
    """
    anchors = explicit_anchors(user_text)
    if not anchors:
        return []
    texts_lower = [(h.get("text") or "").lower() for h in hits or []]
    missing: list[str] = []
    for a in anchors:
        al = a.lower()
        if not any(al in t for t in texts_lower):
            missing.append(a)
    return missing


def answer_min_score_from_cfg(rag_cfg: dict[str, Any] | None) -> float:
    if not rag_cfg:
        return _DEFAULT_ANSWER_MIN
    raw = rag_cfg.get("answer_min_score")
    if raw is None:
        return _DEFAULT_ANSWER_MIN
    try:
        return float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_ANSWER_MIN


def max_hit_score(hits: list[dict[str, Any]]) -> float:
    if not hits:
        return 0.0
    return max(float(h["score"]) for h in hits)


def insufficient_evidence(
    hits: list[dict[str, Any]],
    min_score: float,
    *,
    user_text: str = "",
) -> bool:
    """
    True — не отвечать по корпусу уверенно: нет чанков, слабая семантика без keyword-буста,
    либо в запросе есть явные якоря (PL-###-RET / NODE-Q-###), которых нет ни в одном чанке.
    """
    if not hits:
        return True
    if user_text and unmatched_anchors(user_text, hits):
        return True
    if any(is_keyword_boosted_hit(h) for h in hits):
        return False
    return max_hit_score(hits) < float(min_score)


def refusal_system_message(
    *,
    best_score: float,
    threshold: float,
    hit_count: int,
    missing_anchors: list[str] | None = None,
) -> str:
    ma = list(missing_anchors or [])
    anchor_line = ""
    anchor_instr = ""
    if ma:
        joined = ", ".join(f"`{a}`" for a in ma)
        anchor_line = f" Якоря без совпадений в найденных отрывках: {joined}."
        anchor_instr = (
            f"В первой же фразе **перечисли** эти якоря как отсутствующие в корпусе: {joined}. "
        )
    return (
        "[RAG · День 24] Контекст для уверенного ответа **недостаточен** "
        f"(отрывков после отбора: {hit_count}; лучший score среди семантических: {best_score:.4f}; "
        f"порог уверенности: {threshold:.4f}). Якорных exact-match чанков нет."
        f"{anchor_line}\n\n"
        "Ответь пользователю **строго** в markdown в такой структуре (заголовки как ниже):\n\n"
        "## Ответ\n"
        "Обязательно начни с явного **«Не знаю»** (например: «Не знаю: …»). "
        f"{anchor_instr}"
        "По корпусу **нельзя** дать надёжный ответ при текущем запросе "
        "(релевантность ниже порога или запрошенный код/раздел отсутствует в корпусе). "
        "Попроси **уточнить** запрос: конкретный код PL-* / NODE-Q-*, раздел справочника, имя файла или цитату.\n"
        "**Запрещено** придумывать факты из «головы».\n\n"
        "## Источники\n"
        "- (не использованы — отрывки не переданы в ответ)\n\n"
        "## Цитаты\n"
        "- (нет)\n"
    )


_COMPARE_CHUNKING_HEADING = re.compile(r"(?mi)^##\s+Сравнение\s+chunking\s*$")


def splice_day24_appendix_before_compare(assistant_markdown: str, appendix_md: str) -> str:
    """
    Порядок по заданию День 24: Ответ → Источники → Цитаты.
    В режиме compare модель пишет «Сравнение chunking» после ответа; сервер вставляет приложение перед этим блоком.
    """
    if not appendix_md:
        return assistant_markdown
    m = _COMPARE_CHUNKING_HEADING.search(assistant_markdown)
    if not m:
        return f"{assistant_markdown.rstrip()}{appendix_md}"
    head = assistant_markdown[: m.start()].rstrip()
    tail = assistant_markdown[m.start() :].lstrip()
    return f"{head}{appendix_md}\n\n{tail}"


def day24_output_format_block(*, compare_mode: bool = False) -> str:
    """Устаревший длинный блок формата; оставлен для тестов совместимости."""
    cmp = ""
    if compare_mode:
        cmp = "После `## Ответ` — блок **«Сравнение chunking»** (2–5 предложений).\n\n"
    return (
        "### Формат (День 24)\n\n"
        "## Ответ\n"
        "Только факты из «Набор отрывков».\n\n"
        f"{cmp}"
        "Разделы `## Источники` и `## Цитаты` добавляет сервер.\n\n"
    )


def merge_hits_for_appendix(*hit_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Объединение чанков из compare (fixed + structural): лучший score на пару chunk_id+source."""
    best: dict[tuple[Any, Any], dict[str, Any]] = {}
    for hits in hit_lists:
        for h in hits or []:
            key = (h.get("chunk_id"), h.get("source"))
            sc = float(h.get("score") or 0.0)
            if key not in best or sc > float(best[key].get("score") or 0.0):
                best[key] = h
    return sorted(best.values(), key=lambda x: float(x.get("score") or 0.0), reverse=True)


def build_day24_appendix_markdown(hits: list[dict[str, Any]], *, quote_max_chars: int = 320) -> str:
    """Детерминированные Источники + Цитаты из фактических чанков RAG (после ответа модели)."""
    if not hits:
        return ""
    src_lines: list[str] = []
    quote_lines: list[str] = []
    for i, h in enumerate(hits, 1):
        cid = h.get("chunk_id", "—")
        src = h.get("source", "—")
        sec = h.get("section") or "—"
        src_lines.append(f"- `{cid}` · `{src}` · section: {sec}")
        raw = str(h.get("text") or "").strip()
        one_line = " ".join(raw.split())
        if len(one_line) > quote_max_chars:
            frag = one_line[: quote_max_chars - 1] + "…"
        else:
            frag = one_line
        frag = frag.replace("«", '"').replace("»", '"')
        quote_lines.append(f"- [{i}] «{frag}» — `{src}`, chunk_id: `{cid}`")
    return (
        "\n\n## Источники\n"
        + "\n".join(src_lines)
        + "\n\n## Цитаты\n"
        + "\n".join(quote_lines)
    )


def combined_insufficient_evidence(
    hit_lists: list[list[dict[str, Any]]],
    min_score: float,
    *,
    user_text: str = "",
) -> bool:
    """Compare: отказ только если **оба** набора отрывков недостаточны."""
    parts = [h for h in hit_lists if h is not None]
    if not parts:
        return True
    return all(
        insufficient_evidence(h, min_score, user_text=user_text) for h in parts
    )
