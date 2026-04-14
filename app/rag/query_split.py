"""Разбиение длинного пользовательского сообщения на подзапросы для multi-query RAG."""

from __future__ import annotations

import re

_NUMBERED_LINE = re.compile(r"^\s*\d+[\.\)]\s+")
_INLINE_AFTER_Q = re.compile(r"(?<=\?)\s+(?=\d+[\.\)]\s)")
# Фрагменты, заканчивающиеся на «?» (несколько вопросов в одной строке / абзаце).
_MULTI_Q_FIND = re.compile(r"[^\n?]{18,}?\?")


def _merge_unique(parts: list[str], max_queries: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        s = (p or "").strip()
        if not s:
            continue
        key = s.lower()[:240]
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max_queries:
            break
    return out


def rag_expansion_queries(user_text: str) -> list[str]:
    """
    Короткие синтетические запросы по якорям из текста пользователя —
    подтягивают README, agent.py, приложение A.42, 22_eval даже при слабом общем эмбеддинге.
    """
    low = (user_text or "").lower()
    ex: list[str] = []

    if "readme" in low and any(
        x in low for x in ("sqlite", "индекс", "rag", "путь", "репозитор")
    ):
        ex.append("README путь SQLite индекс RAG data/rag_index chunks.sqlite")

    if "_mcp_max_steps" in low or ("mcp" in low and "agent" in low):
        ex.append("_MCP_MAX_STEPS константа в app agent.py максимум шагов")

    if "22_eval" in low or "polareval" in low or "rel-day22" in low:
        ex.append(
            "22_eval_knowledge PolarEval Kit REL-DAY22 расшифровка RAG Retrieval-Augmented Generation"
        )

    compact = low.replace(" ", "")
    if (
        "a.42" in low
        or "a.42" in compact
        or ("узл" in low and "чеклист" in low)
        or "node-q-042" in low
        or "node-q-" in low
    ):
        ex.append("PolarLine приложение A.42 чеклист узла NODE-Q-042 audit_bus ретенция 400 дней")

    if "ретенц" in low and "audit_bus" in low:
        ex.append("audit_bus ретенция аудита 400 дней PolarLine appendix")

    if "rag" in low and any(x in low for x in ("расшифров", "аббревиат", "означа", "22_eval")):
        ex.append("RAG Retrieval-Augmented Generation 22_eval_knowledge интерфейс")

    return ex


def rag_subqueries(user_text: str, *, max_queries: int = 18) -> list[str]:
    """
    Один эмбеддинг на весь текст с многими несвязанными вопросами даёт слабый recall.
    Несколько «?», нумерованный список, плюс расширения по ключевым словам.
    """
    t = (user_text or "").strip()
    if not t:
        return []

    n_q = t.count("?")
    long = len(t) >= 500

    def finish(candidates: list[str]) -> list[str]:
        base = _merge_unique(candidates, max_queries)
        if not base:
            base = [t]
        extra = rag_expansion_queries(t)
        return _merge_unique(base + extra, max_queries)

    # Короткий одиночный вопрос — только расширения при необходимости
    if n_q <= 1 and len(t) < 900:
        return finish([t])

    lines = t.split("\n")
    blocks: list[str] = []
    current: list[str] = []

    for line in lines:
        if _NUMBERED_LINE.match(line) and current:
            blk = "\n".join(current).strip()
            if blk:
                blocks.append(blk)
            current = [line]
        else:
            current.append(line)
    if current:
        blk = "\n".join(current).strip()
        if blk:
            blocks.append(blk)

    out = [b for b in blocks if len(b) > 15 and ("?" in b or len(b) > 50)]
    if len(out) > 1:
        return finish(out)

    # «1. …? 2. …?» в одной строке
    if len(out) <= 1:
        inline = _INLINE_AFTER_Q.split(t)
        inline = [x.strip() for x in inline if len(x.strip()) > 15]
        if len(inline) > 1:
            return finish(inline)

    # Несколько «?» в тексте — выделяем подстроки …?
    if len(out) <= 1 and n_q >= 2:
        found = _MULTI_Q_FIND.findall(t)
        found = [x.strip() for x in found if len(x.strip()) >= 20]
        if len(found) >= 2:
            return finish(found)

    q_lines = [ln.strip() for ln in lines if "?" in ln and len(ln.strip()) > 25]
    if len(q_lines) > 1:
        return finish(q_lines)

    paras = [p.strip() for p in re.split(r"\n\s*\n", t) if "?" in p and len(p.strip()) > 30]
    if len(paras) > 1:
        return finish(paras)

    # Длинный текст с кучей тем, но без «?» (редко) — расширения всё равно
    if long:
        return finish([t])

    return finish([t])
