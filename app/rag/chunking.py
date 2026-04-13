"""Две стратегии chunking: фиксированное окно и структура (заголовки md / файл)."""

from __future__ import annotations

import hashlib
import re
from typing import TypedDict


class ChunkRecord(TypedDict):
    chunk_id: str
    source: str
    title: str
    section: str | None
    text: str
    strategy: str


def _make_chunk_id(strategy: str, source: str, section: str | None, index: int, text: str) -> str:
    raw = f"{strategy}|{source}|{section or ''}|{index}|{text[:128]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def chunk_fixed_size(
    text: str,
    *,
    source: str,
    title: str,
    chunk_chars: int = 1400,
    overlap: int = 200,
) -> list[ChunkRecord]:
    """Скользящее окно по символам (грубое, без учёта заголовков)."""
    text = text.strip()
    if not text:
        return []
    out: list[ChunkRecord] = []
    start = 0
    idx = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        piece = text[start:end].strip()
        if piece:
            out.append(
                {
                    "chunk_id": _make_chunk_id("fixed", source, None, idx, piece),
                    "source": source,
                    "title": title,
                    "section": None,
                    "text": piece,
                    "strategy": "fixed",
                }
            )
            idx += 1
        if end >= n:
            break
        start = max(0, end - overlap)
    return out


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def chunk_structural(
    text: str,
    *,
    source: str,
    title: str,
    is_markdown: bool,
    max_section_chars: int = 2400,
    subchunk_chars: int = 1200,
    subchunk_overlap: int = 150,
) -> list[ChunkRecord]:
    """
    Markdown: секции по заголовкам; длинные секции режутся подчанками с тем же section.
    Не-md: один «раздел» = имя файла, внутри — fixed-подчанки.
    """
    text = text.strip()
    if not text:
        return []

    if not is_markdown:
        subs = chunk_fixed_size(
            text,
            source=source,
            title=title,
            chunk_chars=subchunk_chars,
            overlap=subchunk_overlap,
        )
        for i, c in enumerate(subs):
            c["strategy"] = "structural"
            c["section"] = title
            c["chunk_id"] = _make_chunk_id("structural", source, title, i, c["text"])
        return subs

    lines = text.splitlines()
    stack: list[str] = []
    buf: list[str] = []
    out: list[ChunkRecord] = []
    global_idx = 0

    def flush_section(sec_path: str | None) -> None:
        nonlocal global_idx, buf
        body = "\n".join(buf).strip()
        buf = []
        if not body:
            return
        if len(body) <= max_section_chars:
            out.append(
                {
                    "chunk_id": _make_chunk_id("structural", source, sec_path, global_idx, body),
                    "source": source,
                    "title": title,
                    "section": sec_path,
                    "text": body,
                    "strategy": "structural",
                }
            )
            global_idx += 1
            return
        subs = chunk_fixed_size(
            body,
            source=source,
            title=title,
            chunk_chars=subchunk_chars,
            overlap=subchunk_overlap,
        )
        for c in subs:
            c["strategy"] = "structural"
            c["section"] = sec_path
            c["chunk_id"] = _make_chunk_id(
                "structural", source, sec_path, global_idx, c["text"]
            )
            global_idx += 1
            out.append(c)  # type: ignore[arg-type]

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush_section(" / ".join(stack) if stack else None)
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            while len(stack) >= level:
                stack.pop()
            stack.append(heading_text)
            continue
        buf.append(line)
    flush_section(" / ".join(stack) if stack else None)
    return out
