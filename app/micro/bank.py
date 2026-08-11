"""Банк размеченных примеров для micro-model: чтение jsonl и кэш векторов на диске."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..agent_constants import MICRO_BANK_PATH, MICRO_CACHE_DIR, MICRO_EMBED_MODEL, MICRO_LABELS
from ..rag.embeddings import embed_texts_async

_SLUG_RE = re.compile(r"[^a-z0-9]+")
# Банк маленький и меняется руками, поэтому держим его в памяти процесса до перезапуска.
_BANK_CACHE: dict[str, list["BankItem"]] = {}
_VECTOR_CACHE: dict[str, list[list[float]]] = {}
_VECTOR_LOCK = asyncio.Lock()


@dataclass(frozen=True)
class BankItem:
    """Один размеченный пример: текст обращения и его метка."""

    label: str
    text: str


def _parse_line(line: str, number: int) -> BankItem:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"банк, строка {number}: JSON не разбирается ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"банк, строка {number}: ожидался JSON-объект")
    label = str(data.get("label") or "").strip().lower()
    text = str(data.get("text") or "").strip()
    if label not in MICRO_LABELS:
        raise ValueError(f"банк, строка {number}: метка {label!r} вне списка {list(MICRO_LABELS)}")
    if not text:
        raise ValueError(f"банк, строка {number}: пустой текст примера")
    return BankItem(label=label, text=text)


def load_bank(path: str | Path = MICRO_BANK_PATH) -> list[BankItem]:
    """Читает банк примеров; результат кэшируется по пути файла."""
    key = str(path)
    if cached := _BANK_CACHE.get(key):
        return cached
    file = Path(path)
    if not file.exists():
        raise ValueError(f"Банк примеров не найден: {file}")
    items = [
        _parse_line(line, number)
        for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1)
        if line.strip()
    ]
    if not items:
        raise ValueError(f"Банк примеров пуст: {file}")
    _BANK_CACHE[key] = items
    return items


def bank_labels(items: list[BankItem]) -> dict[str, int]:
    """Сколько примеров каждой метки лежит в банке — нужно и для статуса, и для отчёта."""
    counts: dict[str, int] = {label: 0 for label in MICRO_LABELS}
    for item in items:
        counts[item.label] += 1
    return counts


def bank_digest(items: list[BankItem]) -> str:
    """Отпечаток содержимого банка: по нему понимаем, что кэш векторов устарел."""
    blob = "\n".join(f"{item.label}\t{item.text}" for item in items)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _cache_file(model: str, cache_dir: str | Path) -> Path:
    slug = _SLUG_RE.sub("-", model.lower()).strip("-")
    return Path(cache_dir) / f"{slug}.json"


def _read_cache(file: Path, digest: str, expected: int) -> list[list[float]] | None:
    if not file.exists():
        return None
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or data.get("digest") != digest:
        return None
    vectors = data.get("vectors")
    if not isinstance(vectors, list) or len(vectors) != expected:
        return None
    return [[float(v) for v in row] for row in vectors]


async def bank_vectors(
    items: list[BankItem],
    *,
    model: str = MICRO_EMBED_MODEL,
    cache_dir: str | Path = MICRO_CACHE_DIR,
) -> list[list[float]]:
    """Векторы примеров банка: считаются один раз на модель и складываются в data/micro_cache."""
    digest = bank_digest(items)
    key = f"{model}:{digest}"
    if cached := _VECTOR_CACHE.get(key):
        return cached
    async with _VECTOR_LOCK:
        # Пока ждали лок, банк мог посчитать другой запрос — проверяем ещё раз.
        if cached := _VECTOR_CACHE.get(key):
            return cached
        file = _cache_file(model, cache_dir)
        vectors = _read_cache(file, digest, len(items))
        if vectors is None:
            vectors = await embed_texts_async([item.text for item in items], model=model)
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(
                json.dumps({"model": model, "digest": digest, "vectors": vectors}),
                encoding="utf-8",
            )
        _VECTOR_CACHE[key] = vectors
        return vectors
