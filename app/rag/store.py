"""SQLite: чанки + эмбеддинги (float32 blob)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import ChunkRecord


def _blob_from_vec(vec: list[float]) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes()


def _vec_from_blob(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                strategy TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                section TEXT,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                embed_model TEXT NOT NULL,
                PRIMARY KEY (strategy, chunk_id)
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks(strategy)")
        con.commit()
    finally:
        con.close()


def clear_strategy(path: Path, strategy: str) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute("DELETE FROM chunks WHERE strategy = ?", (strategy,))
        con.commit()
    finally:
        con.close()


def insert_chunks(
    path: Path,
    records: list[ChunkRecord],
    embeddings: list[list[float]],
    embed_model: str,
) -> None:
    if len(records) != len(embeddings):
        raise ValueError("records и embeddings разной длины")
    con = sqlite3.connect(path)
    try:
        for rec, vec in zip(records, embeddings, strict=True):
            dim = len(vec)
            con.execute(
                """
                INSERT OR REPLACE INTO chunks
                (strategy, chunk_id, source, title, section, text, embedding, dim, embed_model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec["strategy"],
                    rec["chunk_id"],
                    rec["source"],
                    rec["title"],
                    rec["section"],
                    rec["text"],
                    _blob_from_vec(vec),
                    dim,
                    embed_model,
                ),
            )
        con.commit()
    finally:
        con.close()


def load_matrix_for_strategy(path: Path, strategy: str) -> tuple[list[ChunkRecord], np.ndarray]:
    con = sqlite3.connect(path)
    try:
        cur = con.execute(
            """
            SELECT chunk_id, source, title, section, text, strategy, embedding, dim
            FROM chunks WHERE strategy = ?
            """,
            (strategy,),
        )
        rows = cur.fetchall()
    finally:
        con.close()
    if not rows:
        return [], np.zeros((0, 1), dtype=np.float32)
    dim = int(rows[0][7])
    meta: list[ChunkRecord] = []
    mats: list[np.ndarray] = []
    for cid, src, title, section, text, strat, blob, d in rows:
        if int(d) != dim:
            raise ValueError("разная размерность в БД")
        meta.append(
            {
                "chunk_id": cid,
                "source": src,
                "title": title,
                "section": section,
                "text": text,
                "strategy": strat,
            }
        )
        mats.append(_vec_from_blob(blob, dim))
    mat = np.stack(mats, axis=0)
    return meta, mat


def fetch_chunks_by_substrings(
    path: Path,
    strategy: str,
    needles: list[str],
    *,
    per_needle_limit: int = 5,
    max_total: int = 14,
) -> list[dict[str, Any]]:
    """
    Подобрать чанки, в тексте которых встречается хотя бы одна подстрока (LIKE %needle%).
    Регистронезависимо через LOWER(text).
    """
    if not needles or not path.is_file():
        return []
    esc = str.maketrans({"%": r"\%", "_": r"\_"})
    con = sqlite3.connect(path)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    try:
        for needle in needles:
            if not needle or len(needle.strip()) < 3:
                continue
            pat = f"%{needle.translate(esc)}%"
            cur = con.execute(
                """
                SELECT chunk_id, source, title, section, text, strategy
                FROM chunks
                WHERE strategy = ? AND LOWER(text) LIKE LOWER(?) ESCAPE '\\'
                LIMIT ?
                """,
                (strategy, pat, per_needle_limit),
            )
            for cid, src, title, section, text, strat in cur.fetchall():
                if cid in seen:
                    continue
                seen.add(cid)
                out.append(
                    {
                        "chunk_id": cid,
                        "source": src,
                        "title": title,
                        "section": section,
                        "text": (text or "")[:4000],
                        "strategy": strat,
                    }
                )
                if len(out) >= max_total:
                    return out
    finally:
        con.close()
    return out


def stats(path: Path) -> dict[str, Any]:
    from .index_meta import rag_index_stats

    return rag_index_stats(path)
