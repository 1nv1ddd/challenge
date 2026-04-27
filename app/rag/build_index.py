"""Сборка SQLite-индекса RAG (чанки + эмбеддинги). Используется скриптом и при старте приложения."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from .chunking import ChunkRecord
from .embeddings import DEFAULT_EMBED_MODEL, embed_texts_sync
from .pipeline import (
    build_all_chunks,
    collect_document_paths,
    default_corpus_dir,
    default_index_path,
    project_root,
)
from .index_meta import index_needs_build
from .store import clear_strategy, init_db, insert_chunks

log = logging.getLogger(__name__)


def _write_report(
    fixed: list[ChunkRecord],
    structural: list[ChunkRecord],
    report_path: Path,
    embed_model: str,
) -> None:
    def stats(rows: list[ChunkRecord]) -> dict:
        lens = [len(r["text"]) for r in rows]
        sections = sum(1 for r in rows if r.get("section"))
        return {
            "chunks": len(rows),
            "avg_chars": round(sum(lens) / len(lens), 1) if lens else 0,
            "median_chars": sorted(lens)[len(lens) // 2] if lens else 0,
            "chunks_with_section_meta": sections,
        }

    lines = [
        "# Сравнение стратегий chunking (автоотчёт)",
        "",
        f"Модель эмбеддингов: `{embed_model}`.",
        "",
        "## Фиксированное окно (`fixed`)",
        "",
        "```json",
        json.dumps(stats(fixed), ensure_ascii=False, indent=2),
        "```",
        "",
        "## По структуре (`structural`)",
        "",
        "```json",
        json.dumps(stats(structural), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Как интерпретировать",
        "",
        "- **fixed** — равномерные окна; `section` обычно пустой.",
        "- **structural** — границы по заголовкам Markdown; длинные секции режутся подчанками с одним `section`.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_rag_index(
    *,
    corpus_dir: Path | None = None,
    index_path: Path | None = None,
    include_extra: bool = True,
    write_report: bool = True,
) -> Path:
    """
    Полная пересборка индекса. Нужен ROUTERAI_API_KEY.
    Возвращает путь к SQLite.
    """
    root = project_root()
    corpus = corpus_dir or default_corpus_dir()
    idx = index_path or default_index_path()
    extra: list[Path] = []
    if include_extra:
        readme = root / "README.md"
        if readme.is_file():
            extra.append(readme)
        # Day 31: курированная документация проекта.
        docs_dir = root / "docs"
        if docs_dir.is_dir():
            extra.extend(sorted(p for p in docs_dir.rglob("*.md") if p.is_file()))
        # API-карту и точку входа — в индекс, чтобы /help мог сослаться на роутеры.
        for rel in (
            "app/main.py",
            "app/payloads.py",
            "nginx.conf",
            "nginx-ratelimit.conf",
            "docker-compose.yml",
            "CLAUDE.md",
        ):
            p = root / rel
            if p.is_file():
                extra.append(p)
        routers_dir = root / "app" / "routers"
        if routers_dir.is_dir():
            extra.extend(sorted(p for p in routers_dir.rglob("*.py") if p.is_file()))
        agent_pkg = root / "app" / "agent"
        if agent_pkg.is_dir():
            extra.extend(sorted(p for p in agent_pkg.rglob("*.py") if p.is_file()))
        else:
            legacy_agent = root / "app" / "agent.py"
            if legacy_agent.is_file():
                extra.append(legacy_agent)

    paths = collect_document_paths(corpus, extra_files=extra)
    if not paths:
        raise FileNotFoundError(
            "Нет файлов корпуса: добавьте md/txt в data/rag_corpus/",
        )

    embed_url = os.getenv("RAG_EMBEDDINGS_URL", "")
    if "routerai.ru" in embed_url or not embed_url:
        api_key = (os.getenv("ROUTERAI_API_KEY") or "").strip()
        if not api_key:
            raise ValueError(
                "ROUTERAI_API_KEY не задан — эмбеддинги через RouterAI недоступны. "
                "Для локальной сборки задайте RAG_EMBEDDINGS_URL=http://localhost:11434/v1/embeddings "
                "и RAG_EMBEDDING_MODEL=bge-m3 (нужен `ollama pull bge-m3`)."
            )

    embed_model = os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)

    fixed_chunks, structural_chunks = build_all_chunks(paths, root)
    log.info(
        "RAG build: files=%s chunks fixed=%s structural=%s",
        len(paths),
        len(fixed_chunks),
        len(structural_chunks),
    )

    init_db(idx)
    clear_strategy(idx, "fixed")
    clear_strategy(idx, "structural")

    vec_f = embed_texts_sync([c["text"] for c in fixed_chunks], model=embed_model)
    vec_s = embed_texts_sync([c["text"] for c in structural_chunks], model=embed_model)

    insert_chunks(idx, fixed_chunks, vec_f, embed_model)
    insert_chunks(idx, structural_chunks, vec_s, embed_model)

    if write_report:
        _write_report(
            fixed_chunks,
            structural_chunks,
            idx.parent / "chunking_report.md",
            embed_model,
        )
    return idx


def main_cli(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--no-extra", action="store_true")
    ap.add_argument("--index", type=Path, default=None)
    args = ap.parse_args(argv)

    try:
        idx = build_rag_index(
            corpus_dir=args.corpus,
            index_path=args.index,
            include_extra=not args.no_extra,
        )
        print(f"Индекс: {idx}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)
