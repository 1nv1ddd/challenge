"""Сбор файлов корпуса и построение чанков обеими стратегиями."""

from __future__ import annotations

from pathlib import Path

from .chunking import ChunkRecord, chunk_fixed_size, chunk_structural


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def default_index_path() -> Path:
    return project_root() / "data" / "rag_index" / "chunks.sqlite"


def default_corpus_dir() -> Path:
    return project_root() / "data" / "rag_corpus"


def collect_document_paths(
    corpus_dir: Path,
    *,
    extra_files: list[Path] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    if corpus_dir.is_dir():
        for p in sorted(corpus_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".md", ".markdown", ".txt"}:
                paths.append(p)
    for p in extra_files or []:
        if p.is_file() and p not in paths:
            paths.append(p)
    return paths


def path_to_source(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def build_chunks_for_file(
    path: Path,
    root: Path,
) -> tuple[list[ChunkRecord], list[ChunkRecord]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    source = path_to_source(path, root)
    title = path.name
    is_md = path.suffix.lower() in {".md", ".markdown"}

    fixed = chunk_fixed_size(text, source=source, title=title)
    structural = chunk_structural(
        text,
        source=source,
        title=title,
        is_markdown=is_md,
    )
    return fixed, structural


def build_all_chunks(paths: list[Path], root: Path) -> tuple[list[ChunkRecord], list[ChunkRecord]]:
    all_f: list[ChunkRecord] = []
    all_s: list[ChunkRecord] = []
    for p in paths:
        f, s = build_chunks_for_file(p, root)
        all_f.extend(f)
        all_s.extend(s)
    return all_f, all_s
