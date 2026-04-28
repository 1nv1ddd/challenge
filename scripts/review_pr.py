#!/usr/bin/env python3
"""День 32: AI-ревью PR через RAG + RouterAI.

Использование:
  git diff origin/main...HEAD | python scripts/review_pr.py > review.md

Зависимости: ROUTERAI_API_KEY в env, собранный data/rag_index/chunks.sqlite.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.rag.embeddings import embed_texts_sync  # noqa: E402
from app.rag.retrieve import multi_search_merge, search_cosine  # noqa: E402
from app.rag.store import load_matrix_for_strategy  # noqa: E402

INDEX = ROOT / "data" / "rag_index" / "chunks.sqlite"
DIFF_LIMIT = 30_000  # символов diff-а в промпт
CHUNK_TEXT_LIMIT = 1200  # символов на чанк
MAX_CHUNKS = 10
ROUTERAI_URL = "https://routerai.ru/api/v1/chat/completions"
LLM_MODEL = os.environ.get("REVIEW_MODEL", "openai/gpt-4o-mini")


_DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.MULTILINE)


def read_diff() -> str:
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return os.environ.get("PR_DIFF", "")


def parse_changed_files(diff: str) -> list[str]:
    seen: list[str] = []
    for m in _DIFF_FILE_RE.finditer(diff):
        path = m.group(2)
        if path not in seen:
            seen.append(path)
    return seen


def rag_context(diff: str, files: list[str]) -> list[dict]:
    """Несколько подзапросов: путь файла + первые ~400 символов его hunk'а."""
    if not INDEX.is_file():
        return []
    queries: list[str] = []
    for path in files[:6]:
        # Возьмём короткий контекст из diff'а вокруг этого файла
        snippet = ""
        marker = f"diff --git a/{path} b/{path}"
        i = diff.find(marker)
        if i >= 0:
            snippet = diff[i : i + 800]
        queries.append(f"{path}\n{snippet}".strip())
    if not queries:
        queries = [diff[:1500]]

    try:
        vecs = embed_texts_sync(queries)
    except Exception as e:  # noqa: BLE001
        print(f"::warning::RAG embed failed: {e}", file=sys.stderr)
        return []

    chunks_meta, matrix = load_matrix_for_strategy(INDEX, "structural")
    if len(vecs) == 1:
        hits = search_cosine(vecs[0], chunks_meta, matrix, top_k=MAX_CHUNKS)
    else:
        hits = multi_search_merge(
            vecs, chunks_meta, matrix, per_k=4, max_chunks=MAX_CHUNKS
        )
    return hits


def build_prompt(diff: str, files: list[str], hits: list[dict]) -> str:
    if hits:
        rag_block = "\n\n".join(
            f"### {h.get('source', '?')} · {h.get('section', '—')}\n"
            f"{(h.get('text') or '')[:CHUNK_TEXT_LIMIT].strip()}"
            for h in hits
        )
    else:
        rag_block = "(RAG-индекс пуст или недоступен — отвечай только по diff'у.)"

    files_block = "\n".join(f"- `{f}`" for f in files) or "(файлов не найдено)"
    diff_block = diff[:DIFF_LIMIT]
    if len(diff) > DIFF_LIMIT:
        diff_block += f"\n\n... [обрезано, всего {len(diff)} символов]"

    return f"""Ты — опытный code reviewer этого репозитория (Python, FastAPI, RAG, Ollama, MCP).

Используй фрагменты ниже («Контекст из RAG») чтобы понять архитектуру и существующие соглашения, прежде чем оценивать diff. Если в RAG-контексте есть противоречие с diff'ом — отметь.

## Контекст из RAG
{rag_block}

## Изменённые файлы
{files_block}

## Diff
```diff
{diff_block}
```

## Задание
Дай **краткое** ревью на русском в трёх блоках. Каждый — список из 0–5 пунктов, по одному факту на пункт. Указывай файл и (если возможно) номер строки.

### 🐞 Потенциальные баги
Реальные ошибки/edge cases/race conditions/security issues. Если ничего серьёзного — напиши «не нашёл».

### 🏗 Архитектурные проблемы
Нарушения слоёв проекта (см. RAG-контекст), дублирование, неправильные зависимости, плохие границы абстракций. Если всё ок — «не нашёл».

### 💡 Рекомендации
Читаемость, тесты, перформанс, naming, dead code. До 5 пунктов.

Без воды, без вступлений типа «отличное PR», без перечисления того, что просто работает. Только то, на что **стоит обратить внимание**.
"""


def call_llm(prompt: str) -> str:
    key = (os.environ.get("ROUTERAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("ROUTERAI_API_KEY не задан")
    r = httpx.post(
        ROUTERAI_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=180,
    )
    r.raise_for_status()
    payload = r.json()
    return payload["choices"][0]["message"]["content"].strip()


def main() -> int:
    diff = read_diff().strip()
    if not diff:
        print("Diff пустой — нечего ревьюить.")
        return 0

    files = parse_changed_files(diff)
    hits = rag_context(diff, files)
    prompt = build_prompt(diff, files, hits)

    try:
        review = call_llm(prompt)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ AI-ревью упало: {e}")
        return 1

    n_chunks = len(hits)
    n_files = len(files)
    print(f"# 🤖 AI Review\n")
    print(f"_GPT-4o-mini · {n_files} файлов · {n_chunks} RAG-чанков из проекта_\n")
    print(review)
    return 0


if __name__ == "__main__":
    sys.exit(main())
