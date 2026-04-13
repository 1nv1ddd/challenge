#!/usr/bin/env python3
"""
Сборка локального RAG-индекса (День 21). Реализация: app.rag.build_index.

 python scripts/build_rag_index.py
  python scripts/build_rag_index.py --corpus data/rag_corpus --no-extra
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.build_index import main_cli  # noqa: E402

if __name__ == "__main__":
    main_cli()
