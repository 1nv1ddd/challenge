"""Бэкенды micro-model: kNN по эмбеддингам и tfidf по символьным n-граммам на numpy."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..agent_constants import (
    MICRO_BACKENDS,
    MICRO_BANK_PATH,
    MICRO_EMBED_MODEL,
    MICRO_KNN_K,
    MICRO_TFIDF_NGRAMS,
)
from ..rag.embeddings import embed_texts_async
from .bank import BankItem, bank_vectors, load_bank
from .schema import Neighbor

_NON_WORD_RE = re.compile(r"[^\w]+", re.U)
# Бэкенды строятся один раз на (тип, банк, модель): банк маленький, пересчитывать его на запрос глупо.
_BACKEND_CACHE: dict[str, "MicroBackend"] = {}


def normalize_text(text: str) -> str:
    """Приведение текста к канону для tfidf: регистр, «ё» и пунктуация не должны различать примеры."""
    lowered = (text or "").lower().replace("ё", "е")
    return " ".join(_NON_WORD_RE.sub(" ", lowered).split())


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


class MicroBackend(ABC):
    """Уровень 1 инференса: находит ближайшие размеченные примеры без обращения к большой модели."""

    name: str

    def __init__(self, items: list[BankItem]) -> None:
        self.items = items

    @abstractmethod
    async def similarities(self, text: str) -> np.ndarray:
        """Близость запроса к каждому примеру банка, в порядке примеров."""

    async def neighbors(self, text: str, k: int = MICRO_KNN_K) -> list[Neighbor]:
        """Top-k ближайших примеров банка по убыванию близости."""
        sims = await self.similarities(text)
        top = np.argsort(-sims)[: max(1, k)]
        return [
            Neighbor(label=self.items[i].label, similarity=float(sims[i]), text=self.items[i].text)
            for i in top
        ]


class EmbedBackend(MicroBackend):
    """kNN по эмбеддингам: один дешёвый вызов /v1/embeddings вместо вызова большой модели."""

    name = "embed"

    def __init__(self, items: list[BankItem], vectors: list[list[float]], model: str) -> None:
        super().__init__(items)
        self.model = model
        self.matrix = _l2_normalize(np.asarray(vectors, dtype=np.float32))

    async def similarities(self, text: str) -> np.ndarray:
        query = await embed_texts_async([text], model=self.model)
        vector = np.asarray(query[0], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return np.zeros(len(self.items), dtype=np.float32)
        return self.matrix @ (vector / norm)


class TfidfBackend(MicroBackend):
    """Символьные n-граммы + tfidf на numpy: полностью офлайн, без единого сетевого вызова."""

    name = "tfidf"

    def __init__(self, items: list[BankItem], ngrams: tuple[int, int] = MICRO_TFIDF_NGRAMS) -> None:
        super().__init__(items)
        self.ngrams = ngrams
        docs = [self._ngrams(item.text) for item in items]
        vocab: dict[str, int] = {}
        for doc in docs:
            for gram in doc:
                vocab.setdefault(gram, len(vocab))
        self.vocab = vocab
        counts = np.zeros((len(docs), len(vocab)), dtype=np.float32)
        for row, doc in enumerate(docs):
            for gram in doc:
                counts[row, vocab[gram]] += 1.0
        doc_freq = np.count_nonzero(counts, axis=0)
        self.idf = np.log((1.0 + len(docs)) / (1.0 + doc_freq)).astype(np.float32) + 1.0
        self.matrix = _l2_normalize(self._sublinear(counts) * self.idf)

    def _ngrams(self, text: str) -> list[str]:
        padded = f" {normalize_text(text)} "
        low, high = self.ngrams
        return [
            padded[i : i + size]
            for size in range(low, high + 1)
            for i in range(len(padded) - size + 1)
        ]

    @staticmethod
    def _sublinear(counts: np.ndarray) -> np.ndarray:
        """Сублинейный tf: во фразе важен сам факт n-граммы, а не то, что она повторилась пять раз."""
        return np.where(counts > 0.0, 1.0 + np.log(np.maximum(counts, 1.0)), 0.0)

    async def similarities(self, text: str) -> np.ndarray:
        vector = np.zeros(len(self.vocab), dtype=np.float32)
        for gram in self._ngrams(text):
            index = self.vocab.get(gram)
            if index is not None:
                vector[index] += 1.0
        vector = self._sublinear(vector.reshape(1, -1))[0] * self.idf
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return np.zeros(len(self.items), dtype=np.float32)
        return self.matrix @ (vector / norm)


async def get_backend(
    name: str,
    *,
    bank_path: str | Path = MICRO_BANK_PATH,
    embed_model: str = MICRO_EMBED_MODEL,
) -> MicroBackend:
    """Готовый бэкенд по имени; банк и его векторы считаются один раз на процесс."""
    if name not in MICRO_BACKENDS:
        raise ValueError(f"Неизвестный бэкенд «{name}»; доступны: {', '.join(MICRO_BACKENDS)}.")
    key = f"{name}:{bank_path}:{embed_model if name == 'embed' else '-'}"
    if cached := _BACKEND_CACHE.get(key):
        return cached
    items = load_bank(bank_path)
    if name == "tfidf":
        backend: MicroBackend = TfidfBackend(items)
    else:
        backend = EmbedBackend(items, await bank_vectors(items, model=embed_model), embed_model)
    _BACKEND_CACHE[key] = backend
    return backend
