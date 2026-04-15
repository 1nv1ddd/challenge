from __future__ import annotations

import os
from pathlib import Path

from ..providers import Message

_DEFAULT_MIN_SIM = float(os.getenv("RAG_MIN_SIMILARITY", "0") or "0")


class AgentRagMixin:
    def _format_rag_hits_block(self, label: str, hits: list[dict]) -> str:
        lines = [f"### Набор отрывков: **{label}**", ""]
        for i, h in enumerate(hits, 1):
            sec = h.get("section") or "—"
            lines.append(
                f"**[{i}]** score={float(h['score']):.4f} | файл `{h['source']}` | section: {sec}"
            )
            lines.append("")
            lines.append(str(h.get("text", ""))[:3200].strip())
            lines.append("")
        return "\n".join(lines)

    async def _rag_context_message(
        self,
        user_content: str,
        rag_cfg: dict | None,
    ) -> tuple[Message | None, dict]:
        if not rag_cfg or not rag_cfg.get("enabled"):
            return None, {}
        from ..rag import pipeline as rag_pipeline
        from ..rag.embeddings import embed_texts_async
        from ..rag.post_retrieval import postprocess_hits, rag_enhancements_enabled
        from ..rag.query_rewrite import heuristic_query_rewrite
        from ..rag.query_split import rag_subqueries
        from ..rag.anchors import rag_keyword_needles as _rag_kw_needles
        from ..rag.retrieve import augment_hits_with_keyword_match, multi_search_merge, search_cosine
        from ..rag.store import load_matrix_for_strategy

        raw_path = rag_cfg.get("index_path")
        idx = Path(raw_path) if raw_path else rag_pipeline.default_index_path()
        if not idx.is_file():
            return (
                Message(
                    role="system",
                    content=(
                        "[RAG] Локальный индекс не найден. Соберите: "
                        "`python scripts/build_rag_index.py` (нужен ROUTERAI_API_KEY)."
                    ),
                ),
                {"rag_error": "index_missing"},
            )
        mode = str(rag_cfg.get("strategy") or "fixed").lower().strip()
        top_k = int(rag_cfg.get("top_k") or 5)
        enhanced = rag_enhancements_enabled(rag_cfg)

        text_for_subq = user_content
        rewrite_meta: dict = {}
        if enhanced and rag_cfg.get("query_rewrite"):
            text_for_subq, rewrite_meta = heuristic_query_rewrite(user_content)

        subqs = rag_subqueries(text_for_subq)

        try:
            raw_min = rag_cfg.get("min_similarity")
            min_sim = float(raw_min) if raw_min is not None else _DEFAULT_MIN_SIM
        except (TypeError, ValueError):
            min_sim = _DEFAULT_MIN_SIM
        if not enhanced:
            min_sim = 0.0

        rerank_m = str(rag_cfg.get("rerank") or "none").strip().lower() if enhanced else "none"

        try:
            tkf = rag_cfg.get("top_k_fetch")
            top_k_fetch = int(tkf) if tkf is not None else None
        except (TypeError, ValueError):
            top_k_fetch = None

        k_search = (
            (top_k_fetch if top_k_fetch is not None else max(top_k * 3, 18))
            if enhanced
            else top_k
        )

        meta_out: dict = {
            "rag_mode": mode,
            "rag_top_k": top_k,
            "rag_subqueries": len(subqs),
            "rag_keyword_needles": len(_rag_kw_needles(user_content)),
            "rag_day23_enhanced": enhanced,
            "rag_top_k_fetch": k_search if enhanced else top_k,
            "rag_min_similarity": min_sim if enhanced else 0.0,
            "rag_rerank": rerank_m if enhanced else "none",
            "rag_query_rewrite": bool(enhanced and rag_cfg.get("query_rewrite")),
        }
        if rewrite_meta:
            meta_out["rag_rewrite_meta"] = rewrite_meta

        try:
            q_vecs = await embed_texts_async(subqs)
        except Exception as e:  # noqa: BLE001
            return (
                Message(role="system", content=f"[RAG] Ошибка эмбеддинга запроса: {e}"),
                {"rag_error": str(e)},
            )

        def _shrink(hits: list[dict]) -> list[dict]:
            return [
                {
                    "chunk_id": h["chunk_id"],
                    "source": h["source"],
                    "section": h.get("section"),
                    "score": round(float(h["score"]), 4),
                }
                for h in hits
            ]

        def _after_augment(hits: list[dict], strategy: str) -> tuple[list[dict], dict]:
            if not enhanced:
                return hits, {}
            cap = min(40 if enhanced else 28, k_search + max(len(subqs), 8) + (10 if enhanced else 6))
            aug = augment_hits_with_keyword_match(
                idx, strategy, user_content, hits, max_total=cap
            )
            return postprocess_hits(
                aug,
                user_content,
                top_k_final=top_k,
                min_similarity=min_sim,
                rerank_mode=rerank_m,
            )

        _rag_grounding = (
            "ПРАВИЛА (обязательны):\n"
            "- **RAG** в этом приложении = **Retrieval-Augmented Generation** (поиск по локальному индексу текстов), "
            "не «Red-Amber-Green» и не другие расшифровки.\n"
            "- **MCP** в этом репозитории = **Model Context Protocol** (см. отрывки из кода/README).\n"
            "- Набор отрывков включает **семантический поиск** и чанки с **точным вхождением** кодов из запроса "
            "(PL-*-RET, NODE-Q-*, константы в пакете app/agent). Если отрывок явно отвечает на подвопрос — используй его.\n"
            "- Отвечай **только** по фактам из блоков «Набор отрывков» ниже. Не дополняй «типичными советами» и общими шаблонами.\n"
            "- Если точного ответа в отрывках нет — напиши явно: «В переданных фрагментах нет ответа на …» и не выдумывай цифры, контакты и SLA.\n"
            "- Укажи **номер отрывка [n]** и **файл** `source`, откуда взял факт.\n\n"
        )

        kw_ctx_cap = min(28, top_k + max(len(subqs), 8) + 6)

        if mode == "compare":
            mf, matf = load_matrix_for_strategy(idx, "fixed")
            ms, mats = load_matrix_for_strategy(idx, "structural")
            if len(subqs) == 1:
                hf = search_cosine(q_vecs[0], mf, matf, top_k=k_search)
                hs = search_cosine(q_vecs[0], ms, mats, top_k=k_search)
            else:
                per_k = max(5, min(k_search + 3, (k_search * 3) // max(1, len(subqs)) + 5))
                max_merged = min(48 if enhanced else 26, k_search + len(subqs) + (8 if enhanced else 6))
                hf = multi_search_merge(q_vecs, mf, matf, per_k=per_k, max_chunks=max_merged)
                hs = multi_search_merge(q_vecs, ms, mats, per_k=per_k, max_chunks=max_merged)
            if enhanced:
                hf, pp_f = _after_augment(hf, "fixed")
                hs, pp_s = _after_augment(hs, "structural")
                meta_out["rag_postprocess_fixed"] = pp_f
                meta_out["rag_postprocess_structural"] = pp_s
            else:
                hf = augment_hits_with_keyword_match(
                    idx, "fixed", user_content, hf, max_total=kw_ctx_cap
                )
                hs = augment_hits_with_keyword_match(
                    idx, "structural", user_content, hs, max_total=kw_ctx_cap
                )
            meta_out["rag_hits_fixed"] = _shrink(hf)
            meta_out["rag_hits_structural"] = _shrink(hs)
            mq_note = (
                f"Запрос разбит на **{len(subqs)}** подвопросов для поиска; для каждого — отдельный эмбеддинг, "
                "результаты объединены (лучший score на чанк).\n\n"
                if len(subqs) > 1
                else ""
            )
            day23_note = ""
            if enhanced:
                day23_note = (
                    f"\n_(День 23: поиск с top_k_fetch≈{k_search}, затем порог {min_sim}, реранк: {rerank_m})_\n\n"
                )
            intro = (
                _rag_grounding
                + day23_note
                + mq_note
                + "Ниже — фрагменты из **локального корпуса** (учебный справочник + README + код `app/agent`), "
                "подобранные **семантическим поиском** по двум стратегиям chunking:\n"
                "- **fixed** — окна фиксированной длины;\n"
                "- **structural** — по заголовкам Markdown.\n\n"
                "Сначала ответь на вопрос пользователя **строго по отрывкам**. "
                "Затем добавь блок **«Сравнение chunking»** (2–5 предложений): какой набор отрывков полезнее для этого вопроса.\n\n"
            )
            body = "\n".join(
                [
                    intro,
                    self._format_rag_hits_block("fixed", hf),
                    self._format_rag_hits_block("structural", hs),
                ]
            )
            return Message(role="system", content=body), meta_out

        strat = "fixed" if mode == "fixed" else "structural"
        m, matrix = load_matrix_for_strategy(idx, strat)
        if len(subqs) == 1:
            h = search_cosine(q_vecs[0], m, matrix, top_k=k_search)
        else:
            per_k = max(5, min(k_search + 3, (k_search * 3) // max(1, len(subqs)) + 5))
            max_merged = min(48 if enhanced else 26, k_search + len(subqs) + (8 if enhanced else 6))
            h = multi_search_merge(q_vecs, m, matrix, per_k=per_k, max_chunks=max_merged)
        pp_one: dict = {}
        if enhanced:
            h, pp_one = _after_augment(h, strat)
            meta_out["rag_postprocess"] = pp_one
        else:
            h = augment_hits_with_keyword_match(
                idx, strat, user_content, h, max_total=kw_ctx_cap
            )
        meta_out[f"rag_hits_{strat}"] = _shrink(h)
        mq_note = (
            (
                f"Запрос разбит на **{len(subqs)}** подвопросов; ниже до **{len(h)}** объединённых отрывков "
                f"(chunking: **{strat}**).\n\n"
            )
            if len(subqs) > 1
            else f"Ниже — топ-{top_k} отрывков из локального индекса (chunking: **{strat}**).\n\n"
        )
        day23_note = ""
        if enhanced:
            day23_note = (
                f"_(День 23: отбор с fetch≈{k_search}, min_sim={min_sim}, rerank={rerank_m})_\n\n"
            )
        intro = _rag_grounding + day23_note + mq_note
        body = intro + self._format_rag_hits_block(strat, h)
        return Message(role="system", content=body), meta_out
