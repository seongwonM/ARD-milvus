"""인덱싱 / 검색 파이프라인.

사용 예시:
    pipe = Pipeline.from_env()

    # 인덱싱 (dense)
    pipe.index(docs)

    # 인덱싱 (sparse — API 미지원 시 SparseUnsupported 발생)
    pipe.index(docs, vector_mode="sparse")

    # 검색
    results = pipe.search("쿼리", top_k=5)

    # ColBERT 리랭킹 (dense 100개 후보 → ColBERT 재정렬)
    results = pipe.search("쿼리", top_k=5, with_colbert_rerank=True)

    # 8B 모델 파이프라인
    pipe_8b = Pipeline.for_8b()
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np

from .config import Config, EmbeddingConfig
from .embedding import EmbeddingClient, SparseUnsupported, ColBERTUnsupported
from .milvus_store import MilvusStore

_CHUNK = 2_000

logger = logging.getLogger(__name__)


class Pipeline:

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._store = MilvusStore(config.milvus.uri, config.milvus.token)
        self._emb = EmbeddingClient(config.embedding)

    @classmethod
    def from_env(cls) -> "Pipeline":
        return cls(Config.from_env())

    @classmethod
    def for_8b(cls) -> "Pipeline":
        """8B 모델을 사용하는 파이프라인. EMBEDDING_MODEL_8B 미설정 시 ValueError."""
        cfg = Config.from_env()
        if not cfg.embedding.model_8b:
            raise ValueError("EMBEDDING_MODEL_8B 환경변수가 설정되지 않았습니다.")
        emb_8b = EmbeddingConfig(
            endpoint=cfg.embedding.endpoint,
            api_key=cfg.embedding.api_key,
            model=cfg.embedding.model_8b,
            dim=cfg.embedding.dim_8b,
            batch_size=cfg.embedding.batch_size,
            timeout=cfg.embedding.timeout,
        )
        return cls(Config(milvus=cfg.milvus, embedding=emb_8b))

    # ── 인덱싱 ───────────────────────────────────────────────────────────────

    def index(
        self,
        docs: list[dict[str, Any]],
        *,
        collection: str | None = None,
        vector_mode: str = "dense",
        recreate: bool = False,
        batch_size: int | None = None,
    ) -> None:
        """문서 목록을 임베딩 후 Milvus에 저장.

        docs 형식: [{"id": "doc-1", "text": "내용"}, ...]
        vector_mode: "dense" | "sparse"
        """
        name = collection or self._cfg.milvus.collection
        dim  = self._cfg.embedding.dim
        bs   = batch_size or self._cfg.embedding.batch_size

        if recreate and self._store.has_collection(name):
            logger.info(f"기존 컬렉션 삭제: {name}")
            self._store.drop_collection(name)

        if not self._store.has_collection(name):
            self._store.create_collection(name, dim=dim, vector_mode=vector_mode)

        def _text(doc: dict) -> str:
            title = doc.get("title", "")
            body  = doc.get("text", doc.get("chunk", ""))
            return f"{title} {body}".strip()

        def _data_iter():
            n = len(docs)
            for start in range(0, n, _CHUNK):
                chunk = docs[start: start + _CHUNK]
                texts = [_text(d) for d in chunk]

                logger.info(f"  임베딩 중: {start+1}~{start+len(chunk):,}/{n:,}")
                if vector_mode == "sparse":
                    vecs = self._emb.encode_sparse(texts, batch_size=bs)
                else:
                    vecs = self._emb.encode(texts, batch_size=bs)

                for pid, (doc, vec) in enumerate(zip(chunk, vecs), start=start):
                    yield pid, doc["id"], _text(doc), vec

        total = self._store.upload(name, _data_iter(), vector_mode=vector_mode)
        self._store.finalize(name)
        logger.info(f"인덱싱 완료: {total:,}건 → '{name}'")

    # ── 검색 ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str | list[str],
        *,
        top_k: int = 10,
        collection: str | None = None,
        vector_mode: str = "dense",
        with_colbert_rerank: bool = False,
        colbert_top_n: int = 100,
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
        """쿼리 텍스트로 유사 문서 검색.

        단일 쿼리(str) → [(doc_id, score), ...]
        복수 쿼리(list) → [[(doc_id, score), ...], ...]

        with_colbert_rerank=True: dense/sparse로 colbert_top_n 후보 수집 후 ColBERT 리랭킹.
        ColBERT API 미지원 시 경고 후 원래 순서 결과 반환.
        """
        name      = collection or self._cfg.milvus.collection
        is_single = isinstance(query, str)
        texts     = [query] if is_single else query

        if vector_mode == "sparse":
            vectors = self._emb.encode_sparse(texts)
        else:
            vectors = self._emb.encode(texts)

        if with_colbert_rerank:
            results = self._search_colbert(name, texts, vectors, top_k, vector_mode, colbert_top_n)
        else:
            results = self._store.search(name, vectors, top_k=top_k, vector_mode=vector_mode)

        return results[0] if is_single else results

    def _search_colbert(
        self,
        name: str,
        query_texts: list[str],
        vectors,
        top_k: int,
        vector_mode: str,
        colbert_top_n: int,
    ) -> list[list[tuple[str, float]]]:
        fetch_n = max(top_k, colbert_top_n)
        candidates_per_q = self._store.search_with_text(
            name, vectors, top_k=fetch_n, vector_mode=vector_mode
        )

        try:
            # query + 모든 후보 텍스트를 한 번에 ColBERT 인코딩
            cand_texts = [text for cands in candidates_per_q for _, _, text in cands]
            all_tok_vecs = self._emb.encode_colbert(query_texts + cand_texts)

            q_tok_vecs  = all_tok_vecs[:len(query_texts)]
            d_tok_vecs  = all_tok_vecs[len(query_texts):]

            results: list[list[tuple[str, float]]] = []
            offset = 0
            for i, cands in enumerate(candidates_per_q):
                n = len(cands)
                reranked = MilvusStore.colbert_rerank(
                    cands, q_tok_vecs[i], d_tok_vecs[offset: offset + n], top_k
                )
                results.append(reranked)
                offset += n

        except ColBERTUnsupported as exc:
            logger.warning(f"ColBERT 미지원 → ANN 결과 그대로 반환: {exc}")
            results = [
                [(doc_id, score) for doc_id, score, _ in cands[:top_k]]
                for cands in candidates_per_q
            ]

        return results

    # ── 유틸 ─────────────────────────────────────────────────────────────────

    def collection_size(self, collection: str | None = None) -> int:
        return self._store.collection_size(collection or self._cfg.milvus.collection)

    def drop_collection(self, collection: str | None = None) -> None:
        self._store.drop_collection(collection or self._cfg.milvus.collection)
