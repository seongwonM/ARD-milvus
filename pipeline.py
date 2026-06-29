"""인덱싱 / 검색 파이프라인.

사용 예시:
    from milvus_migration.pipeline import Pipeline

    pipe = Pipeline.from_env()

    # 인덱싱
    docs = [
        {"id": "doc-1", "text": "Milvus는 오픈소스 벡터 데이터베이스입니다."},
        {"id": "doc-2", "text": "PyMilvus는 Python SDK입니다."},
    ]
    pipe.index(docs)

    # 검색
    results = pipe.search("벡터 데이터베이스 추천", top_k=5)
    for doc_id, score in results:
        print(f"{doc_id}: {score:.4f}")
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import Config
from .embedding import EmbeddingClient
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

        docs 형식:
            [{"id": "doc-1", "text": "내용"}, ...]
            또는
            [{"id": "doc-1", "title": "제목", "text": "내용"}, ...]

        Args:
            docs:        인덱싱할 문서 리스트
            collection:  컬렉션 이름 (None이면 config 값 사용)
            vector_mode: "dense" | "sparse"
            recreate:    True이면 기존 컬렉션 삭제 후 재생성
            batch_size:  임베딩 API 배치 크기
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
            text  = doc.get("text", doc.get("chunk", ""))
            return f"{title} {text}".strip()

        def _data_iter():
            n = len(docs)
            for start in range(0, n, _CHUNK):
                chunk = docs[start : start + _CHUNK]
                texts = [_text(d) for d in chunk]

                logger.info(f"  임베딩 중: {start+1}~{start+len(chunk):,}/{n:,}")
                embs = self._emb.encode(texts, batch_size=bs)

                for pid, (doc, vec) in enumerate(zip(chunk, embs), start=start):
                    yield pid, doc["id"], vec

        total = self._store.upload(name, _data_iter(), vector_mode=vector_mode)
        self._store.finalize(name)
        logger.info(f"인덱싱 완료: {total:,}건 → 컬렉션 '{name}'")

    # ── 검색 ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str | list[str],
        *,
        top_k: int = 10,
        collection: str | None = None,
        vector_mode: str = "dense",
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
        """쿼리 텍스트로 유사 문서를 검색.

        단일 쿼리(str) → [(doc_id, score), ...]
        복수 쿼리(list) → [[(doc_id, score), ...], ...]
        """
        name       = collection or self._cfg.milvus.collection
        is_single  = isinstance(query, str)
        texts      = [query] if is_single else query

        vectors = self._emb.encode(texts)
        results = self._store.search(name, vectors, top_k=top_k, vector_mode=vector_mode)

        return results[0] if is_single else results

    # ── 유틸 ─────────────────────────────────────────────────────────────────

    def collection_size(self, collection: str | None = None) -> int:
        name = collection or self._cfg.milvus.collection
        return self._store.collection_size(name)

    def drop_collection(self, collection: str | None = None) -> None:
        name = collection or self._cfg.milvus.collection
        self._store.drop_collection(name)
