"""Milvus VectorStore - dense / sparse 지원, ColBERT 리랭킹 포함.

컬렉션 스키마:
  id      INT64 (PK)
  doc_id  VARCHAR(512)
  text    VARCHAR(2048)   ← ColBERT 리랭킹 시 재인코딩에 사용
  vector  FLOAT_VECTOR(dim)  또는  SPARSE_FLOAT_VECTOR
"""
from __future__ import annotations

import logging
import time

import numpy as np

_INSERT_BATCH = 64
_TEXT_MAX_LEN = 2048

logger = logging.getLogger(__name__)


class MilvusStore:

    def __init__(self, uri: str, token: str = "") -> None:
        from pymilvus import MilvusClient

        kwargs: dict = {"uri": uri, "timeout": 300}
        if token:
            kwargs["token"] = token
        self._client = MilvusClient(**kwargs)
        self._uri   = uri
        self._token = token
        logger.info(f"[Milvus] 연결: {uri}")

    # ── 컬렉션 관리 ──────────────────────────────────────────────────────────

    def has_collection(self, name: str) -> bool:
        return self._client.has_collection(name)

    def collection_size(self, name: str) -> int:
        stats = self._client.get_collection_stats(name)
        return int(stats.get("row_count", 0))

    def create_collection(self, name: str, dim: int, vector_mode: str = "dense") -> None:
        from pymilvus import MilvusClient, DataType

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id",     DataType.INT64,   is_primary=True)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=512)
        schema.add_field("text",   DataType.VARCHAR, max_length=_TEXT_MAX_LEN)

        index_params = MilvusClient.prepare_index_params()

        if vector_mode == "sparse":
            schema.add_field("vector", DataType.SPARSE_FLOAT_VECTOR)
            index_params.add_index(
                field_name="vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )
        else:
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 100},
            )

        self._client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"  컬렉션 생성: {name}  mode={vector_mode}  dim={dim}")

    def drop_collection(self, name: str) -> None:
        self._client.drop_collection(name)
        logger.info(f"  컬렉션 삭제: {name}")

    # ── 데이터 삽입 ──────────────────────────────────────────────────────────

    def upload(self, name: str, data_iter, vector_mode: str = "dense") -> int:
        """(id, doc_id, text, vector) 이터레이터를 받아 Milvus에 삽입.

        dense  : vector = np.ndarray (dim,)
        sparse : vector = dict[int, float]
        """
        batch: list[dict] = []
        total = 0

        for pid, doc_id, text, vec in data_iter:
            if vector_mode == "sparse":
                row = {
                    "id": pid,
                    "doc_id": doc_id,
                    "text": text[:_TEXT_MAX_LEN],
                    "vector": {int(k): float(v) for k, v in vec.items()},
                }
            else:
                row = {
                    "id": pid,
                    "doc_id": doc_id,
                    "text": text[:_TEXT_MAX_LEN],
                    "vector": vec.tolist(),
                }
            batch.append(row)

            if len(batch) >= _INSERT_BATCH:
                self._insert_with_retry(name, batch)
                total += len(batch)
                batch.clear()

        if batch:
            self._insert_with_retry(name, batch)
            total += len(batch)

        logger.info(f"  삽입 완료: {total:,}건")
        return total

    def _insert_with_retry(self, name: str, batch: list[dict]) -> None:
        for attempt in range(3):
            try:
                self._client.insert(collection_name=name, data=batch)
                return
            except Exception as exc:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                logger.warning(f"  insert 실패, {wait}s 후 재시도: {exc}")
                time.sleep(wait)

    def finalize(self, name: str) -> None:
        self._client.load_collection(name)
        logger.info(f"  인덱스 로드 완료: {name}")

    # ── 검색 ─────────────────────────────────────────────────────────────────

    def search(
        self,
        name: str,
        vectors,
        top_k: int = 10,
        vector_mode: str = "dense",
    ) -> list[list[tuple[str, float]]]:
        """ANN 검색. 반환: [[("doc_id", score), ...], ...]"""
        return [
            [(doc_id, score) for doc_id, score, _ in hits]
            for hits in self._search_raw(name, vectors, top_k, vector_mode)
        ]

    def search_with_text(
        self,
        name: str,
        vectors,
        top_k: int = 10,
        vector_mode: str = "dense",
    ) -> list[list[tuple[str, float, str]]]:
        """ANN 검색 + text 반환 (ColBERT 리랭킹용).

        반환: [[("doc_id", score, "text"), ...], ...]
        """
        return self._search_raw(name, vectors, top_k, vector_mode)

    def _search_raw(
        self,
        name: str,
        vectors,
        top_k: int,
        vector_mode: str,
    ) -> list[list[tuple[str, float, str]]]:
        CHUNK = 256
        all_results: list[list[tuple[str, float, str]]] = []

        for start in range(0, len(vectors), CHUNK):
            end = min(start + CHUNK, len(vectors))

            if vector_mode == "sparse":
                query_data = [
                    {int(k): float(v) for k, v in vectors[i].items()}
                    for i in range(start, end)
                ]
                results = self._client.search(
                    collection_name=name,
                    data=query_data,
                    anns_field="vector",
                    search_params={"metric_type": "IP", "params": {}},
                    limit=top_k,
                    output_fields=["doc_id", "text"],
                )
            else:
                query_data = [vectors[i].tolist() for i in range(start, end)]
                results = self._client.search(
                    collection_name=name,
                    data=query_data,
                    anns_field="vector",
                    search_params={"metric_type": "COSINE", "params": {"ef": 100}},
                    limit=top_k,
                    output_fields=["doc_id", "text"],
                )

            for hits in results:
                all_results.append([
                    (h["entity"]["doc_id"], h["distance"], h["entity"].get("text", ""))
                    for h in hits
                ])

        return all_results

    def search_one(
        self,
        name: str,
        vector,
        top_k: int = 10,
        vector_mode: str = "dense",
    ) -> list[tuple[str, float]]:
        vec = np.array(vector, dtype=np.float32) if not isinstance(vector, np.ndarray) else vector
        return self.search(name, [vec], top_k=top_k, vector_mode=vector_mode)[0]

    # ── ColBERT 리랭킹 ───────────────────────────────────────────────────────

    @staticmethod
    def colbert_rerank(
        candidates: list[tuple[str, float, str]],
        query_tok_vecs: np.ndarray,
        doc_tok_vecs: list[np.ndarray],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """MaxSim late interaction 리랭킹.

        Args:
            candidates:      search_with_text() 결과 한 쿼리분
            query_tok_vecs:  shape (num_q_tokens, dim)
            doc_tok_vecs:    candidates 순서와 동일, 각 shape (num_d_tokens, dim)
            top_k:           최종 반환 수
        Returns:
            [(doc_id, colbert_score), ...]  내림차순
        """
        qv = query_tok_vecs / (np.linalg.norm(query_tok_vecs, axis=1, keepdims=True) + 1e-9)
        scored: list[tuple[str, float]] = []
        for (doc_id, _, _), dv_raw in zip(candidates, doc_tok_vecs):
            dv = dv_raw / (np.linalg.norm(dv_raw, axis=1, keepdims=True) + 1e-9)
            sim = qv @ dv.T          # (q_tokens, d_tokens)
            score = float(sim.max(axis=1).sum())
            scored.append((doc_id, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
