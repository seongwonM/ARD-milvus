"""Milvus VectorStore - dense 임베딩 검색.

컬렉션 스키마:
  id      INT64 (PK)
  doc_id  VARCHAR(512)
  text    VARCHAR(2048)
  vector  FLOAT_VECTOR(dim)
"""
from __future__ import annotations

import logging
import time

import numpy as np

_INSERT_BATCH = 64
_TEXT_MAX_BYTES = 2000  # VARCHAR max_length은 바이트 기준 — 여유분 확보


def _trunc(text: str) -> str:
    b = text.encode("utf-8")
    if len(b) <= _TEXT_MAX_BYTES:
        return text
    return b[:_TEXT_MAX_BYTES].decode("utf-8", errors="ignore")

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

    def create_collection(self, name: str, dim: int) -> None:
        from pymilvus import MilvusClient, DataType

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id",     DataType.INT64,   is_primary=True)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=512)
        schema.add_field("text",   DataType.VARCHAR, max_length=_TEXT_MAX_BYTES)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)

        index_params = MilvusClient.prepare_index_params()
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
        logger.info(f"  컬렉션 생성: {name}  dim={dim}")

    def drop_collection(self, name: str) -> None:
        self._client.drop_collection(name)
        logger.info(f"  컬렉션 삭제: {name}")

    # ── 데이터 삽입 ──────────────────────────────────────────────────────────

    def upload(self, name: str, data_iter) -> int:
        """(id, doc_id, text, vector) 이터레이터를 받아 Milvus에 삽입. vector: np.ndarray (dim,)"""
        batch: list[dict] = []
        total = 0

        for pid, doc_id, text, vec in data_iter:
            batch.append({
                "id": pid,
                "doc_id": doc_id,
                "text": _trunc(text),
                "vector": vec.tolist(),
            })

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

    def search(self, name: str, vectors, top_k: int = 10) -> list[list[tuple[str, float]]]:
        """ANN 검색. 반환: [[("doc_id", score), ...], ...]"""
        # 256 쿼리 × 100 결과 ≈ gRPC 메시지 크기 초과 방지
        CHUNK = 128
        all_results: list[list[tuple[str, float]]] = []

        for start in range(0, len(vectors), CHUNK):
            end = min(start + CHUNK, len(vectors))
            query_data = [vectors[i].tolist() for i in range(start, end)]
            results = self._client.search(
                collection_name=name,
                data=query_data,
                anns_field="vector",
                search_params={"metric_type": "COSINE", "params": {"ef": 100}},
                limit=top_k,
                output_fields=["doc_id"],
            )
            for hits in results:
                all_results.append([(h["entity"]["doc_id"], h["distance"]) for h in hits])

        return all_results

    def search_one(self, name: str, vector, top_k: int = 10) -> list[tuple[str, float]]:
        vec = np.array(vector, dtype=np.float32) if not isinstance(vector, np.ndarray) else vector
        return self.search(name, [vec], top_k=top_k)[0]
