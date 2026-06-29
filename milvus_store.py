"""Milvus VectorStore - dense / sparse 지원.

dense  : FLOAT_VECTOR + HNSW(COSINE)  ← 기본값
sparse : SPARSE_FLOAT_VECTOR + SPARSE_INVERTED_INDEX(IP)
"""
from __future__ import annotations

import logging
import time

_INSERT_BATCH = 64

logger = logging.getLogger(__name__)


class MilvusStore:

    def __init__(self, uri: str, token: str = "") -> None:
        from pymilvus import MilvusClient

        kwargs: dict = {"uri": uri, "timeout": 300}
        if token:
            kwargs["token"] = token
        self._client = MilvusClient(**kwargs)
        self._uri = uri
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

        index_params = MilvusClient.prepare_index_params()

        if vector_mode == "sparse":
            schema.add_field("vector", DataType.SPARSE_FLOAT_VECTOR)
            index_params.add_index(
                field_name="vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )
        else:  # dense
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

    def load_collection(self, name: str) -> None:
        self._client.load_collection(name)
        logger.info(f"  컬렉션 로드: {name}")

    # ── 데이터 삽입 ──────────────────────────────────────────────────────────

    def upload(
        self,
        name: str,
        data_iter,
        vector_mode: str = "dense",
    ) -> int:
        """(id, doc_id, vector) 이터레이터를 받아 Milvus에 삽입.

        dense  : vector = np.ndarray (dim,)
        sparse : vector = dict[int, float]
        """
        batch: list[dict] = []
        total = 0

        for pid, doc_id, vec in data_iter:
            if vector_mode == "sparse":
                row = {"id": pid, "doc_id": doc_id, "vector": {int(k): float(v) for k, v in vec.items()}}
            else:
                row = {"id": pid, "doc_id": doc_id, "vector": vec.tolist()}
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
        """인덱싱 후 컬렉션을 메모리에 로드."""
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
        """벡터 목록으로 ANN 검색.

        반환: [[("doc_id", score), ...], ...]  (쿼리 수 × top_k)
        """
        CHUNK = 256
        all_results: list[list[tuple[str, float]]] = []
        n = len(vectors)

        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)

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
                    output_fields=["doc_id"],
                )
            else:  # dense
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

    def search_one(
        self,
        name: str,
        vector,
        top_k: int = 10,
        vector_mode: str = "dense",
    ) -> list[tuple[str, float]]:
        """단일 쿼리 벡터로 검색. 반환: [("doc_id", score), ...]"""
        import numpy as np
        vec = np.array(vector, dtype=np.float32) if not isinstance(vector, np.ndarray) else vector
        return self.search(name, [vec], top_k=top_k, vector_mode=vector_mode)[0]
