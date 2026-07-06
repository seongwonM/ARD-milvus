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

# StarRocksStore와 공유(백엔드 비교 벤치마크의 공정성을 위해 값을 하나로 통일) —
# 기존 코드가 `from .milvus_store import _trunc, _TEXT_MAX_BYTES, _SEARCH_CHUNK`로
# 가져다 쓰던 것도 그대로 동작하도록 여기서 재노출한다.
from .vector_store_common import _SEARCH_CHUNK, _TEXT_MAX_BYTES, _trunc

_INSERT_BATCH = 64

# bench/retrieval_runner.py가 청크를 얼마나 모아서 한 번에 _insert_with_retry로
# 넘길지 계산할 때 쓰는 예산 — gRPC 기본 메시지 크기 제한(64MB)보다 한참
# 여유있게(고차원 모델에서 batch가 너무 크면 "received message larger than max" 에러가 남).
_INSERT_BUDGET_BYTES = 20_000_000

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

    def insert_budget_bytes(self) -> int:
        """bench/retrieval_runner.py가 인덱싱 중 한 번의 _insert_with_retry 호출에
        몇 바이트어치 청크를 모아 넘길지 결정할 때 쓰는 예산."""
        return _INSERT_BUDGET_BYTES

    def estimate_row_bytes(self, doc_id: str, text: str, vector) -> int:
        """gRPC(protobuf 바이너리)로 보내는 한 행의 대략적인 바이트 수.

        float32 벡터는 4바이트/개로 고정이라(StarRocksStore처럼 SQL 텍스트 리터럴로
        바뀌며 크기가 들쭉날쭉해지는 문제가 없음) 추정이 곧 실제와 거의 같다.
        """
        return len(vector) * 4 + len(text.encode("utf-8")) + len(doc_id.encode("utf-8")) + 200

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
        # flush 없이 바로 load_collection()을 부르면, 아직 세그먼트가 seal되지
        # 않은 최근 insert 데이터가 통계(get_collection_stats)에도 검색 대상
        # (load된 세그먼트)에도 빠질 수 있다 — flush로 먼저 전부 persist시킨다.
        self._client.flush(name)
        self._client.load_collection(name)
        logger.info(f"  인덱스 로드 완료: {name}")

    # ── 검색 ─────────────────────────────────────────────────────────────────

    def search(self, name: str, vectors, top_k: int = 10) -> list[list[tuple[str, float]]]:
        """ANN 검색. 반환: [[("doc_id", score), ...], ...]"""
        all_results: list[list[tuple[str, float]]] = []

        for start in range(0, len(vectors), _SEARCH_CHUNK):
            end = min(start + _SEARCH_CHUNK, len(vectors))
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
