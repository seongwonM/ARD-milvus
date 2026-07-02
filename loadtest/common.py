"""Locust 부하테스트에서 공유하는 유틸리티.

- 사전 캐싱된 쿼리 벡터 로드 (cache_query_vectors.py가 생성)
- 전역 커서 — 12,000개 쿼리를 순서대로 순환 순회
- Milvus 연결 생성
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..config import Config
from ..milvus_store import MilvusStore

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"


def load_query_cache(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> tuple[list[str], np.ndarray]:
    """cache_query_vectors.py가 저장한 쿼리 id/벡터를 읽어온다."""
    d = Path(cache_dir)
    ids_path = d / "query_ids.json"
    vecs_path = d / "query_vectors.npy"
    if not ids_path.exists() or not vecs_path.exists():
        raise FileNotFoundError(
            f"쿼리 캐시가 없습니다: {ids_path}, {vecs_path}\n"
            f"먼저 'python -m milvus_migration.loadtest.cache_query_vectors'를 실행하세요."
        )
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    vecs = np.load(vecs_path)
    if len(ids) != len(vecs):
        raise ValueError(f"쿼리 id({len(ids)})와 벡터({len(vecs)}) 개수가 다릅니다.")
    logger.info(f"[쿼리 캐시] {len(ids):,}개 로드 (dim={vecs.shape[1]})")
    return ids, vecs


def load_chunk_cache(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> tuple[list[str], np.ndarray]:
    """cache_chunk_vectors.py가 저장한 corpus id/벡터를 읽어온다."""
    d = Path(cache_dir)
    ids_path = d / "chunk_ids.json"
    vecs_path = d / "chunk_vectors.npy"
    if not ids_path.exists() or not vecs_path.exists():
        raise FileNotFoundError(
            f"corpus 임베딩 캐시가 없습니다: {ids_path}, {vecs_path}\n"
            f"먼저 'python -m milvus_migration.loadtest.cache_chunk_vectors'를 실행하세요."
        )
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    vecs = np.load(vecs_path)
    if len(ids) != len(vecs):
        raise ValueError(f"corpus id({len(ids)})와 벡터({len(vecs)}) 개수가 다릅니다.")
    logger.info(f"[corpus 캐시] {len(ids):,}개 로드 (dim={vecs.shape[1]})")
    return ids, vecs


class QueryCursor:
    """N개의 쿼리 인덱스를 순서대로 순환 순회하는 공유 커서.

    gevent는 협조적 스케줄링이라 I/O가 없는 단순 증가 연산 도중에는 다른
    그린렛으로 컨텍스트 전환이 일어나지 않으므로, 별도 락 없이 여러 greenlet이
    안전하게 공유할 수 있다.
    """

    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError("n은 1 이상이어야 합니다.")
        self._n = n
        self._idx = 0

    def next(self) -> int:
        i = self._idx % self._n
        self._idx += 1
        return i


def build_store(cfg: Config) -> MilvusStore:
    return MilvusStore(cfg.milvus.uri, cfg.milvus.token)
