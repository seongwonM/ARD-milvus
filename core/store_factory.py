"""벡터 백엔드(Milvus/StarRocks) 선택 — VECTOR_BACKEND 환경변수로 전환.

MilvusStore/StarRocksStore는 동일한 인터페이스를 구현하므로 Pipeline/
retrieval_runner/loadtest 쪽 코드는 어느 백엔드인지 신경 쓸 필요가 없다.
"""
from __future__ import annotations

from typing import Union

from .config import Config
from .milvus_store import MilvusStore
from .starrocks_store import StarRocksStore


def build_store(cfg: Config) -> Union[MilvusStore, StarRocksStore]:
    if cfg.backend == "starrocks":
        return StarRocksStore(
            host=cfg.starrocks.host,
            port=cfg.starrocks.port,
            user=cfg.starrocks.user,
            password=cfg.starrocks.password,
            database=cfg.starrocks.database,
        )
    if cfg.backend == "milvus":
        return MilvusStore(cfg.milvus.uri, cfg.milvus.token)
    raise ValueError(f"알 수 없는 VECTOR_BACKEND: {cfg.backend!r} (milvus|starrocks 중 하나여야 함)")


def collection_name(cfg: Config) -> str:
    """현재 백엔드 기준 컬렉션/테이블 이름."""
    return cfg.starrocks.collection if cfg.backend == "starrocks" else cfg.milvus.collection
