"""환경변수 기반 설정 관리."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class MilvusConfig:
    uri: str        = field(default_factory=lambda: os.environ.get("MILVUS_URI", "http://localhost:19530"))
    token: str      = field(default_factory=lambda: os.environ.get("MILVUS_TOKEN", ""))
    collection: str = field(default_factory=lambda: os.environ.get("MILVUS_COLLECTION", "documents"))


@dataclass
class StarRocksConfig:
    """StarRocks는 MySQL 프로토콜(기본 포트 9030)로 접속 — pymysql 사용."""
    host: str       = field(default_factory=lambda: os.environ.get("STARROCKS_HOST", "localhost"))
    port: int       = field(default_factory=lambda: int(os.environ.get("STARROCKS_PORT", "9030").split(":")[-1]))
    user: str       = field(default_factory=lambda: os.environ.get("STARROCKS_USER", "root"))
    password: str   = field(default_factory=lambda: os.environ.get("STARROCKS_PASSWORD", ""))
    database: str   = field(default_factory=lambda: os.environ.get("STARROCKS_DATABASE", "milvus_migration"))
    # MilvusConfig.collection과 동일한 역할(테이블/컬렉션 이름) — 백엔드 전환 시 이 값만 다르면 됨.
    collection: str = field(default_factory=lambda: os.environ.get("MILVUS_COLLECTION", "documents"))


@dataclass
class EmbeddingConfig:
    endpoint: str   = field(default_factory=lambda: os.environ["EMBEDDING_API_ENDPOINT"])
    # rerank API는 embedding과 헤더/인증은 같지만 경로가 다름 (.../embeddings 대신 .../rerank).
    # 미설정 시 endpoint의 마지막 경로만 "rerank"로 바꿔서 자동 유도.
    rerank_endpoint: str = field(default_factory=lambda: os.environ.get("RERANK_API_ENDPOINT", ""))
    api_key: str    = field(default_factory=lambda: os.environ.get("EMBEDDING_API_KEY", ""))
    model: str      = field(default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))
    # 8B 모델 (EMBEDDING_MODEL_8B 미설정 시 8B 조합 스킵)
    model_8b: str   = field(default_factory=lambda: os.environ.get("EMBEDDING_MODEL_8B", ""))
    batch_size: int = field(default_factory=lambda: int(os.environ.get("EMBEDDING_BATCH_SIZE", "64")))
    timeout: float  = field(default_factory=lambda: float(os.environ.get("EMBEDDING_TIMEOUT", "60")))


@dataclass
class Config:
    milvus: MilvusConfig         = field(default_factory=MilvusConfig)
    starrocks: StarRocksConfig   = field(default_factory=StarRocksConfig)
    embedding: EmbeddingConfig   = field(default_factory=EmbeddingConfig)
    # 벡터 백엔드 선택 — "milvus"(기본) | "starrocks". store_factory.build_store()가 이 값을 본다.
    backend: str = field(default_factory=lambda: os.environ.get("VECTOR_BACKEND", "milvus"))

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            milvus=MilvusConfig(),
            starrocks=StarRocksConfig(),
            embedding=EmbeddingConfig(),
        )
