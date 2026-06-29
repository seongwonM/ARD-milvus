"""환경변수 기반 설정 관리."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class MilvusConfig:
    uri: str         = field(default_factory=lambda: os.environ.get("MILVUS_URI", "http://localhost:19530"))
    token: str       = field(default_factory=lambda: os.environ.get("MILVUS_TOKEN", ""))
    collection: str  = field(default_factory=lambda: os.environ.get("MILVUS_COLLECTION", "documents"))


@dataclass
class EmbeddingConfig:
    # Cloud Platform API 엔드포인트 (예: https://api.company.internal/v1/embeddings)
    endpoint: str    = field(default_factory=lambda: os.environ["EMBEDDING_API_ENDPOINT"])
    api_key: str     = field(default_factory=lambda: os.environ.get("EMBEDDING_API_KEY", ""))
    model: str       = field(default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))
    dim: int         = field(default_factory=lambda: int(os.environ.get("EMBEDDING_DIM", "1536")))
    batch_size: int  = field(default_factory=lambda: int(os.environ.get("EMBEDDING_BATCH_SIZE", "64")))
    timeout: float   = field(default_factory=lambda: float(os.environ.get("EMBEDDING_TIMEOUT", "60")))


@dataclass
class Config:
    milvus: MilvusConfig       = field(default_factory=MilvusConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            milvus=MilvusConfig(),
            embedding=EmbeddingConfig(),
        )
