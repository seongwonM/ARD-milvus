"""Cloud Platform Rerank API 클라이언트.

embedding API와 헤더/인증은 같지만 경로가 다릅니다 (예: .../embeddings 대신 .../rerank).
RERANK_API_ENDPOINT 환경변수로 명시하거나, 미설정 시 EMBEDDING_API_ENDPOINT의 마지막
경로 세그먼트만 "rerank"로 바꿔서 자동 유도합니다.

요청 형식:
    POST {rerank_endpoint}
    Headers: Authorization: Bearer {api_key}
    Body: {"model": "...", "query": "...", "documents": ["...", ...], "top_n": N}

응답 형식:
    {"results": [{"index": 0, "document": {"text": "..."}, "relevance_score": 0.93}, ...]}
"""
from __future__ import annotations

import logging
import time

import requests

from ..config import EmbeddingConfig

logger = logging.getLogger(__name__)


def _derive_rerank_endpoint(embedding_endpoint: str) -> str:
    base, _, _last = embedding_endpoint.rpartition("/")
    return f"{base}/rerank" if base else embedding_endpoint


class RerankClient:

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._endpoint = config.rerank_endpoint or _derive_rerank_endpoint(config.endpoint)
        logger.info(f"[Rerank] endpoint: {self._endpoint}")
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        })

    def rerank(
        self, model: str, query: str, documents: list[str], top_n: int,
    ) -> list[tuple[int, float]]:
        """query에 대해 documents를 재정렬.

        반환: [(원본 documents 내 index, relevance_score), ...] (score 내림차순, API 응답 순서 그대로)
        """
        payload = {"model": model, "query": query, "documents": documents, "top_n": top_n}
        for attempt in range(3):
            try:
                resp = self._session.post(
                    self._endpoint, json=payload, timeout=self._config.timeout,
                )
                resp.raise_for_status()
                results = resp.json()["results"]
                return [(int(item["index"]), float(item["relevance_score"])) for item in results]
            except Exception as exc:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                logger.warning(f"rerank API 호출 실패 (attempt {attempt+1}/3), {wait}s 후 재시도: {exc}")
                time.sleep(wait)
        raise RuntimeError("unreachable")
