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
        # 재시도 통계 (rerank_runner.py가 실행 구간별로 스냅샷 차이를 내서 리포트)
        self.retry_count = 0          # 실패한 시도 총 횟수(성공 전 재시도만, 최종 실패 포함)
        self.retry_success_at: list[int] = []  # 재시도 끝에 성공했을 때 몇 번째 시도였는지(1-base)
        self.failed_time_sec = 0.0    # 실패 시도 자체 + 백오프 sleep에 쓴 시간 누적 (성공 시간과 분리)

    def rerank(
        self, model: str, query: str, documents: list[str], top_n: int,
    ) -> list[tuple[int, float]]:
        """query에 대해 documents를 재정렬.

        반환: [(원본 documents 내 index, relevance_score), ...] (score 내림차순, API 응답 순서 그대로)
        """
        payload = {"model": model, "query": query, "documents": documents, "top_n": top_n}
        attempt = 0
        while True:
            t_attempt = time.time()
            try:
                resp = self._session.post(
                    self._endpoint, json=payload, timeout=self._config.timeout,
                )
                resp.raise_for_status()
                results = resp.json()["results"]
                if attempt > 0:
                    self.retry_success_at.append(attempt + 1)
                return [(int(item["index"]), float(item["relevance_score"])) for item in results]
            except Exception as exc:
                self.retry_count += 1
                self.failed_time_sec += time.time() - t_attempt
                wait = min(2 ** attempt, 30)  # 쿼리를 버릴 수 없으니 성공할 때까지 무한 재시도 (백오프는 30s에서 상한)
                logger.warning(f"rerank API 호출 실패 (attempt {attempt+1}), {wait}s 후 재시도: {exc}")
                time.sleep(wait)
                self.failed_time_sec += wait
                attempt += 1
