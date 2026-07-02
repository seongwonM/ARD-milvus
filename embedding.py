"""Cloud Platform Embedding API 클라이언트.

OpenAI-compatible 형식 기본 지원:
  POST {endpoint}
  Body: {"input": ["text1", "text2"], "model": "..."}
  Response: {"data": [{"index": 0, "embedding": [...]}, ...]}
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import requests

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingClient:

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        })
        # 재시도 통계 (rerank_runner.py가 실행 구간별로 스냅샷 차이를 내서 리포트)
        self.retry_count = 0          # 실패한 시도 총 횟수(성공 전 재시도만, 최종 실패 포함)
        self.retry_success_at: list[int] = []  # 재시도 끝에 성공했을 때 몇 번째 시도였는지(1-base)
        self.failed_time_sec = 0.0    # 실패 시도 자체 + 백오프 sleep에 쓴 시간 누적 (성공 시간과 분리)

    # ── dense ─────────────────────────────────────────────────────────────────

    def _build_payload(self, texts: list[str], **extra) -> dict[str, Any]:
        return {"input": texts, "model": self._config.model, **extra}

    def _parse_response(self, data: dict) -> list[list[float]]:
        return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]

    def _call_api(self, texts: list[str], payload_extra: dict | None = None) -> list[list[float]]:
        payload = self._build_payload(texts, **(payload_extra or {}))
        attempt = 0
        while True:
            t_attempt = time.time()
            try:
                resp = self._session.post(
                    self._config.endpoint, json=payload, timeout=self._config.timeout,
                )
                resp.raise_for_status()
                if attempt > 0:
                    self.retry_success_at.append(attempt + 1)
                return self._parse_response(resp.json())
            except Exception as exc:
                self.retry_count += 1
                self.failed_time_sec += time.time() - t_attempt
                wait = min(2 ** attempt, 30)  # 쿼리를 버릴 수 없으니 성공할 때까지 무한 재시도 (백오프는 30s에서 상한)
                logger.warning(f"API 호출 실패 (attempt {attempt+1}), {wait}s 후 재시도: {exc}")
                time.sleep(wait)
                self.failed_time_sec += wait
                attempt += 1

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """텍스트 → dense 벡터. shape: (n, dim)."""
        bs = batch_size or self._config.batch_size
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), bs):
            all_embeddings.extend(self._call_api(texts[i: i + bs]))
            logger.debug(f"  dense 진행: {min(i+bs, len(texts)):,}/{len(texts):,}")
        return np.array(all_embeddings, dtype=np.float32)

    def detect_dim(self) -> int:
        """실제 임베딩 차원을 API 호출로 확인."""
        return int(self.encode(["dimension probe"]).shape[1])
