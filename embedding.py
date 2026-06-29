"""Cloud Platform Embedding API 클라이언트.

OpenAI-compatible 형식을 기본으로 지원합니다:
  POST {endpoint}
  Body: {"input": ["text1", "text2"], "model": "..."}
  Response: {"data": [{"index": 0, "embedding": [...]}, ...]}

Cloud Platform 응답 형식이 다를 경우 _parse_response() 메서드를 오버라이드하세요.
"""
from __future__ import annotations

import time
import logging
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

    def _build_payload(self, texts: list[str]) -> dict[str, Any]:
        return {"input": texts, "model": self._config.model}

    def _parse_response(self, data: dict) -> list[list[float]]:
        """OpenAI-compatible 응답 파싱. 다른 형식이면 이 메서드를 수정하세요."""
        return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        payload = self._build_payload(texts)
        for attempt in range(3):
            try:
                resp = self._session.post(
                    self._config.endpoint,
                    json=payload,
                    timeout=self._config.timeout,
                )
                resp.raise_for_status()
                return self._parse_response(resp.json())
            except Exception as exc:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                logger.warning(f"API 호출 실패 (attempt {attempt+1}/3), {wait}s 후 재시도: {exc}")
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """텍스트 목록을 임베딩 벡터로 변환. 반환 shape: (n, dim)."""
        bs = batch_size or self._config.batch_size
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)
            logger.debug(f"  임베딩 진행: {min(i+bs, len(texts)):,}/{len(texts):,}")

        return np.array(all_embeddings, dtype=np.float32)
