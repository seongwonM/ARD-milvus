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

    # ── dense ─────────────────────────────────────────────────────────────────

    def _build_payload(self, texts: list[str], **extra) -> dict[str, Any]:
        return {"input": texts, "model": self._config.model, **extra}

    def _parse_response(self, data: dict) -> list[list[float]]:
        return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]

    def _call_api(self, texts: list[str], payload_extra: dict | None = None) -> list[list[float]]:
        payload = self._build_payload(texts, **(payload_extra or {}))
        for attempt in range(3):
            try:
                resp = self._session.post(
                    self._config.endpoint, json=payload, timeout=self._config.timeout,
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
