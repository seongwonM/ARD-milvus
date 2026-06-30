"""Cloud Platform Embedding API 클라이언트.

OpenAI-compatible 형식 기본 지원:
  POST {endpoint}
  Body: {"input": ["text1", "text2"], "model": "..."}
  Response: {"data": [{"index": 0, "embedding": [...]}, ...]}

Sparse / ColBERT 응답 형식은 API마다 다를 수 있으므로
_parse_sparse_response() / _parse_colbert_response()를 오버라이드하세요.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import requests

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


class SparseUnsupported(Exception):
    """API가 sparse 벡터를 지원하지 않을 때."""


class ColBERTUnsupported(Exception):
    """API가 ColBERT 토큰 벡터를 지원하지 않을 때."""


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

    # ── sparse ────────────────────────────────────────────────────────────────

    def _parse_sparse_response(self, data: dict) -> list[dict[int, float]]:
        """Sparse 응답 파싱. API 형식이 다르면 이 메서드를 오버라이드하세요.

        기본 예상 형식:
            {"data": [{"index": 0, "sparse": {"123": 0.45, ...}}, ...]}
        """
        items = sorted(data["data"], key=lambda x: x["index"])
        return [{int(k): float(v) for k, v in item["sparse"].items()} for item in items]

    def _call_sparse_api(self, texts: list[str]) -> list[dict[int, float]]:
        payload = self._build_payload(texts, encoding_type="sparse")
        resp = self._session.post(self._config.endpoint, json=payload, timeout=self._config.timeout)
        if resp.status_code in (400, 404, 422):
            raise SparseUnsupported(f"HTTP {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        try:
            return self._parse_sparse_response(resp.json())
        except (KeyError, TypeError, ValueError) as exc:
            raise SparseUnsupported(f"응답 형식 불일치: {exc}") from exc

    def encode_sparse(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[dict[int, float]]:
        """텍스트 → sparse 벡터 목록. API 미지원 시 SparseUnsupported 발생."""
        bs = batch_size or self._config.batch_size
        results: list[dict[int, float]] = []
        for i in range(0, len(texts), bs):
            batch = texts[i: i + bs]
            try:
                results.extend(self._call_sparse_api(batch))
            except SparseUnsupported:
                raise
            except Exception as exc:
                raise SparseUnsupported(f"Sparse API 오류: {exc}") from exc
            logger.debug(f"  sparse 진행: {min(i+bs, len(texts)):,}/{len(texts):,}")
        return results

    # ── ColBERT ───────────────────────────────────────────────────────────────

    def _parse_colbert_response(self, data: dict) -> list[np.ndarray]:
        """ColBERT 토큰 벡터 응답 파싱. API 형식이 다르면 오버라이드하세요.

        기본 예상 형식:
            {"data": [{"index": 0, "colbert_vecs": [[...], [...], ...]}, ...]}
        """
        items = sorted(data["data"], key=lambda x: x["index"])
        return [np.array(item["colbert_vecs"], dtype=np.float32) for item in items]

    def _call_colbert_api(self, texts: list[str]) -> list[np.ndarray]:
        payload = self._build_payload(texts, return_colbert_vecs=True)
        resp = self._session.post(self._config.endpoint, json=payload, timeout=self._config.timeout)
        if resp.status_code in (400, 404, 422):
            raise ColBERTUnsupported(f"HTTP {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        try:
            return self._parse_colbert_response(resp.json())
        except (KeyError, TypeError, ValueError) as exc:
            raise ColBERTUnsupported(f"응답 형식 불일치: {exc}") from exc

    def encode_colbert(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[np.ndarray]:
        """텍스트 → 토큰별 벡터 목록 (ColBERT 리랭킹용). 미지원 시 ColBERTUnsupported 발생.

        반환: [np.ndarray shape (num_tokens, dim), ...]  (텍스트 수만큼)
        """
        bs = batch_size or self._config.batch_size
        results: list[np.ndarray] = []
        for i in range(0, len(texts), bs):
            batch = texts[i: i + bs]
            try:
                results.extend(self._call_colbert_api(batch))
            except ColBERTUnsupported:
                raise
            except Exception as exc:
                raise ColBERTUnsupported(f"ColBERT API 오류: {exc}") from exc
            logger.debug(f"  ColBERT 진행: {min(i+bs, len(texts)):,}/{len(texts):,}")
        return results
