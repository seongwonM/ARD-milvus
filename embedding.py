"""Cloud Platform Embedding API 클라이언트.

OpenAI-compatible 형식 기본 지원:
  POST {endpoint}
  Body: {"input": ["text1", "text2"], "model": "..."}
  Response: {"data": [{"index": 0, "embedding": [...]}, ...]}
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
import requests
from requests.adapters import HTTPAdapter

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
        # 큐/워커 풀로 동시에 여러 요청을 흘려보내므로 기본 pool_maxsize(10)보다 넉넉하게 잡아둠
        # (RERANK_CONCURRENCY 기본값 32와 맞춤 — 낮으면 커넥션이 매번 새로 열려 TCP/TLS
        # 핸드셰이크 시간이 순수 응답시간 측정에 섞여 들어감)
        adapter = HTTPAdapter(pool_maxsize=32)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        # 재시도 통계 (rerank_runner.py가 실행 구간별로 스냅샷 차이를 내서 리포트)
        self.retry_count = 0          # 실패한 시도 총 횟수(성공 전 재시도만, 최종 실패 포함)
        self.retry_success_at: list[int] = []  # 재시도 끝에 성공했을 때 몇 번째 시도였는지(1-base)
        self.failed_time_sec = 0.0    # 실패 시도 자체 + 공유 쿨다운 대기에 쓴 시간 누적 (성공 시간과 분리)
        # 동시 요청(큐/워커 풀) 환경에서 스레드별 순수 응답 시간을 구분해서 재기 위한 스레드-로컬
        # 누적값 — encode()가 batch_size 단위로 여러 번 _call_api를 호출할 수 있어서
        # encode() 시작 시 0으로 리셋하고 그 안에서 성공한 호출들의 시간을 누적한다.
        self._local = threading.local()
        # retry_count/failed_time_sec/_consecutive_failures/_paused_until은 여러 워커 스레드가
        # 동시에 건드릴 수 있어 락으로 보호
        self._retry_lock = threading.Lock()
        # 공유 쿨다운(서킷 브레이커): reranker.py의 RerankClient와 동일한 이유로 도입 —
        # 개별 호출이 각자 자기 실패 횟수만 보고 backoff하면 여러 워커가 같은 타이밍에 몰려서
        # 재시도하는 "thundering herd"가 생김. 실패 한 건이라도 나오면 이 시각(_paused_until)까지
        # 풀 전체가 새 시도를 안 하게 해서 API가 회복할 시간을 준다.
        self._consecutive_failures = 0
        self._paused_until = 0.0  # time.monotonic() 기준

    @property
    def last_latency_sec(self) -> float:
        """직전 이 스레드의 encode() 호출에 걸린 순수 응답 시간 합(재시도/백오프 제외)."""
        return getattr(self._local, "last_latency_sec", 0.0)

    def _wait_if_paused(self) -> float:
        """공유 쿨다운이 걸려 있으면 풀릴 때까지 대기하고, 실제로 대기한 시간(초)을 반환."""
        waited = 0.0
        while True:
            with self._retry_lock:
                remaining = self._paused_until - time.monotonic()
            if remaining <= 0:
                return waited
            time.sleep(remaining)
            waited += remaining

    # ── dense ─────────────────────────────────────────────────────────────────

    def _build_payload(self, texts: list[str], **extra) -> dict[str, Any]:
        return {"input": texts, "model": self._config.model, **extra}

    def _parse_response(self, data: dict) -> list[list[float]]:
        return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]

    def _call_api(self, texts: list[str], payload_extra: dict | None = None) -> list[list[float]]:
        payload = self._build_payload(texts, **(payload_extra or {}))
        attempt = 0
        while True:
            pause_wait = self._wait_if_paused()
            if pause_wait > 0:
                with self._retry_lock:
                    self.failed_time_sec += pause_wait
            t_attempt = time.time()
            try:
                resp = self._session.post(
                    self._config.endpoint, json=payload, timeout=self._config.timeout,
                )
                resp.raise_for_status()
                self._local.last_latency_sec = getattr(self._local, "last_latency_sec", 0.0) + (time.time() - t_attempt)
                if attempt > 0:
                    self.retry_success_at.append(attempt + 1)
                with self._retry_lock:
                    self._consecutive_failures = 0
                return self._parse_response(resp.json())
            except Exception as exc:
                with self._retry_lock:
                    self.retry_count += 1
                    self.failed_time_sec += time.time() - t_attempt
                    self._consecutive_failures += 1
                    wait = min(2 ** self._consecutive_failures, 30)  # 쿼리를 버릴 수 없으니 성공할 때까지 무한 재시도
                    self._paused_until = max(self._paused_until, time.monotonic() + wait)
                logger.warning(
                    f"API 호출 실패 (attempt {attempt+1}, 공유 연속실패 {self._consecutive_failures}회), "
                    f"전체 워커 {wait}s 쿨다운: {exc}"
                )
                attempt += 1

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """텍스트 → dense 벡터. shape: (n, dim)."""
        bs = batch_size or self._config.batch_size
        self._local.last_latency_sec = 0.0
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), bs):
            all_embeddings.extend(self._call_api(texts[i: i + bs]))
            logger.debug(f"  dense 진행: {min(i+bs, len(texts)):,}/{len(texts):,}")
        return np.array(all_embeddings, dtype=np.float32)

    def detect_dim(self) -> int:
        """실제 임베딩 차원을 API 호출로 확인."""
        return int(self.encode(["dimension probe"]).shape[1])
