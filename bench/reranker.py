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
import random
import threading
import time

import requests
from requests.adapters import HTTPAdapter

from ..config import EmbeddingConfig

logger = logging.getLogger(__name__)

_MAX_BACKOFF_EXPONENT = 5   # 2**5=32 > 30 상한이라 이 이상은 무의미 — consecutive_failures가 커져도 큰 수 연산 안 하게 캡
_RESUME_JITTER_SEC = 1.0   # 공유 쿨다운이 끝난 뒤에도 워커마다 이 범위 안에서 랜덤하게 재개 시각을 흩어서 동시 재시도 방지


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
        # 큐/워커 풀로 동시에 여러 요청을 흘려보내므로 기본 pool_maxsize(10)보다 넉넉하게 잡아둠
        # (RERANK_CONCURRENCY 기본값 16과 맞춤 — 낮으면 커넥션이 매번 새로 열려 TCP/TLS
        # 핸드셰이크 시간이 순수 응답시간 측정에 섞여 들어감)
        adapter = HTTPAdapter(pool_maxsize=16)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        # 재시도 통계 (rerank_runner.py가 실행 구간별로 스냅샷 차이를 내서 리포트)
        self.retry_count = 0          # 실패한 시도 총 횟수(성공 전 재시도만, 최종 실패 포함)
        self.retry_success_at: list[int] = []  # 재시도 끝에 성공했을 때 몇 번째 시도였는지(1-base)
        self.failed_time_sec = 0.0    # 실패 시도 자체 + 공유 쿨다운 대기에 쓴 시간 누적 (성공 시간과 분리)
        # 동시 요청(큐/워커 풀) 환경에서 호출 하나하나의 순수 응답 시간을 재려면 전역 카운터로는
        # 어느 스레드의 latency인지 구분이 안 됨 — 스레드-로컬로 "이 스레드가 방금 성공한 호출의
        # 걸린 시간"만 들고 있게 해서 race 없이 호출자가 바로 읽어갈 수 있게 함.
        self._local = threading.local()
        # retry_count/failed_time_sec/_consecutive_failures/_paused_until은 여러 워커 스레드가
        # 동시에 건드릴 수 있어 락으로 보호
        self._retry_lock = threading.Lock()
        # 공유 쿨다운(서킷 브레이커): 개별 호출이 각자 자기 실패 횟수만 보고 backoff하면,
        # 동시에 여러 워커가 같은 타이밍에 몰려서 재시도하는 "thundering herd"가 생김 —
        # 대신 어느 워커든 실패하면 이 시각(_paused_until)까지는 풀 전체가 새 시도를 안 하게 해서
        # API가 회복할 시간을 준다. 실패가 계속되면 이 값도 계속 뒤로 밀림(지수 증가, 상한 30s).
        self._consecutive_failures = 0
        self._paused_until = 0.0  # time.monotonic() 기준

    @property
    def last_latency_sec(self) -> float:
        """직전에 이 스레드에서 성공한 rerank() 호출의 순수 응답 시간(재시도/백오프 제외)."""
        return getattr(self._local, "last_latency_sec", 0.0)

    def _wait_if_paused(self) -> float:
        """공유 쿨다운이 걸려 있으면 풀릴 때까지 대기하고, 실제로 대기한 시간(초)을 반환.

        쿨다운 자체는 모든 워커가 공유하는 시각이라, 그 시각에 그대로 깨어나면 워커 전부가
        똑같은 순간에 다시 API를 때리는 새로운 thundering herd가 생긴다(AWS "Exponential
        Backoff And Jitter" 글에서 경고하는 패턴). 그래서 쿨다운이 끝난 뒤 워커마다 짧은
        랜덤 지터를 한 번 더 둬서 재개 시각 자체를 흩뜨린다.
        """
        waited = 0.0
        while True:
            with self._retry_lock:
                remaining = self._paused_until - time.monotonic()
            if remaining <= 0:
                jitter = random.uniform(0, _RESUME_JITTER_SEC)
                if jitter > 0:
                    time.sleep(jitter)
                    waited += jitter
                return waited
            time.sleep(remaining)
            waited += remaining

    def rerank(
        self, model: str, query: str, documents: list[str], top_n: int,
    ) -> list[tuple[int, float]]:
        """query에 대해 documents를 재정렬.

        반환: [(원본 documents 내 index, relevance_score), ...] (score 내림차순, API 응답 순서 그대로)
        """
        payload = {"model": model, "query": query, "documents": documents, "top_n": top_n}
        attempt = 0
        while True:
            pause_wait = self._wait_if_paused()
            if pause_wait > 0:
                with self._retry_lock:
                    self.failed_time_sec += pause_wait
            t_attempt = time.time()
            try:
                resp = self._session.post(
                    self._endpoint, json=payload, timeout=self._config.timeout,
                )
                resp.raise_for_status()
                results = resp.json()["results"]
                self._local.last_latency_sec = time.time() - t_attempt
                if attempt > 0:
                    self.retry_success_at.append(attempt + 1)
                with self._retry_lock:
                    self._consecutive_failures = 0
                return [(int(item["index"]), float(item["relevance_score"])) for item in results]
            except Exception as exc:
                with self._retry_lock:
                    self.retry_count += 1
                    self.failed_time_sec += time.time() - t_attempt
                    self._consecutive_failures += 1
                    # 쿼리를 버릴 수 없으니 성공할 때까지 무한 재시도. AWS "Full Jitter" 공식
                    # (sleep = random(0, min(cap, base*2^n)))을 그대로 적용 — 지수 자체에도
                    # 상한을 둬서 실패가 오래 이어져도 2**n이 쓸데없이 커지지 않게 함.
                    cap = min(2 ** min(self._consecutive_failures, _MAX_BACKOFF_EXPONENT), 30)
                    wait = random.uniform(0, cap)
                    self._paused_until = max(self._paused_until, time.monotonic() + wait)
                logger.warning(
                    f"rerank API 호출 실패 (attempt {attempt+1}, 공유 연속실패 {self._consecutive_failures}회), "
                    f"전체 워커 {wait:.1f}s 쿨다운: {exc}"
                )
                attempt += 1
