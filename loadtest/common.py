"""Locust 부하테스트에서 공유하는 유틸리티.

- 사전 캐싱된 쿼리 벡터 로드 (cache_query_vectors.py가 생성)
- 전역 커서 — 12,000개 쿼리를 순서대로 순환 순회
- Milvus 연결 생성
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import gevent
import numpy as np

from ..config import Config
from ..store_factory import build_store as _build_store

logger = logging.getLogger(__name__)


def _debug(msg: str) -> None:
    """locust는 --logfile로 logging 출력을 파일로만 보내서 kubectl logs에 안 보인다.
    print()는 logging 핸들러를 안 거치고 stdout에 바로 쓰이므로 --logfile과
    무관하게 항상 보인다 — 어느 단계에서 멈추는지 확인하려고 넣은 체크포인트.
    """
    print(f"[DEBUG {time.strftime('%H:%M:%S')}] {msg}", file=sys.stdout, flush=True)

# k8s Job에서는 LOADTEST_CACHE_DIR=/results/cache로 지정해 PVC(loadtest-results)에
# 저장한다 — 그래야 Job을 지우고 값 바꿔서 다시 apply해도(파드가 새로 뜨어도) 캐시가
# 남아있어 재임베딩하지 않는다. 로컬에서 직접 실행할 땐 기본값(코드 옆 cache/)을 쓴다.
DEFAULT_CACHE_DIR = Path(os.environ.get("LOADTEST_CACHE_DIR", str(Path(__file__).parent / "cache")))


def load_query_cache(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> tuple[list[str], np.ndarray]:
    """cache_query_vectors.py가 저장한 쿼리 id/벡터를 읽어온다."""
    _debug(f"쿼리 캐시 로드 시작: {cache_dir}")
    d = Path(cache_dir)
    ids_path = d / "query_ids.json"
    vecs_path = d / "query_vectors.npy"
    if not ids_path.exists() or not vecs_path.exists():
        raise FileNotFoundError(
            f"쿼리 캐시가 없습니다: {ids_path}, {vecs_path}\n"
            f"먼저 'python -m milvus_migration.loadtest.cache_query_vectors'를 실행하세요."
        )
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    vecs = np.load(vecs_path)
    if len(ids) != len(vecs):
        raise ValueError(f"쿼리 id({len(ids)})와 벡터({len(vecs)}) 개수가 다릅니다.")
    logger.info(f"[쿼리 캐시] {len(ids):,}개 로드 (dim={vecs.shape[1]})")
    _debug(f"쿼리 캐시 로드 완료: {len(ids):,}개")
    return ids, vecs


def load_chunk_cache(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> tuple[list[str], np.ndarray]:
    """cache_chunk_vectors.py가 저장한 corpus id/벡터를 읽어온다."""
    d = Path(cache_dir)
    ids_path = d / "chunk_ids.json"
    vecs_path = d / "chunk_vectors.npy"
    if not ids_path.exists() or not vecs_path.exists():
        raise FileNotFoundError(
            f"corpus 임베딩 캐시가 없습니다: {ids_path}, {vecs_path}\n"
            f"먼저 'python -m milvus_migration.loadtest.cache_chunk_vectors'를 실행하세요."
        )
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    vecs = np.load(vecs_path)
    if len(ids) != len(vecs):
        raise ValueError(f"corpus id({len(ids)})와 벡터({len(vecs)}) 개수가 다릅니다.")
    logger.info(f"[corpus 캐시] {len(ids):,}개 로드 (dim={vecs.shape[1]})")
    return ids, vecs


class QueryCursor:
    """N개의 쿼리 인덱스를 순서대로 순환 순회하는 공유 커서.

    gevent는 협조적 스케줄링이라 I/O가 없는 단순 증가 연산 도중에는 다른
    그린렛으로 컨텍스트 전환이 일어나지 않으므로, 별도 락 없이 여러 greenlet이
    안전하게 공유할 수 있다.
    """

    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError("n은 1 이상이어야 합니다.")
        self._n = n
        self._idx = 0

    def next(self) -> int:
        i = self._idx % self._n
        self._idx += 1
        return i


def build_store(cfg: Config):
    """VECTOR_BACKEND(milvus|starrocks)에 따라 알맞은 store를 만든다 (store_factory.py).

    gevent 내장 threadpool(진짜 OS 스레드)에서 생성한다 — MilvusClient 생성자가
    내부적으로 서버와 동기 gRPC handshake를 할 수 있는데, 이걸 그린렛에서 직접
    호출하면 threadpool_search_one과 똑같은 이유로 이벤트 루프 전체가 멈출 수
    있다(연결 시점이라 search_one을 아무리 잘 감싸도 소용없음). 그래서 store
    생성 자체도 항상 threadpool을 거친다.
    """
    _debug(f"build_store 시작 (backend={cfg.backend})")
    result = gevent.get_hub().threadpool.apply(_build_store, (cfg,))
    _debug("build_store 완료")
    return result


_first_search_logged = False


def threadpool_search_one(store, collection: str, vector, top_k: int):
    """store.search_one을 gevent 내장 threadpool(진짜 OS 스레드)에서 실행한다.

    pymilvus는 동기 gRPC(grpcio C-core)를 쓰는데, 이는 gevent monkey-patch와
    협조하지 않는다 — 그린렛에서 직접 호출하면 이 블로킹 호출이 프로세스의
    유일한 이벤트 루프 자체를 통째로 멈춰서(타임아웃도 안 걸리고 무한 hang),
    Locust 프로세스가 아예 응답 없이 멈춰버린다. 실제 OS 스레드로 넘기면
    호출한 그린렛만 대기하고 이벤트 루프(run-time 타이머 등)는 계속 돈다.
    """
    global _first_search_logged
    first = not _first_search_logged
    if first:
        _debug("첫 search_one 시작")
    result = gevent.get_hub().threadpool.apply(store.search_one, (collection, vector), {"top_k": top_k})
    if first:
        _debug("첫 search_one 완료")
        _first_search_logged = True
    return result
