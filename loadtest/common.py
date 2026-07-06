"""Locust 부하테스트에서 공유하는 유틸리티.

- 사전 캐싱된 쿼리 벡터 로드 (cache_query_vectors.py가 생성)
- 전역 커서 — 12,000개 쿼리를 순서대로 순환 순회
- Milvus 연결 생성
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import gevent
import gevent.event
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


def _run_loop_forever(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _run_coro_threadsafe(loop: asyncio.AbstractEventLoop, coro):
    """다른(백그라운드) 스레드에서 도는 asyncio 루프에 코루틴을 넘기고, 호출한
    그린렛에서 결과를 기다린다.

    asyncio.run_coroutine_threadsafe(...)가 리턴하는 concurrent.futures.Future의
    .result()를 그린렛에서 직접 부르면 gevent가 patch한 threading.Condition을
    타는데, 이게 간헐적으로 LoopExit을 던진다(gevent 자체에 보고된 알려진 문제 —
    gevent/gevent#1350). 대신 gevent가 스스로의 threadpool에서 쓰는 것과 같은
    방식 — 백그라운드 스레드가 hub.loop.run_callback_threadsafe로 메인 hub에
    안전하게 콜백을 넣고, 그린렛은 gevent 네이티브 AsyncResult.get()으로
    기다린다 — 을 그대로 써서 이 문제를 피한다.
    """
    main_loop = gevent.get_hub().loop
    async_result = gevent.event.AsyncResult()

    def _on_done(fut: "asyncio.Future") -> None:
        try:
            value = fut.result()
        except Exception as e:  # noqa: BLE001 - 원본 예외를 그대로 그린렛 쪽에 전달
            main_loop.run_callback_threadsafe(async_result.set_exception, e)
        else:
            main_loop.run_callback_threadsafe(async_result.set, value)

    future = asyncio.run_coroutine_threadsafe(coro, loop)
    future.add_done_callback(_on_done)
    return async_result.get()


class _AsyncMilvusSearchStore:
    """pymilvus의 AsyncMilvusClient(grpc.aio 기반)를 gevent 프로세스 안에서 쓰기 위한 래퍼.

    동기 MilvusClient는 grpc의 동기(blocking) C-core 경로를 타는데, 이건 Locust가
    걸어둔 gevent monkey-patch와 근본적으로 안 맞아서 생성 시점부터 hang한다
    (2026-07 실측 — threadpool로 감싸도 소용없었음: MilvusClient 생성자 자체가
    내부에서 만드는 백그라운드 소켓/스레드가 이미 monkey-patch된 상태라 이걸
    처리해줄 이벤트 루프가 없는 채로 멈춰버림). grpc.experimental.gevent.init_gevent()도
    시도했으나 2019~2020년대 grpcio(1.1x~1.2x)에서만 검증된 기능이라 그 이후 grpc
    내부 스레딩 모델 변화를 못 따라가 최신 grpcio에서는 오히려 RuntimeError
    (greenlet is being finalized)로 깨짐 — Python 3.12 wheel도 없는 옛날 grpcio라
    다운그레이드도 불가능.

    대신 AsyncMilvusClient(grpc.aio, 진짜 asyncio)는 이 충돌 자체가 없다(실측 확인됨).
    다만 asyncio 이벤트 루프를 매 호출마다 새로 만들면(asyncio.run) 매번 채널을
    재생성하는 꼴이라 baseline 테스트가 재려는 순수 latency(L0)에 연결 비용이
    섞인다. 그래서 이벤트 루프 하나를 gevent 실제 threadpool(진짜 OS 스레드)에서
    프로세스 생애주기 내내 돌리고(run_forever), 매 검색 요청은 _run_coro_threadsafe로
    그 루프에 넘긴다.
    """

    def __init__(self, cfg: Config) -> None:
        from pymilvus import AsyncMilvusClient

        self._loop = asyncio.new_event_loop()
        gevent.get_hub().threadpool.spawn(_run_loop_forever, self._loop)

        kwargs: dict = {"uri": cfg.milvus.uri, "timeout": 300}
        if cfg.milvus.token:
            kwargs["token"] = cfg.milvus.token

        async def _make_client():
            return AsyncMilvusClient(**kwargs)

        self._client = _run_coro_threadsafe(self._loop, _make_client())

    def search_one(self, collection: str, vector, top_k: int):
        vec = np.asarray(vector, dtype=np.float32)

        async def _do():
            results = await self._client.search(
                collection_name=collection,
                data=[vec.tolist()],
                anns_field="vector",
                search_params={"metric_type": "COSINE", "params": {"ef": 100}},
                limit=top_k,
                output_fields=["doc_id"],
            )
            return [(h["entity"]["doc_id"], h["distance"]) for h in results[0]]

        return _run_coro_threadsafe(self._loop, _do())


def build_store(cfg: Config):
    """VECTOR_BACKEND(milvus|starrocks)에 따라 알맞은 store를 만든다.

    milvus는 _AsyncMilvusSearchStore(grpc.aio 기반, 위 docstring 참고)를 쓴다.
    starrocks는 pymysql 기반이라 gevent monkey-patch와 충돌하는 지점이 없어
    기존처럼 store_factory.build_store를 gevent threadpool(진짜 OS 스레드)에서
    그대로 호출한다.
    """
    _debug(f"build_store 시작 (backend={cfg.backend})")
    if cfg.backend == "milvus":
        result = _AsyncMilvusSearchStore(cfg)
    else:
        result = gevent.get_hub().threadpool.apply(_build_store, (cfg,))
    _debug("build_store 완료")
    return result


_first_search_logged = False


def threadpool_search_one(store, collection: str, vector, top_k: int):
    """store.search_one을 실행한다.

    _AsyncMilvusSearchStore는 내부적으로 이미 전용 백그라운드 스레드(asyncio 루프)로
    작업을 넘기므로 여기서 다시 threadpool로 감싸지 않는다(감싸면 그 real thread
    안에서 concurrent.futures.Future.result()를 기다리게 되는데, 그 스레드엔 이걸
    처리할 gevent hub가 없어 또 hang한다 — 그래서 이 대기는 반드시 hub가 살아있는
    메인 그린렛에서 일어나야 한다). starrocks(pymysql, 동기 blocking)는 여전히
    threadpool로 감싸서 이벤트 루프가 멈추지 않게 한다.
    """
    global _first_search_logged
    first = not _first_search_logged
    if first:
        _debug("첫 search_one 시작")
    if isinstance(store, _AsyncMilvusSearchStore):
        result = store.search_one(collection, vector, top_k)
    else:
        result = gevent.get_hub().threadpool.apply(store.search_one, (collection, vector), {"top_k": top_k})
    if first:
        _debug("첫 search_one 완료")
        _first_search_logged = True
    return result
