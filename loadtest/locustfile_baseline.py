"""[테스트 A: 기준선] 동시 사용자 1명 고정, 요청 간격(N)만 바꿔가며
Milvus 단일 커넥션의 순수 왕복 latency 바닥값(L0)을 측정한다.

동시성과 간격(N)을 동시에 바꾸면 관찰된 변화가 어느 쪽 때문인지 구분할 수
없다(confounding). 그래서 이 테스트는 항상 -u 1 -r 1로만 실행하고, 간격(N)은
--interval 옵션으로 바꿔가며 여러 번 "독립된 프로세스"로 실행한다.
여기서 얻는 L0는 테스트 B(locustfile_ramp.py)의 동시성 단계를 Little's Law로
역산하는 데 사용한다 (Concurrency ≈ Target_TPS × L0).

milvus 백엔드는 Locust 공식 locust.contrib.milvus.MilvusUser를 그대로 쓴다 —
동기 MilvusClient를 직접 gevent 프로세스 안에서 쓰려던 시도들(threadpool로
감싸기, grpc.experimental.gevent.init_gevent() 수동 호출, AsyncMilvusClient +
asyncio 브릿지)이 전부 다른 방식으로 hang/RuntimeError/LoopExit이 났었는데
(2026-07 실측), MilvusUser는 Locust 프로젝트 자신이 CI로 검증하며 유지보수하는
구현이라 이 gevent+pymilvus 호환 문제를 이미 해결해뒀다. starrocks는 pymysql
기반이라 애초에 이 문제가 없어 기존 방식(store_factory + threadpool)을 그대로 쓴다.

사용법 (N값을 바꿔가며 반복 실행 — 각각 별도 프로세스):
    locust -f milvus_migration/loadtest/locustfile_baseline.py --headless \
        -u 1 -r 1 --run-time 3m --interval 10 \
        --csv=results/loadtest/baseline_N10 --logfile=results/loadtest/baseline_N10.log

    locust -f milvus_migration/loadtest/locustfile_baseline.py --headless \
        -u 1 -r 1 --run-time 3m --interval 5 \
        --csv=results/loadtest/baseline_N5 --logfile=results/loadtest/baseline_N5.log

    (이후 N=2, 1, 0.5, 0.2 ... 순서로 반복. 컬렉션은 seed_collection.py로,
     쿼리 벡터 캐시는 cache_query_vectors.py로 미리 준비되어 있어야 한다.)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Locust는 locustfile을 `python -m`이 아니라 자체 파일로더로 불러오기 때문에
# 상대 임포트(from .. import)가 동작하지 않는다. milvus_migration 패키지의
# 부모 디렉터리(repo 루트)를 직접 sys.path에 넣어 절대 임포트로 해결한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from locust import User, events, task

from milvus_migration.config import Config
from milvus_migration.loadtest.common import QueryCursor, _debug, build_store, load_query_cache, threadpool_search_one
from milvus_migration.store_factory import collection_name

logger = logging.getLogger(__name__)


@events.init_command_line_parser.add_listener
def _add_args(parser) -> None:
    parser.add_argument(
        "--interval", type=float, default=10.0,
        help="쿼리 1개를 보내는 간격 N(초). 이 값을 바꿔가며 여러 번 독립 실행한다.",
    )


_debug("locustfile_baseline 모듈 로딩 시작")
_cfg = Config.from_env()
_collection = collection_name(_cfg)
_query_ids, _query_vecs = load_query_cache()
_cursor = QueryCursor(len(_query_ids))
_debug("locustfile_baseline 모듈 로딩 완료")


if _cfg.backend == "milvus":
    from locust.contrib.milvus import MilvusUser

    class LoadUser(MilvusUser):
        """반드시 -u 1 -r 1로 실행 (동시 사용자 1명 고정) — 순차 순회 + 고정 간격."""

        def __init__(self, environment) -> None:
            super().__init__(
                environment,
                uri=_cfg.milvus.uri,
                token=_cfg.milvus.token,
                collection_name=_collection,
                timeout=300,
            )

        def wait_time(self) -> float:
            return self.environment.parsed_options.interval

        @task
        def search_one(self) -> None:
            vec = _query_vecs[_cursor.next()]
            self.search(
                data=[vec.tolist()],
                anns_field="vector",
                limit=10,
                search_params={"metric_type": "COSINE", "params": {"ef": 100}},
                output_fields=["doc_id"],
            )

else:
    _store = build_store(_cfg)

    class LoadUser(User):
        """반드시 -u 1 -r 1로 실행 (동시 사용자 1명 고정) — 순차 순회 + 고정 간격."""

        def wait_time(self) -> float:
            return self.environment.parsed_options.interval

        @task
        def search_one(self) -> None:
            vec = _query_vecs[_cursor.next()]
            start = time.perf_counter()
            exc = None
            try:
                threadpool_search_one(_store, _collection, vec, top_k=10)
            except Exception as e:
                exc = e
            events.request.fire(
                request_type=_cfg.backend,
                name="search",
                response_time=(time.perf_counter() - start) * 1000,
                response_length=0,
                context={},
                exception=exc,
            )
