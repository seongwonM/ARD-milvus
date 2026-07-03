"""[테스트 A: 기준선] 동시 사용자 1명 고정, 요청 간격(N)만 바꿔가며
Milvus 단일 커넥션의 순수 왕복 latency 바닥값(L0)을 측정한다.

동시성과 간격(N)을 동시에 바꾸면 관찰된 변화가 어느 쪽 때문인지 구분할 수
없다(confounding). 그래서 이 테스트는 항상 -u 1 -r 1로만 실행하고, 간격(N)은
--interval 옵션으로 바꿔가며 여러 번 "독립된 프로세스"로 실행한다.
여기서 얻는 L0는 테스트 B(locustfile_ramp.py)의 동시성 단계를 Little's Law로
역산하는 데 사용한다 (Concurrency ≈ Target_TPS × L0).

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
from milvus_migration.loadtest.common import QueryCursor, build_store, load_query_cache, threadpool_search_one
from milvus_migration.store_factory import collection_name

logger = logging.getLogger(__name__)


@events.init_command_line_parser.add_listener
def _add_args(parser) -> None:
    parser.add_argument(
        "--interval", type=float, default=10.0,
        help="쿼리 1개를 보내는 간격 N(초). 이 값을 바꿔가며 여러 번 독립 실행한다.",
    )


_cfg = Config.from_env()
_store = build_store(_cfg)
_collection = collection_name(_cfg)
_query_ids, _query_vecs = load_query_cache()
_cursor = QueryCursor(len(_query_ids))


class MilvusBaselineUser(User):
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
