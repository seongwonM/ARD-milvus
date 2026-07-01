"""[테스트 B: 본 부하테스트] 페이싱은 0으로 고정하고 동시 사용자 수만 단계적으로
늘려서, Milvus+MinIO가 실제로 버틸 수 있는 처리량(TPS) 천장을 찾는다.
(테스트 A와 반대로 이번엔 동시성만 변수 — OFAT 원칙)

설계 포인트
-----------
1. 누적(의도됨): 새 stage에서 유저가 추가돼도 이전 stage 유저는 죽지 않고
   그대로 유지된다 — 실제 트래픽이 점점 늘어나는 상황을 재현하기 위한
   의도된 동작이다.
2. stage 경계 오염 방지: 요청을 "발사(dispatch)한 시점"의 stage 번호를
   이벤트 이름/컨텍스트에 태깅한다. 시스템이 밀리기 시작하면 이전 stage에서
   발사된 요청의 응답이 다음 stage 시간대에 도착할 수 있는데, wall-clock으로만
   구간을 나누면 이런 요청이 엉뚱한 stage로 잘못 집계된다. 발사 시점 태깅은
   이 문제를 피한다.
3. 워밍업 제외: 각 stage 시작 후 WARMUP_SEC 동안의 요청은 report.py 집계에서
   제외한다(캐시/커넥션 풀이 새 동시성 수준에 안정화될 시간을 줌).
4. sentinel: 맨 처음 stage(가장 낮은 동시성)를 맨 끝에 한 번 더 반복한다.
   처음과 끝 값이 비슷하면 테스트 내내 누적된 오염(자원 누수 등)이 없다는
   근거가 되고, 크게 다르면 원인을 조사해야 한다는 신호다.
5. load generator 자체 병목: Locust는 gevent 기반 단일 프로세스라 코어 하나만
   쓴다. 동시성이 매우 높아지면 Milvus가 아니라 Locust 프로세스 자체가 병목이
   될 수 있는데, Locust는 이 경우 로그에 "CPU usage ..." 경고를 남긴다.
   report.py가 --logfile을 스캔해서 이 경고를 찾아준다.

사용법:
    locust -f milvus_migration/loadtest/locustfile_ramp.py --headless \
        --logfile=results/loadtest/ramp.log --html=results/loadtest/ramp.html \
        --csv=results/loadtest/ramp --request-log=results/loadtest/ramp_requests.csv

리포트 생성:
    python -m milvus_migration.loadtest.report ramp \
        --request-log results/loadtest/ramp_requests.csv --logfile results/loadtest/ramp.log
"""
from __future__ import annotations

import csv
import logging
import os
import sys
import time
from pathlib import Path

# Locust는 locustfile을 `python -m`이 아니라 자체 파일로더로 불러오기 때문에
# 상대 임포트(from .. import)가 동작하지 않는다. milvus_migration 패키지의
# 부모 디렉터리(repo 루트)를 직접 sys.path에 넣어 절대 임포트로 해결한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from locust import LoadTestShape, User, constant, events, task

from milvus_migration.config import Config
from milvus_migration.loadtest.common import QueryCursor, build_store, load_query_cache

logger = logging.getLogger(__name__)

# (지속시간_초, 동시_사용자수) — Little's Law로 구한 L0(테스트 A 결과)를 바탕으로
# 조정할 것. 이미지를 다시 빌드하지 않고도 k8s Job의 env로 튜닝할 수 있도록
# LOADTEST_STAGE_USERS(콤마 구분)/LOADTEST_STAGE_DURATION_SEC 환경변수로 덮어쓸 수 있다.
_STAGE_DURATION_SEC = int(os.environ.get("LOADTEST_STAGE_DURATION_SEC", "240"))
_USER_STEPS = [
    int(x) for x in os.environ.get("LOADTEST_STAGE_USERS", "1,2,5,10,20,50,100,200,500").split(",")
    if x.strip()
]
STAGES: list[tuple[int, int]] = [(_STAGE_DURATION_SEC, u) for u in _USER_STEPS]
STAGES.append(STAGES[0])  # sentinel: 첫 stage 조건을 맨 끝에 재현

WARMUP_SEC = 20.0  # report.py가 각 stage 시작 후 이 시간만큼은 집계에서 제외


@events.init_command_line_parser.add_listener
def _add_args(parser) -> None:
    parser.add_argument(
        "--request-log", type=str, default="results/loadtest/ramp_requests.csv",
        help="stage별 원시 요청 로그(CSV) 저장 경로 — report.py가 이 파일을 읽는다.",
    )


_cfg = Config.from_env()
_store = build_store(_cfg)
_query_ids, _query_vecs = load_query_cache()
_cursor = QueryCursor(len(_query_ids))

_stage_state = {"idx": -1, "target_users": 0, "started_at": time.time()}
_request_log: list[dict] = []


@events.request.add_listener
def _log_request(name, response_time, exception, context, **kwargs) -> None:
    _request_log.append({
        "ts": round(time.time(), 3),
        "stage_idx": context.get("stage_idx"),
        "target_users": context.get("target_users"),
        "elapsed_in_stage": context.get("elapsed_in_stage"),
        "name": name,
        "response_time_ms": round(response_time, 2),
        "exception": "" if exception is None else str(exception),
    })


@events.quitting.add_listener
def _flush_request_log(environment, **kwargs) -> None:
    out_path = Path(environment.parsed_options.request_log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ts", "stage_idx", "target_users", "elapsed_in_stage",
            "name", "response_time_ms", "exception",
        ])
        writer.writeheader()
        writer.writerows(_request_log)
    logger.info(f"[요청 로그] {len(_request_log):,}건 저장: {out_path}")


class RampShape(LoadTestShape):
    stages = STAGES

    def tick(self):
        run_time = self.get_run_time()
        elapsed = 0
        for idx, (duration, users) in enumerate(self.stages):
            elapsed += duration
            if run_time < elapsed:
                if _stage_state["idx"] != idx:
                    _stage_state["idx"] = idx
                    _stage_state["target_users"] = users
                    _stage_state["started_at"] = time.time()
                    logger.info(f"[stage {idx}] 동시 사용자 {users}명으로 전환 (run_time={run_time:.0f}s)")
                return (users, users)
        return None


class MilvusLoadUser(User):
    wait_time = constant(0)  # 페이싱 고정(0) — 동시성만 변수로 둔다

    @task
    def search_one(self) -> None:
        stage_idx = _stage_state["idx"]
        target_users = _stage_state["target_users"]
        elapsed_in_stage = round(time.time() - _stage_state["started_at"], 1)

        vec = _query_vecs[_cursor.next()]
        start = time.perf_counter()
        exc = None
        try:
            _store.search_one(_cfg.milvus.collection, vec, top_k=10)
        except Exception as e:
            exc = e
        events.request.fire(
            request_type="milvus",
            name=f"search/stage{stage_idx}_u{target_users}",
            response_time=(time.perf_counter() - start) * 1000,
            response_length=0,
            context={
                "stage_idx": stage_idx,
                "target_users": target_users,
                "elapsed_in_stage": elapsed_in_stage,
            },
            exception=exc,
        )
