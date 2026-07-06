"""StarRocks VectorStore - dense 임베딩 검색 (Vector Index / HNSW).

MilvusStore와 동일한 인터페이스(has_collection/collection_size/create_collection/
drop_collection/upload/_insert_with_retry/finalize/search/search_one)를 제공해서
Pipeline/retrieval_runner/loadtest 쪽 코드가 백엔드 차이를 신경 쓰지 않게 한다
(store_factory.build_store()가 VECTOR_BACKEND 환경변수로 둘 중 하나를 고른다).

테이블 스키마:
  id      BIGINT
  doc_id  VARCHAR(512)
  text    VARCHAR(2048)
  vector  ARRAY<FLOAT> + VECTOR INDEX(HNSW, cosine_similarity)

주의(StarRocks 문서 기준, 2026-07 확인):
- 벡터 인덱스는 v3.4+ shared-nothing 클러스터에서만 지원 (starrocks/allin1-ubuntu:3.5.x 이상 사용).
- FE에 enable_experimental_vector=true가 켜져 있어야 함 — 연결 시 자동으로 켠다(멱등).
- ANN 검색은 Milvus처럼 여러 쿼리 벡터를 한 번에 묶어 보낼 수 없고, 쿼리 벡터 1개당
  SQL 1건이 필요하다 — search()는 벡터별로 순차 쿼리한다 (배치 API가 없는 게 아니라
  StarRocks ANN 문법 자체가 상수 벡터 1개 기준이라 구조적으로 그렇다).
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .vector_store_common import _TEXT_MAX_BYTES, _trunc

logger = logging.getLogger(__name__)


def _vec_literal(vec) -> str:
    """ARRAY<FLOAT> SQL 리터럴로 변환: [0.1,0.2,...] (숫자만 들어가므로 injection 위험 없음)."""
    if isinstance(vec, np.ndarray):
        vec = vec.tolist()
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class StarRocksStore:

    def __init__(
        self,
        host: str,
        port: int = 9030,
        user: str = "root",
        password: str = "",
        database: str = "milvus_migration",
    ) -> None:
        import pymysql

        self._database = database
        self._conn = pymysql.connect(
            host=host, port=port, user=user, password=password,
            autocommit=True, connect_timeout=300, read_timeout=300, write_timeout=300,
        )
        # allin1 이미지는 쿼리 포트(9030)가 열려도 FE가 DDL을 처리할 준비(리더 선출 등)가
        # 조금 더 걸릴 수 있어서, 최초 연결 직후의 DDL만 짧게 재시도한다.
        self._exec_with_retry(f"CREATE DATABASE IF NOT EXISTS `{database}`")
        self._exec_with_retry(f"USE `{database}`")
        try:
            self._exec('ADMIN SET FRONTEND CONFIG ("enable_experimental_vector" = "true")')
        except Exception as exc:
            logger.warning(f"[StarRocks] enable_experimental_vector 설정 실패(이미 켜져 있으면 무해): {exc}")

        # insert_budget_bytes()가 실제 클러스터 설정을 기준으로 예산을 잡을 수 있도록
        # max_allowed_packet(MySQL 프로토콜 — 세션/서버당 다르게 설정될 수 있음)을 직접
        # 조회한다. 하드코딩된 값을 가정하지 않기 위함(2026-07 실측: 기본값 32MB로 가정한
        # 계산도 실제로는 틀릴 수 있어 안전하게 클러스터에 직접 물어봄).
        try:
            rows = self._fetchall("SHOW VARIABLES LIKE 'max_allowed_packet'")
            self._max_packet_bytes = int(rows[0][1])
        except Exception as exc:
            self._max_packet_bytes = 32 * 1024 * 1024  # 조회 실패 시 StarRocks 기본값으로 폴백
            logger.warning(f"[StarRocks] max_allowed_packet 조회 실패, 기본값(32MB) 사용: {exc}")

        logger.info(f"[StarRocks] 연결: {host}:{port}/{database} (max_allowed_packet={self._max_packet_bytes:,}B)")

    def _exec(self, sql: str):
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return cur

    def _exec_with_retry(self, sql: str, attempts: int = 12, wait_sec: float = 5.0):
        for attempt in range(attempts):
            try:
                return self._exec(sql)
            except Exception as exc:
                if attempt == attempts - 1:
                    raise
                logger.warning(f"[StarRocks] 초기화 쿼리 실패(FE 초기화 중일 수 있음), {wait_sec}s 후 재시도: {exc}")
                time.sleep(wait_sec)

    def _fetchall(self, sql: str):
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()

    # ── 컬렉션(테이블) 관리 ──────────────────────────────────────────────────

    def has_collection(self, name: str) -> bool:
        rows = self._fetchall(
            f"SELECT COUNT(*) FROM information_schema.tables "
            f"WHERE table_schema='{self._database}' AND table_name='{name}'"
        )
        return rows[0][0] > 0

    def collection_size(self, name: str) -> int:
        rows = self._fetchall(f"SELECT COUNT(*) FROM `{name}`")
        return int(rows[0][0])

    def create_collection(self, name: str, dim: int) -> None:
        self._exec(f"""
            CREATE TABLE `{name}` (
                `id`     BIGINT       NOT NULL,
                `doc_id` VARCHAR(512) NOT NULL,
                `text`   VARCHAR({_TEXT_MAX_BYTES}),
                `vector` ARRAY<FLOAT> NOT NULL,
                INDEX vec_idx (vector) USING VECTOR (
                    "index_type" = "hnsw",
                    "dim" = "{dim}",
                    "metric_type" = "cosine_similarity",
                    "is_vector_normed" = "false",
                    "M" = "16",
                    "efconstruction" = "100"
                )
            ) ENGINE=OLAP
            DUPLICATE KEY(id)
            DISTRIBUTED BY HASH(id) BUCKETS 4
            PROPERTIES ("replication_num" = "1")
        """)
        logger.info(f"  테이블 생성: {name}  dim={dim}")

    def drop_collection(self, name: str) -> None:
        self._exec(f"DROP TABLE IF EXISTS `{name}`")
        logger.info(f"  테이블 삭제: {name}")

    # ── 데이터 삽입 ──────────────────────────────────────────────────────────

    def insert_budget_bytes(self) -> int:
        """호출자(bench/retrieval_runner.py, upload())가 한 번의 INSERT SQL에
        몇 바이트어치를 모아 보낼지 결정할 때 쓰는 예산.

        실제 조회한 max_allowed_packet의 절반만 쓴다(다른 세션 동시 사용,
        헤더 오버헤드 감안한 여유분). 행 하나의 바이트 수는 추정하지 않고
        estimate_row_bytes()가 실제 SQL 조각을 만들어 정확히 잰다 — float repr
        길이나 escape 오버헤드를 추정치로 잡았다가 실제보다 작아서
        max_allowed_packet을 넘겨 연결이 끊긴 적이 있어서(2026-07 실측),
        이제는 추정 대신 실측으로 바꿨다.
        """
        return self._max_packet_bytes // 2

    def estimate_row_bytes(self, doc_id: str, text: str, vector) -> int:
        """이 행이 INSERT SQL에 실제로 차지할 바이트 수 — 정확히 그 SQL 조각을
        만들어서 잰다(부호/자릿수가 들쭉날쭉한 float repr, escape 오버헤드 등을
        추정하지 않음)."""
        frag = f"(0, {self._conn.escape(doc_id)}, {self._conn.escape(text)}, {_vec_literal(vector)})"
        return len(frag.encode("utf-8"))

    def upload(self, name: str, data_iter) -> int:
        """(id, doc_id, text, vector) 이터레이터를 받아 삽입. vector: np.ndarray 또는 list."""
        batch: list[dict] = []
        batch_bytes = 0
        total = 0
        budget = self.insert_budget_bytes()

        for pid, doc_id, text, vec in data_iter:
            text = _trunc(text)
            row_bytes = self.estimate_row_bytes(doc_id, text, vec)
            if batch and batch_bytes + row_bytes > budget:
                self._insert_with_retry(name, batch)
                total += len(batch)
                batch.clear()
                batch_bytes = 0
            batch.append({"id": pid, "doc_id": doc_id, "text": text, "vector": vec})
            batch_bytes += row_bytes

        if batch:
            self._insert_with_retry(name, batch)
            total += len(batch)

        logger.info(f"  삽입 완료: {total:,}건")
        return total

    def _insert_with_retry(self, name: str, batch: list[dict]) -> None:
        # doc_id/text는 사용자 텍스트를 그대로 담으므로 conn.escape()로 안전하게 이스케이프한다
        # (vector는 숫자 리스트만 문자열화하므로 injection 경로가 없음).
        rows_sql = ",".join(
            f"({row['id']}, {self._conn.escape(row['doc_id'])}, "
            f"{self._conn.escape(row['text'])}, {_vec_literal(row['vector'])})"
            for row in batch
        )
        sql = f"INSERT INTO `{name}` (`id`, `doc_id`, `text`, `vector`) VALUES {rows_sql}"
        for attempt in range(3):
            try:
                self._exec(sql)
                return
            except Exception as exc:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                logger.warning(f"  insert 실패, {wait}s 후 재시도: {exc}")
                time.sleep(wait)

    def finalize(self, name: str) -> None:
        # StarRocks는 Milvus처럼 별도 load_collection 단계가 없음 — INSERT가 커밋되면
        # 바로 조회 가능. 다만 벡터 인덱스(HNSW) 빌드는 백그라운드에서 비동기로 진행되므로,
        # 색인 직후 바로 검색하면 인덱스가 아직 없어 첫 검색 몇 건이 브루트포스로 처리되며
        # 느릴 수 있다 (벤치마크에서 이 점을 감안할 것).
        logger.info(f"  삽입 완료 확인: {name} ({self.collection_size(name):,}건)")

    # ── 검색 ─────────────────────────────────────────────────────────────────

    def search(self, name: str, vectors, top_k: int = 10) -> list[list[tuple[str, float]]]:
        """ANN 검색. StarRocks는 쿼리 벡터를 한 번에 묶어 보낼 수 없어 벡터별로 순차 쿼리한다."""
        return [self._search_single(name, vectors[i], top_k) for i in range(len(vectors))]

    def search_one(self, name: str, vector, top_k: int = 10) -> list[tuple[str, float]]:
        return self._search_single(name, vector, top_k)

    def _search_single(self, name: str, vector, top_k: int) -> list[tuple[str, float]]:
        lit = _vec_literal(vector)
        sql = (
            f"SELECT `doc_id`, approx_cosine_similarity({lit}, `vector`) AS score "
            f"FROM `{name}` "
            f"ORDER BY approx_cosine_similarity({lit}, `vector`) DESC "
            f"LIMIT {int(top_k)}"
        )
        rows = self._fetchall(sql)
        return [(doc_id, float(score)) for doc_id, score in rows]
