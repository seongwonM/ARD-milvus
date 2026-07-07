"""StarRocks VectorStore - dense 임베딩 검색 (Vector Index / HNSW).

MilvusStore와 동일한 인터페이스(has_collection/collection_size/create_collection/
drop_collection/upload/_insert_with_retry/finalize/search/search_one)를 제공해서
Pipeline/retrieval_runner/loadtest 쪽 코드가 백엔드 차이를 신경 쓰지 않게 한다
(store_factory.build_store()가 VECTOR_BACKEND 환경변수로 둘 중 하나를 고른다).

테이블 스키마:
  id      BIGINT
  doc_id  VARCHAR(512)
  text    VARCHAR(2048)
  vector  ARRAY<FLOAT> + VECTOR INDEX(HNSW, cosine_similarity) — 인덱스는 적재 완료 후
          finalize()에서 ALTER TABLE로 한 번에 추가한다(create_collection()엔 없음).

주의(StarRocks 문서 기준, 2026-07 확인):
- 벡터 인덱스는 v3.4+ shared-nothing 클러스터에서만 지원 (starrocks/allin1-ubuntu:3.5.x 이상 사용).
- FE에 enable_experimental_vector=true가 켜져 있어야 함 — 연결 시 자동으로 켠다(멱등).
- ANN 검색은 Milvus처럼 여러 쿼리 벡터를 한 번에 묶어 보낼 수 없고, 쿼리 벡터 1개당
  SQL 1건이 필요하다 — search()는 벡터별로 순차 쿼리한다 (배치 API가 없는 게 아니라
  StarRocks ANN 문법 자체가 상수 벡터 1개 기준이라 구조적으로 그렇다).
- 인덱스를 CREATE TABLE 시점에 걸어두고 계속 INSERT하면, compaction이 매번 HNSW 인덱스도
  다시 빌드해야 해서 INSERT 속도를 못 따라가 tablet에 버전이 쌓이고 결국 쓰기가 거부되거나
  (StarRocks#56697 "too many versions") BE가 죽어 커넥션이 끊긴다(2026-07 실측) — 그래서
  적재 중엔 인덱스 없이 넣고 finalize()에서 한 번만 빌드한다.
"""
from __future__ import annotations

import logging
import threading
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

        self._pymysql = pymysql
        self._host, self._port, self._user, self._password = host, port, user, password
        self._database = database
        self._pending_index_dim: dict[str, int] = {}  # create_collection()이 채우고 finalize()가 씀
        # 커넥션을 스레드-로컬로 둔다: loadtest는 이 인스턴스 하나를 gevent 내장
        # threadpool(진짜 OS 스레드 여러 개)에서 동시에 호출하는데(threadpool_search_one),
        # pymysql 커넥션 하나를 여러 스레드가 동시에 쓰면 MySQL 프로토콜 스트림이 깨져
        # "Lost connection to MySQL server"가 난다(2026-07 실측). bench는 삽입을 단일
        # 스레드로 순차 호출해서 이 문제가 안 보였을 뿐이다.
        self._local = threading.local()
        self._connect()
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

    def _connect(self) -> None:
        """호출한 스레드용 커넥션을 새로 만든다(스레드-로컬)."""
        self._local.conn = self._pymysql.connect(
            host=self._host, port=self._port, user=self._user, password=self._password,
            autocommit=True, connect_timeout=300, read_timeout=300, write_timeout=300,
        )

    @property
    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            # 이 스레드에서 처음 쓰는 커넥션 — 새로 만들고 DB를 선택해둔다
            # (__init__을 실행한 스레드가 아닌 threadpool 워커 스레드에서 최초 호출될 때).
            self._connect()
            conn = self._local.conn
            with conn.cursor() as cur:
                cur.execute(f"USE `{self._database}`")
        return conn

    def _is_connection_error(self, exc: Exception) -> bool:
        """커넥션 자체가 죽어서 그냥 재시도해봐야 소용없고 재연결이 필요한 경우인지 판별.

        max_allowed_packet 초과 등으로 서버가 소켓을 끊으면, 같은 커넥션으로 재시도할 때
        원래 에러(packet too large 등) 대신 pymysql.err.InterfaceError(0, '')처럼 애매한
        에러만 반복돼서 원인이 가려진다(2026-07 실측) — 이럴 땐 재시도 전에 재연결해야 한다.
        """
        if isinstance(exc, self._pymysql.err.InterfaceError):
            return True
        if isinstance(exc, self._pymysql.err.OperationalError):
            code = exc.args[0] if exc.args else None
            return code in (2006, 2013, 2014)  # gone away / lost connection / commands out of sync
        return False

    def _reconnect(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
        self._connect()
        self._exec(f"USE `{self._database}`")
        logger.warning("[StarRocks] 재연결 완료")

    def _exec(self, sql: str):
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return cur

    def _exec_with_retry(self, sql: str, attempts: int = 12, wait_sec: float = 5.0):
        for attempt in range(attempts):
            try:
                return self._exec(sql)
            except Exception as exc:
                logger.warning(
                    f"[StarRocks] 초기화 쿼리 실패(시도 {attempt + 1}/{attempts}, FE 초기화 중일 수 있음): "
                    f"{type(exc).__name__}: {exc}"
                )
                if attempt == attempts - 1:
                    raise
                if self._is_connection_error(exc):
                    try:
                        self._reconnect()
                    except Exception as reconnect_exc:
                        logger.warning(f"[StarRocks] 재연결 실패: {reconnect_exc}")
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
        """벡터 인덱스는 여기서 안 만든다 — finalize()가 적재를 다 끝낸 뒤 한 번에 만든다.

        처음부터 INDEX ... USING VECTOR를 걸어두면 매 INSERT마다 compaction이
        HNSW 인덱스까지 다시 빌드해야 해서 훨씬 느려지고, insert가 그보다 빠르면
        tablet에 버전이 쌓여 결국 쓰기가 거부되거나(StarRocks#56697 "too many
        versions") BE가 리소스 부족으로 죽어 커넥션이 끊긴다(2026-07 실측:
        Lost connection to MySQL server during query). 적재 중엔 인덱스 없이
        빠르게 넣고, 다 넣은 뒤 ALTER TABLE로 인덱스를 한 번만 빌드하면 이 문제가
        없다.
        """
        self._exec(f"""
            CREATE TABLE `{name}` (
                `id`     BIGINT       NOT NULL,
                `doc_id` VARCHAR(512) NOT NULL,
                `text`   VARCHAR({_TEXT_MAX_BYTES}),
                `vector` ARRAY<FLOAT> NOT NULL
            ) ENGINE=OLAP
            DUPLICATE KEY(id)
            DISTRIBUTED BY HASH(id) BUCKETS 4
            PROPERTIES ("replication_num" = "1")
        """)
        self._pending_index_dim[name] = dim
        logger.info(f"  테이블 생성(벡터 인덱스는 적재 후 별도 빌드): {name}  dim={dim}")

    def drop_collection(self, name: str) -> None:
        self._exec(f"DROP TABLE IF EXISTS `{name}`")
        logger.info(f"  테이블 삭제: {name}")

    # ── 데이터 삽입 ──────────────────────────────────────────────────────────

    def insert_budget_bytes(self) -> int:
        """호출자(bench/retrieval_runner.py, upload())가 한 번의 INSERT SQL에
        몇 바이트어치를 모아 보낼지 결정할 때 쓰는 예산.

        실제 조회한 max_allowed_packet의 1/4만 쓴다. 행 하나의 바이트 수는
        estimate_row_bytes()가 실제 SQL 조각을 만들어 정확히 재지만, 그래도
        여유를 넉넉히 두는 이유: 이 SQL 텍스트 자체의 바이트 수 말고도 MySQL
        프로토콜 패킷 헤더, 다른 세션의 동시 사용 등 우리가 안 재는 오버헤드가
        더 있을 수 있다. 절반(1/2)으로는 실측치가 예산의 99.94%까지 차서(748행,
        16,766,831B vs 예산 16,777,216B) 여유가 사실상 없었다(2026-07 실측).
        """
        return self._max_packet_bytes // 4

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
                logger.warning(
                    f"  insert 실패(시도 {attempt + 1}/3, {len(batch)}행, "
                    f"{len(sql.encode('utf-8')):,}B): {type(exc).__name__}: {exc}"
                )
                if attempt == 2:
                    raise
                if self._is_connection_error(exc):
                    try:
                        self._reconnect()
                    except Exception as reconnect_exc:
                        logger.warning(f"  재연결 실패: {reconnect_exc}")
                time.sleep(2 ** attempt)

    def finalize(self, name: str) -> None:
        # StarRocks는 Milvus처럼 별도 load_collection 단계가 없음 — INSERT가 커밋되면
        # 바로 조회 가능. create_collection()이 인덱스를 안 만들어뒀다면(정상 경로) 여기서
        # 적재가 끝난 뒤 한 번에 벡터 인덱스를 빌드한다 — 끝날 때까지 대기하므로 finalize()가
        # 반환하면 바로 검색 가능하다(인덱스 없이 브루트포스로 검색되는 구간이 없어짐).
        dim = self._pending_index_dim.pop(name, None)
        if dim is not None:
            logger.info(f"  벡터 인덱스 빌드 시작: {name} (dim={dim})")
            self._exec(f"""
                ALTER TABLE `{name}` ADD INDEX vec_idx (vector) USING VECTOR (
                    "index_type" = "hnsw",
                    "dim" = "{dim}",
                    "metric_type" = "cosine_similarity",
                    "is_vector_normed" = "false",
                    "M" = "16",
                    "efconstruction" = "100"
                )
            """)
            self._wait_index_build(name)
            logger.info(f"  벡터 인덱스 빌드 완료: {name}")
        logger.info(f"  삽입 완료 확인: {name} ({self.collection_size(name):,}건)")

    def _wait_index_build(self, name: str, timeout_sec: float = 1800.0, poll_sec: float = 5.0) -> None:
        """ALTER TABLE ... ADD INDEX ... USING VECTOR job이 끝날 때까지 SHOW ALTER TABLE
        COLUMN을 폴링한다(State=FINISHED)."""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            with self._conn.cursor(self._pymysql.cursors.DictCursor) as cur:
                cur.execute(f"SHOW ALTER TABLE COLUMN FROM `{self._database}`")
                rows = cur.fetchall()
            matching = [r for r in rows if r.get("TableName") == name]
            if matching:
                state = matching[-1].get("State")
                if state == "FINISHED":
                    return
                if state == "CANCELLED":
                    raise RuntimeError(f"[StarRocks] 벡터 인덱스 빌드 실패({name}): {matching[-1].get('Msg')}")
            time.sleep(poll_sec)
        raise TimeoutError(f"[StarRocks] 벡터 인덱스 빌드가 {timeout_sec:.0f}s 안에 안 끝남: {name}")

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
        for attempt in range(3):
            try:
                rows = self._fetchall(sql)
                return [(doc_id, float(score)) for doc_id, score in rows]
            except Exception as exc:
                logger.warning(
                    f"  search 실패(시도 {attempt + 1}/3): {type(exc).__name__}: {exc}"
                )
                if attempt == 2:
                    raise
                if self._is_connection_error(exc):
                    try:
                        self._reconnect()
                    except Exception as reconnect_exc:
                        logger.warning(f"  재연결 실패: {reconnect_exc}")
                time.sleep(2 ** attempt)
