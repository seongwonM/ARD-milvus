"""Milvus/StarRocks store가 공유하는 상수/유틸.

백엔드 비교 벤치마크가 공정하려면 텍스트 길이 제한이나 검색 배치 크기 같은
파라미터가 백엔드에 따라 달라지면 안 된다 — 그래서 한 곳에 모아두고 양쪽에서
동일하게 가져다 쓴다.
"""
from __future__ import annotations

_TEXT_MAX_BYTES = 2000  # VARCHAR max_length은 바이트 기준 — 여유분 확보
_SEARCH_CHUNK = 128     # 배치당 쿼리 수 (Milvus gRPC 메시지 크기 제한 회피 + 백엔드 간 비교 공정성)

# loadtest 시딩(pipeline.index_vectors() → store.upload())이 쓰는 고정 배치 행수.
# 예전엔 StarRocks만 max_allowed_packet 기준 바이트 예산으로 따로 계산해서 Milvus(64행
# 고정)보다 배치가 훨씬 컸다 — 백엔드 차이가 배치 크기에 섞이면 안 되므로 고정값 하나로 통일.
# (행수 기준 공유는 안전함 — 64행이면 어느 모델/차원이든 SQL 텍스트가 몇 MB 수준이라 문제 없음.)
_INSERT_BATCH = 64

# bench/retrieval_runner.py의 _build_index()가 쓰는 insert 배치 바이트 예산은 백엔드 간
# 공유하지 않는다. StarRocks에 이 20MB를 그대로 썼더니 모델 상관없이 첫 insert에서 BE가
# OOMKilled됐고(2026-07 실측), MALLOC_CONF(dirty_decay_ms/muzzy_decay_ms를 0으로) 튜닝을
# 넣어도 동일하게 재발했다 — jemalloc 미반환 문제가 아니라, 20MB짜리 SQL 텍스트를 파싱하는
# 순간 실제로 살아있는(active) 메모리 자체가 크다는 뜻으로 보인다(allocator 튜닝으로 못
# 고치는 종류). SQL 텍스트로 파싱/실행하는 StarRocks의 INSERT는 Milvus의 바이너리 gRPC
# 삽입보다 바이트당 메모리 비용이 훨씬 커서, 같은 바이트 수라도 실제 메모리 부담이 다르다.
# 각 store가 자기 프로토콜에 맞는 안전한 값을 따로 갖는다(Milvus: milvus_store.py의
# _INSERT_BUDGET_BYTES, StarRocks: starrocks_store.py 자체 값 8MB).


def _trunc(text: str) -> str:
    b = text.encode("utf-8")
    if len(b) <= _TEXT_MAX_BYTES:
        return text
    return b[:_TEXT_MAX_BYTES].decode("utf-8", errors="ignore")
