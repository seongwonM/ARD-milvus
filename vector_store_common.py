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

# bench/retrieval_runner.py의 _build_index()가 쓰는 insert 배치 바이트 예산 — 백엔드 간 공유.
# 한때 StarRocks에 이 20MB를 그대로 썼더니 모델 상관없이 첫 insert에서 BE가 OOMKilled돼서
# StarRocks만 8MB로 따로 낮췄었는데(2026-07 실측), 근본 원인이 배치 크기 자체가 아니라
# jemalloc이 해제한 메모리를 OS에 안 돌려주는 문제(StarRocks#67607)로 확인돼 k8s/starrocks.yaml/
# loadtest/k8s/starrocks.yaml에 MALLOC_CONF(dirty_decay_ms/muzzy_decay_ms 단축)를 추가했다.
# 이 완화책을 믿고 다시 Milvus와 동일한 값으로 통일한다 — 그래도 재발하면 원인이 배치 크기
# (또는 jemalloc 튜닝으로 완전히 못 잡는 다른 메모리 문제)일 가능성이 남아있다는 뜻이니
# starrocks_store.py의 8MB 폴백으로 다시 낮출 것.
_INSERT_BUDGET_BYTES = 20_000_000


def _trunc(text: str) -> str:
    b = text.encode("utf-8")
    if len(b) <= _TEXT_MAX_BYTES:
        return text
    return b[:_TEXT_MAX_BYTES].decode("utf-8", errors="ignore")
