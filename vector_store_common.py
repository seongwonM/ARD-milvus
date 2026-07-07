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
_INSERT_BATCH = 64

# bench/retrieval_runner.py의 _build_index()가 쓰는 insert 배치 바이트 예산 — 마찬가지로
# 백엔드별로 다르게 계산하지 않고 하나의 값을 공유한다. StarRocks는 __init__에서
# max_allowed_packet을 이 값보다 넉넉히 올려둔다(store별 안전 마진은 달라도 예산 자체는 통일).
_INSERT_BUDGET_BYTES = 20_000_000


def _trunc(text: str) -> str:
    b = text.encode("utf-8")
    if len(b) <= _TEXT_MAX_BYTES:
        return text
    return b[:_TEXT_MAX_BYTES].decode("utf-8", errors="ignore")
