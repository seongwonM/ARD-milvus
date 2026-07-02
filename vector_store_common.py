"""Milvus/StarRocks store가 공유하는 상수/유틸.

백엔드 비교 벤치마크가 공정하려면 텍스트 길이 제한이나 검색 배치 크기 같은
파라미터가 백엔드에 따라 달라지면 안 된다 — 그래서 한 곳에 모아두고 양쪽에서
동일하게 가져다 쓴다.
"""
from __future__ import annotations

_TEXT_MAX_BYTES = 2000  # VARCHAR max_length은 바이트 기준 — 여유분 확보
_SEARCH_CHUNK = 128     # 배치당 쿼리 수 (Milvus gRPC 메시지 크기 제한 회피 + 백엔드 간 비교 공정성)


def _trunc(text: str) -> str:
    b = text.encode("utf-8")
    if len(b) <= _TEXT_MAX_BYTES:
        return text
    return b[:_TEXT_MAX_BYTES].decode("utf-8", errors="ignore")
