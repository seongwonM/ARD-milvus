"""CLI 진입점.

사용법:
    # 인덱싱 (JSON 파일)
    python -m milvus_migration.main index --file docs.json

    # 검색
    python -m milvus_migration.main search --query "벡터 데이터베이스 추천" --top-k 5

    # 컬렉션 정보
    python -m milvus_migration.main info

JSON 입력 형식:
    [
        {"id": "doc-1", "text": "내용"},
        {"id": "doc-2", "title": "제목", "text": "내용"},
        ...
    ]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def cmd_index(args: argparse.Namespace) -> None:
    from .pipeline import Pipeline

    with open(args.file, encoding="utf-8") as f:
        docs = json.load(f)

    pipe = Pipeline.from_env()
    pipe.index(
        docs,
        vector_mode=args.vector_mode,
        recreate=args.recreate,
        batch_size=args.batch_size or None,
    )


def cmd_search(args: argparse.Namespace) -> None:
    from .pipeline import Pipeline

    pipe = Pipeline.from_env()
    results = pipe.search(
        args.query, top_k=args.top_k, vector_mode=args.vector_mode,
    )

    print(f"\n[검색 결과] '{args.query}' (top-{args.top_k})")
    print("-" * 60)
    for rank, (doc_id, score) in enumerate(results, 1):
        print(f"  {rank:3d}. {doc_id:<40s}  score={score:.4f}")


def cmd_info(args: argparse.Namespace) -> None:
    from .pipeline import Pipeline

    pipe = Pipeline.from_env()
    size = pipe.collection_size()
    print(f"컬렉션 크기: {size:,}건")


def main() -> None:
    parser = argparse.ArgumentParser(description="Milvus VectorDB 마이그레이션 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="문서를 임베딩 후 Milvus에 인덱싱")
    p_index.add_argument("--file",         required=True,      help="JSON 문서 파일 경로")
    p_index.add_argument("--vector-mode",  default="dense",    choices=["dense", "sparse"])
    p_index.add_argument("--recreate",     action="store_true", help="기존 컬렉션 삭제 후 재생성")
    p_index.add_argument("--batch-size",   type=int, default=None, help="임베딩 배치 크기")

    # search
    p_search = sub.add_parser("search", help="텍스트로 유사 문서 검색")
    p_search.add_argument("--query",       required=True)
    p_search.add_argument("--top-k",       type=int, default=10)
    p_search.add_argument("--vector-mode", default="dense", choices=["dense", "sparse"])

    # info
    sub.add_parser("info", help="컬렉션 정보 출력")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "info":
        cmd_info(args)


if __name__ == "__main__":
    main()
