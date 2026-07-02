"""corpus(약 65,000개 chunk)를 Milvus에 색인한다 (milvus/minio 재배포마다 반복 실행).

cache_chunk_vectors.py가 미리 계산해둔 임베딩 캐시를 그대로 삽입한다 — Embedding API를
다시 호출하지 않으므로, baseline/ramp 테스트마다 milvus/minio를 통째로 재배포해도
매번 6만 건을 재임베딩하는 비용 없이 벡터 삽입만 반복하면 된다.
(캐시가 없으면 먼저 cache_chunk_vectors.py를 실행해야 함)

사용법:
    python -m milvus_migration.loadtest.seed_collection --data-root /data
    python -m milvus_migration.loadtest.seed_collection --data-root /data --recreate
"""
from __future__ import annotations

import argparse
import logging
import os

from ..bench.data_loader import load_from_dir
from ..config import Config
from ..milvus_store import MilvusStore
from ..pipeline import Pipeline
from .common import load_chunk_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="부하테스트용 corpus 색인 (캐시된 벡터 삽입)")
    ap.add_argument("--data-root", default=os.getenv("DATA_ROOT", "/data"))
    ap.add_argument("--recreate", action="store_true", help="기존 컬렉션 삭제 후 재색인")
    args = ap.parse_args()

    cfg = Config.from_env()
    docs, _, _ = load_from_dir(args.data_root)
    chunk_ids, vectors = load_chunk_cache()
    texts = [f"{docs[d]['title']} {docs[d]['chunk']}".strip() for d in chunk_ids]

    store = MilvusStore(cfg.milvus.uri, cfg.milvus.token)
    if not args.recreate and store.has_collection(cfg.milvus.collection):
        size = store.collection_size(cfg.milvus.collection)
        if size == len(chunk_ids):
            logger.info(f"[스킵] 이미 색인됨: {cfg.milvus.collection} ({size:,}건) — 재색인하려면 --recreate")
            return
        logger.info(f"[불일치] 기존 컬렉션 크기({size:,}) != corpus({len(chunk_ids):,}) — 재색인 진행")

    logger.info(f"색인 시작(캐시된 벡터 삽입): {len(chunk_ids):,}건 → '{cfg.milvus.collection}'")
    pipe = Pipeline(cfg)
    pipe.index_vectors(chunk_ids, texts, vectors, recreate=args.recreate)
    logger.info(f"색인 완료: {pipe.collection_size():,}건")


if __name__ == "__main__":
    main()
