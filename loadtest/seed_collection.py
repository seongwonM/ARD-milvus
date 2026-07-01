"""corpus(약 65,000개 chunk)를 Milvus에 색인한다 (1회성 준비 작업).

기존 Pipeline.index()를 그대로 재사용한다. 이미 컬렉션이 있고 문서 수가
corpus 크기와 일치하면 기본적으로 스킵한다 (--recreate로 강제 재색인).

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="부하테스트용 corpus 색인 (1회성 준비 작업)")
    ap.add_argument("--data-root", default=os.getenv("DATA_ROOT", "/data"))
    ap.add_argument("--recreate", action="store_true", help="기존 컬렉션 삭제 후 재색인")
    args = ap.parse_args()

    cfg = Config.from_env()
    docs, _, _ = load_from_dir(args.data_root)
    doc_list = [{"id": doc_id, "title": d["title"], "text": d["chunk"]} for doc_id, d in docs.items()]

    store = MilvusStore(cfg.milvus.uri, cfg.milvus.token)
    if not args.recreate and store.has_collection(cfg.milvus.collection):
        size = store.collection_size(cfg.milvus.collection)
        if size == len(doc_list):
            logger.info(f"[스킵] 이미 색인됨: {cfg.milvus.collection} ({size:,}건) — 재색인하려면 --recreate")
            return
        logger.info(f"[불일치] 기존 컬렉션 크기({size:,}) != corpus({len(doc_list):,}) — 재색인 진행")

    logger.info(f"색인 시작: {len(doc_list):,}건 → '{cfg.milvus.collection}'")
    pipe = Pipeline(cfg)
    pipe.index(doc_list, recreate=args.recreate)
    logger.info(f"색인 완료: {pipe.collection_size():,}건")


if __name__ == "__main__":
    main()
