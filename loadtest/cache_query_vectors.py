"""쿼리 12,000개를 사전에 임베딩해서 로컬 파일로 캐싱한다 (1회성 준비 작업).

Locust 부하테스트 루프 안에서는 임베딩 API를 호출하지 않고 이 캐시만 읽어서
Milvus 검색만 반복한다 — 임베딩 API 자체의 지연/레이트리밋이 Milvus+MinIO
부하테스트 결과에 섞이지 않도록 분리하기 위함이다.

사용법:
    python -m milvus_migration.loadtest.cache_query_vectors --data-root /data
    python -m milvus_migration.loadtest.cache_query_vectors --data-root /data --force
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

from ..bench.data_loader import load_from_dir
from ..config import Config
from ..embedding import EmbeddingClient
from .common import DEFAULT_CACHE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="쿼리 임베딩 사전 캐싱 (부하테스트 준비)")
    ap.add_argument("--data-root", default=os.getenv("DATA_ROOT", "/data"))
    ap.add_argument("--out", default=str(DEFAULT_CACHE_DIR), help="캐시 저장 디렉토리")
    ap.add_argument("--force", action="store_true", help="캐시가 이미 있어도 다시 임베딩")
    args = ap.parse_args()

    out_dir = Path(args.out)
    ids_path = out_dir / "query_ids.json"
    vecs_path = out_dir / "query_vectors.npy"

    if ids_path.exists() and vecs_path.exists() and not args.force:
        logger.info(f"[스킵] 이미 캐시 존재: {out_dir} (다시 만들려면 --force)")
        return

    cfg = Config.from_env()
    _, queries, _ = load_from_dir(args.data_root)
    q_ids = list(queries.keys())
    q_texts = [queries[qid] for qid in q_ids]

    client = EmbeddingClient(cfg.embedding)
    logger.info(f"쿼리 {len(q_texts):,}개 임베딩 중 (model={cfg.embedding.model})...")
    vecs = client.encode(q_texts, batch_size=cfg.embedding.batch_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(vecs_path, vecs.astype(np.float32))
    ids_path.write_text(json.dumps(q_ids, ensure_ascii=False), encoding="utf-8")
    logger.info(f"저장 완료: {vecs_path} (shape={vecs.shape}), {ids_path}")


if __name__ == "__main__":
    main()
