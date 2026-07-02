"""corpus(약 65,000개 chunk) 임베딩을 사전에 계산해서 로컬 파일로 캐싱한다 (1회성 준비 작업).

baseline/ramp 테스트마다 milvus/minio를 통째로 재배포해 독립적으로 실행하지만, corpus
내용 자체는 테스트 사이에 바뀌지 않는다 — 매 재배포 때마다 Embedding API로 다시
임베딩할 필요 없이 여기서 한 번만 계산해두면, seed_collection.py는 이 캐시를 Milvus에
삽입만 하면 된다 (cache_query_vectors.py와 동일한 이유/구조).

사용법:
    python -m milvus_migration.loadtest.cache_chunk_vectors --data-root /data
    python -m milvus_migration.loadtest.cache_chunk_vectors --data-root /data --force
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
    ap = argparse.ArgumentParser(description="corpus 임베딩 사전 캐싱 (부하테스트 준비)")
    ap.add_argument("--data-root", default=os.getenv("DATA_ROOT", "/data"))
    ap.add_argument("--out", default=str(DEFAULT_CACHE_DIR), help="캐시 저장 디렉토리")
    ap.add_argument("--force", action="store_true", help="캐시가 이미 있어도 다시 임베딩")
    args = ap.parse_args()

    out_dir = Path(args.out)
    ids_path = out_dir / "chunk_ids.json"
    vecs_path = out_dir / "chunk_vectors.npy"

    if ids_path.exists() and vecs_path.exists() and not args.force:
        logger.info(f"[스킵] 이미 캐시 존재: {out_dir} (다시 만들려면 --force)")
        return

    cfg = Config.from_env()
    docs, _, _ = load_from_dir(args.data_root)
    doc_ids = list(docs.keys())
    texts = [f"{docs[d]['title']} {docs[d]['chunk']}".strip() for d in doc_ids]

    client = EmbeddingClient(cfg.embedding)
    logger.info(f"corpus {len(texts):,}건 임베딩 중 (model={cfg.embedding.model})...")
    vecs = client.encode(texts, batch_size=cfg.embedding.batch_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(vecs_path, vecs.astype(np.float32))
    ids_path.write_text(json.dumps(doc_ids, ensure_ascii=False), encoding="utf-8")
    logger.info(f"저장 완료: {vecs_path} (shape={vecs.shape}), {ids_path}")


if __name__ == "__main__":
    main()
