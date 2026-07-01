"""임베딩 모델별 top-100 후보 검색 (retrieval 전용 — reranking과 분리).

한 번 저장된 결과는 재사용됩니다 (같은 모델로 다시 돌리면 재검색하지 않고 스킵).
다시 계산하려면 --force.

사용법:
    python -m milvus_migration.bench.retrieval_runner --data-root /data --out results/retrieval

환경변수 (모델별로 다르게 설정 — k8s Job에서 모델마다 하나씩):
    MILVUS_URI, MILVUS_TOKEN
    EMBEDDING_API_ENDPOINT, EMBEDDING_API_KEY, EMBEDDING_MODEL

임베딩 차원은 설정값 없이 첫 API 호출 응답으로 자동 감지합니다.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time

from milvus_migration.bench.data_loader import load_from_dir
from milvus_migration.config import Config
from milvus_migration.embedding import EmbeddingClient
from milvus_migration.milvus_store import MilvusStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MILVUS_INSERT_BATCH = 5_000
_TOP_K = 100


def _safe_name(model_id: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", model_id.lower())[:200]


def _build_index(
    store: MilvusStore,
    client: EmbeddingClient,
    collection: str,
    docs: dict,
    dim: int,
    batch_size: int,
) -> tuple[float, float]:
    store.create_collection(collection, dim=dim)
    doc_ids = list(docs.keys())
    n_total = len(doc_ids)

    t0 = time.time()
    pid = 0
    pending: list[dict] = []

    for start in range(0, n_total, batch_size):
        chunk_ids = doc_ids[start: start + batch_size]
        texts = [
            f"{docs[d].get('title', '')} {docs[d]['chunk']}".strip()
            for d in chunk_ids
        ]
        embs = client.encode(texts, batch_size=batch_size)
        for doc_id, text, vec in zip(chunk_ids, texts, embs):
            pending.append({"id": pid, "doc_id": doc_id, "text": text, "vector": vec.tolist()})
            pid += 1

        if len(pending) >= _MILVUS_INSERT_BATCH:
            store._insert_with_retry(collection, pending)
            pending.clear()
            elapsed = round(time.time() - t0, 1)
            dps = round(pid / elapsed, 1) if elapsed > 0 else 0
            logger.info(f"  인덱싱: {pid:,}/{n_total:,}  elapsed={elapsed}s  ({dps} docs/s)")

    if pending:
        store._insert_with_retry(collection, pending)
        logger.info(f"  인덱싱: {pid:,}/{n_total:,}")

    store.finalize(collection)
    elapsed = round(time.time() - t0, 2)
    dps = round(n_total / elapsed, 1) if elapsed > 0 else 0
    logger.info(f"  인덱싱 완료: {elapsed}s  ({dps} docs/s)")
    return elapsed, dps


def run_retrieval(cfg: Config, docs: dict, queries: dict, out_path: str) -> dict:
    client = EmbeddingClient(cfg.embedding)
    store = MilvusStore(cfg.milvus.uri, cfg.milvus.token)

    model_id = cfg.embedding.model
    collection = _safe_name(model_id)

    logger.info(f"모델: {model_id}")
    dim = client.detect_dim()
    logger.info(f"  감지된 임베딩 차원: {dim}")

    n_docs = len(docs)
    index_build_sec = index_docs_per_sec = None
    if store.has_collection(collection) and store.collection_size(collection) == n_docs:
        logger.info(f"  [스킵] 인덱스 존재 ({n_docs:,}건) — 재사용")
    else:
        if store.has_collection(collection):
            logger.info(f"  [재색인] 불완전한 인덱스 삭제: {collection}")
            store.drop_collection(collection)
        index_build_sec, index_docs_per_sec = _build_index(
            store, client, collection, docs, dim, cfg.embedding.batch_size,
        )

    q_ids = list(queries.keys())
    q_texts = [queries[qid] for qid in q_ids]
    logger.info(f"query 인코딩 ({len(q_ids):,}건)...")
    t0 = time.time()
    q_embs = client.encode(q_texts, batch_size=cfg.embedding.batch_size)
    query_encode_sec = round(time.time() - t0, 2)
    logger.info(f"query 인코딩 완료: {query_encode_sec}s")

    logger.info(f"검색 (top-{_TOP_K})...")
    t0 = time.time()
    raw_results = store.search(collection, q_embs, top_k=_TOP_K)
    search_sec = round(time.time() - t0, 2)
    logger.info(f"검색 완료: {search_sec}s")

    results: dict[str, list[str]] = {}
    for qid, hits in zip(q_ids, raw_results):
        ids = [doc_id for doc_id, _ in hits]
        results[qid] = ids
        logger.info(f"  [{qid}] top{len(ids)} 후보: {ids}")

    payload = {
        "model": model_id,
        "dim": dim,
        "collection": collection,
        "n_docs": n_docs,
        "n_queries": len(q_ids),
        "top_k": _TOP_K,
        "index_build_sec": index_build_sec,
        "index_docs_per_sec": index_docs_per_sec,
        "query_encode_sec": query_encode_sec,
        "search_sec": search_sec,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"저장 완료: {out_path}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="임베딩 모델 top-100 후보 검색 (재사용 가능한 결과 저장)")
    ap.add_argument("--data-root", default=os.getenv("DATA_ROOT", "/data"))
    ap.add_argument("--out", default="results/retrieval", help="결과 저장 디렉토리")
    ap.add_argument("--force", action="store_true", help="이미 저장된 결과가 있어도 다시 실행")
    args = ap.parse_args()

    cfg = Config.from_env()
    out_path = os.path.join(args.out, f"{_safe_name(cfg.embedding.model)}_top100.json")

    if os.path.exists(out_path) and not args.force:
        logger.info(f"[스킵] 이미 결과 존재: {out_path} (재실행하려면 --force)")
        return

    logger.info(f"모델={cfg.embedding.model}  Milvus={cfg.milvus.uri}  data_root={args.data_root}")
    docs, queries, _qrels = load_from_dir(args.data_root)

    result = run_retrieval(cfg, docs, queries, out_path)

    print("\n===== RETRIEVAL RESULT SUMMARY =====")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, ensure_ascii=False, indent=2))
    print("===== END =====")


if __name__ == "__main__":
    main()
