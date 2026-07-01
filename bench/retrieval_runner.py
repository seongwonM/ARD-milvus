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
from milvus_migration.bench.evaluator import evaluate
from milvus_migration.config import Config
from milvus_migration.embedding import EmbeddingClient
from milvus_migration.milvus_store import MilvusStore, _trunc, _TEXT_MAX_BYTES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# gRPC 기본 메시지 크기 제한(64MB)보다 한참 여유있게 — 고차원 모델(Qwen3 dim=4096 등)에서
# insert batch가 너무 크면 "received message larger than max" 에러가 남
_INSERT_BUDGET_BYTES = 20_000_000
_TOP_K = 100


def _safe_name(model_id: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", model_id.lower())[:200]


def _insert_batch_size(dim: int) -> int:
    per_row = dim * 4 + _TEXT_MAX_BYTES + 600  # vector(float32) + text + doc_id/protobuf 오버헤드
    return max(200, min(5_000, _INSERT_BUDGET_BYTES // per_row))


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
    insert_batch = _insert_batch_size(dim)
    logger.info(f"  insert batch size: {insert_batch} (dim={dim})")

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
            pending.append({"id": pid, "doc_id": doc_id, "text": _trunc(text), "vector": vec.tolist()})
            pid += 1

        if len(pending) >= insert_batch:
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


def run_retrieval(cfg: Config, docs: dict, queries: dict, qrels: dict, out_path: str) -> dict:
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
    query_encode_qps = round(len(q_ids) / query_encode_sec, 1) if query_encode_sec > 0 else 0.0
    logger.info(f"query 인코딩 완료: {query_encode_sec}s  ({query_encode_qps} q/s)")

    logger.info(f"검색 (top-{_TOP_K})...")
    t0 = time.time()
    raw_results = store.search(collection, q_embs, top_k=_TOP_K)
    search_sec = round(time.time() - t0, 2)
    search_qps = round(len(q_ids) / search_sec, 1) if search_sec > 0 else 0.0
    logger.info(f"검색 완료: {search_sec}s  ({search_qps} q/s)")

    _LOG_SAMPLE = 5
    results: dict[str, list[str]] = {}
    run: dict[str, dict[str, float]] = {}
    for i, (qid, hits) in enumerate(zip(q_ids, raw_results)):
        results[qid] = [doc_id for doc_id, _ in hits]
        run[qid] = {doc_id: score for doc_id, score in hits}
        if i < _LOG_SAMPLE:
            logger.info(f"  [샘플 {i+1}/{_LOG_SAMPLE}] [{qid}] top{len(hits)} 후보: {results[qid]}")
    logger.info(f"  검색 결과 {len(results):,}개 쿼리 확보 (전체 top-{_TOP_K} 후보는 JSON 파일에 저장됨)")

    metrics = evaluate(run, qrels)
    logger.info(
        f"  [리랭킹 전 raw 검색 성능] NDCG@10={metrics.get('ndcg_at_10')}  @20={metrics.get('ndcg_at_20')}  "
        f"@50={metrics.get('ndcg_at_50')}  @100={metrics.get('ndcg_at_100')}"
    )
    logger.info(
        f"  MRR@10={metrics.get('mrr_at_10')}  "
        f"Recall@10={metrics.get('recall_at_10')}  @50={metrics.get('recall_at_50')}  @100={metrics.get('recall_at_100')}"
    )

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
        "query_encode_qps": query_encode_qps,
        "search_sec": search_sec,
        "search_qps": search_qps,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **metrics,
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
    docs, queries, qrels = load_from_dir(args.data_root)

    result = run_retrieval(cfg, docs, queries, qrels, out_path)

    print("\n===== RETRIEVAL RESULT SUMMARY =====")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, ensure_ascii=False, indent=2))
    print("===== END =====")


if __name__ == "__main__":
    main()
