"""임베딩 모델별 top-100 후보 검색 (retrieval 전용 — reranking과 분리).

한 번 저장된 결과는 재사용됩니다 (같은 모델로 다시 돌리면 재검색하지 않고 스킵).
다시 계산하려면 --force. Milvus에 인덱스가 이미 있어도 강제로 재색인하려면 --reindex
(인덱싱 속도를 다시 재고 싶을 때 필요 — 인덱스가 있으면 기본적으로는 재사용하고 스킵함).

배치별 소요 시간(청크 인덱싱 배치, 쿼리 인코딩 배치)도 함께 저장합니다. 임베딩 API가
배치 단위로 호출되기 때문에 개별 청크/쿼리 하나하나의 시간은 잴 수 없고, 실제로 측정
가능한 가장 작은 단위인 "배치 1회 호출" 단위로 sec_per_chunk / sec_per_query를 기록합니다.

사용법:
    python -m milvus_migration.bench.retrieval_runner --data-root /data --out results/retrieval
    python -m milvus_migration.bench.retrieval_runner --data-root /data --reindex

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

import numpy as np

from milvus_migration.bench.data_loader import load_from_dir
from milvus_migration.bench.evaluator import evaluate
from milvus_migration.config import Config
from milvus_migration.embedding import EmbeddingClient
from milvus_migration.store_factory import build_store
from milvus_migration.vector_store_common import _trunc, _SEARCH_CHUNK

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_TOP_K = 100


def _safe_name(model_id: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", model_id.lower())[:200]


def _build_index(
    store,
    client: EmbeddingClient,
    collection: str,
    docs: dict,
    dim: int,
    batch_size: int,
) -> tuple[float, float, list[dict], list[dict]]:
    store.create_collection(collection, dim=dim)
    doc_ids = list(docs.keys())
    n_total = len(doc_ids)
    insert_budget = store.insert_budget_bytes()
    logger.info(f"  insert byte budget: {insert_budget:,} (dim={dim})")

    t0 = time.time()
    pid = 0
    pending: list[dict] = []
    pending_bytes = 0
    chunk_timings: list[dict] = []
    insert_timings: list[dict] = []

    for batch_idx, start in enumerate(range(0, n_total, batch_size)):
        chunk_ids = doc_ids[start: start + batch_size]
        texts = [
            f"{docs[d].get('title', '')} {docs[d]['chunk']}".strip()
            for d in chunk_ids
        ]

        t_batch = time.time()
        embs = client.encode(texts, batch_size=batch_size)
        batch_sec = time.time() - t_batch
        n_chunks = len(chunk_ids)
        chunk_timings.append({
            "batch_index": batch_idx,
            "n_chunks": n_chunks,
            "elapsed_sec": round(batch_sec, 4),
            "sec_per_chunk": round(batch_sec / n_chunks, 5) if n_chunks else 0.0,
            "chunk_ids": chunk_ids,
        })
        logger.info(
            f"  [청크 배치 {batch_idx}] n={n_chunks}  elapsed={round(batch_sec, 3)}s  "
            f"sec/chunk={round(batch_sec / n_chunks, 5) if n_chunks else 0.0}"
        )

        for doc_id, text, vec in zip(chunk_ids, texts, embs):
            trunc_text = _trunc(text)
            pending.append({"id": pid, "doc_id": doc_id, "text": trunc_text, "vector": vec.tolist()})
            pending_bytes += store.estimate_row_bytes(doc_id, trunc_text, vec)
            pid += 1

        if pending_bytes >= insert_budget:
            n_rows = len(pending)
            t_ins = time.time()
            store._insert_with_retry(collection, pending)
            ins_sec = time.time() - t_ins
            insert_timings.append({
                "batch_index": len(insert_timings),
                "n_rows": n_rows,
                "elapsed_sec": round(ins_sec, 4),
                "sec_per_row": round(ins_sec / n_rows, 6) if n_rows else 0.0,
            })
            pending.clear()
            pending_bytes = 0
            elapsed = round(time.time() - t0, 1)
            dps = round(pid / elapsed, 1) if elapsed > 0 else 0
            logger.info(
                f"  인덱싱: {pid:,}/{n_total:,}  elapsed={elapsed}s  ({dps} docs/s)  "
                f"insert={round(ins_sec, 3)}s ({n_rows} rows)"
            )

    if pending:
        n_rows = len(pending)
        t_ins = time.time()
        store._insert_with_retry(collection, pending)
        ins_sec = time.time() - t_ins
        insert_timings.append({
            "batch_index": len(insert_timings),
            "n_rows": n_rows,
            "elapsed_sec": round(ins_sec, 4),
            "sec_per_row": round(ins_sec / n_rows, 6) if n_rows else 0.0,
        })
        logger.info(f"  인덱싱: {pid:,}/{n_total:,}  insert={round(ins_sec, 3)}s ({n_rows} rows)")

    store.finalize(collection)
    elapsed = round(time.time() - t0, 2)
    dps = round(n_total / elapsed, 1) if elapsed > 0 else 0
    total_insert_sec = round(sum(t["elapsed_sec"] for t in insert_timings), 3)
    logger.info(
        f"  인덱싱 완료: {elapsed}s  ({dps} docs/s)  "
        f"insert 총합={total_insert_sec}s ({round(100*total_insert_sec/elapsed, 1) if elapsed > 0 else 0}%)"
    )
    return elapsed, dps, chunk_timings, insert_timings


def run_retrieval(
    cfg: Config,
    docs: dict,
    queries: dict,
    qrels: dict,
    out_path: str,
    force_reindex: bool = False,
) -> dict:
    client = EmbeddingClient(cfg.embedding)
    store = build_store(cfg)

    model_id = cfg.embedding.model
    collection = _safe_name(model_id)

    logger.info(f"모델: {model_id}")
    dim = client.detect_dim()
    logger.info(f"  감지된 임베딩 차원: {dim}")

    n_docs = len(docs)
    index_build_sec = index_docs_per_sec = None
    chunk_timings: list[dict] = []
    insert_timings: list[dict] = []
    already_indexed = store.has_collection(collection) and store.collection_size(collection) == n_docs

    if already_indexed and not force_reindex:
        logger.info(f"  [스킵] 인덱스 존재 ({n_docs:,}건) — 재사용 (재색인하려면 --reindex)")
    else:
        if store.has_collection(collection):
            reason = "강제 재색인(--reindex)" if force_reindex else "불완전한 인덱스"
            logger.info(f"  [재색인] {reason} — 기존 컬렉션 삭제: {collection}")
            store.drop_collection(collection)
        index_build_sec, index_docs_per_sec, chunk_timings, insert_timings = _build_index(
            store, client, collection, docs, dim, cfg.embedding.batch_size,
        )

    q_ids = list(queries.keys())
    bs = cfg.embedding.batch_size
    logger.info(f"query 인코딩 ({len(q_ids):,}건)...")
    t0 = time.time()
    q_embs_list = []
    query_timings: list[dict] = []
    for batch_idx, start in enumerate(range(0, len(q_ids), bs)):
        batch_qids = q_ids[start: start + bs]
        batch_texts = [queries[qid] for qid in batch_qids]

        t_batch = time.time()
        batch_embs = client.encode(batch_texts, batch_size=bs)
        batch_sec = time.time() - t_batch
        n_q = len(batch_qids)
        query_timings.append({
            "batch_index": batch_idx,
            "n_queries": n_q,
            "elapsed_sec": round(batch_sec, 4),
            "sec_per_query": round(batch_sec / n_q, 5) if n_q else 0.0,
            "query_ids": batch_qids,
        })
        logger.info(
            f"  [쿼리 배치 {batch_idx}] n={n_q}  elapsed={round(batch_sec, 3)}s  "
            f"sec/query={round(batch_sec / n_q, 5) if n_q else 0.0}"
        )
        q_embs_list.append(batch_embs)

    q_embs = np.concatenate(q_embs_list, axis=0) if q_embs_list else np.empty((0, dim), dtype=np.float32)
    query_encode_sec = round(time.time() - t0, 2)
    query_encode_qps = round(len(q_ids) / query_encode_sec, 1) if query_encode_sec > 0 else 0.0
    logger.info(f"query 인코딩 완료: {query_encode_sec}s  ({query_encode_qps} q/s)")

    logger.info(f"검색 (top-{_TOP_K})...")
    t0 = time.time()
    raw_results: list = []
    search_timings: list[dict] = []
    for batch_idx, start in enumerate(range(0, len(q_ids), _SEARCH_CHUNK)):
        batch_qids = q_ids[start: start + _SEARCH_CHUNK]
        batch_vecs = q_embs[start: start + _SEARCH_CHUNK]

        t_batch = time.time()
        batch_results = store.search(collection, batch_vecs, top_k=_TOP_K)
        batch_sec = time.time() - t_batch
        n_q = len(batch_qids)
        search_timings.append({
            "batch_index": batch_idx,
            "n_queries": n_q,
            "elapsed_sec": round(batch_sec, 4),
            "sec_per_query": round(batch_sec / n_q, 5) if n_q else 0.0,
            "query_ids": batch_qids,
        })
        logger.info(
            f"  [검색 배치 {batch_idx}] n={n_q}  elapsed={round(batch_sec, 3)}s  "
            f"sec/query={round(batch_sec / n_q, 5) if n_q else 0.0}"
        )
        raw_results.extend(batch_results)

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
        "chunk_timings": chunk_timings,
        "insert_timings": insert_timings,
        "query_timings": query_timings,
        "search_timings": search_timings,
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
    ap.add_argument("--force", action="store_true", help="이미 저장된 결과 파일이 있어도 다시 실행")
    ap.add_argument("--reindex", action="store_true",
                     help="Milvus에 인덱스가 이미 있어도 삭제 후 재색인 (인덱싱 속도 재측정용, --force도 같이 적용됨)")
    args = ap.parse_args()

    cfg = Config.from_env()
    out_path = os.path.join(args.out, f"{_safe_name(cfg.embedding.model)}_top100.json")

    if os.path.exists(out_path) and not args.force and not args.reindex:
        logger.info(f"[스킵] 이미 결과 존재: {out_path} (재실행하려면 --force 또는 --reindex)")
        return

    target = cfg.milvus.uri if cfg.backend == "milvus" else f"{cfg.starrocks.host}:{cfg.starrocks.port}"
    logger.info(f"모델={cfg.embedding.model}  backend={cfg.backend}({target})  data_root={args.data_root}  reindex={args.reindex}")
    docs, queries, qrels = load_from_dir(args.data_root)

    result = run_retrieval(cfg, docs, queries, qrels, out_path, force_reindex=args.reindex)

    print("\n===== RETRIEVAL RESULT SUMMARY =====")
    _detail_keys = ("results", "chunk_timings", "insert_timings", "query_timings", "search_timings")
    print(json.dumps(
        {k: v for k, v in result.items() if k not in _detail_keys},
        ensure_ascii=False, indent=2,
    ))
    print("===== END =====")


if __name__ == "__main__":
    main()
