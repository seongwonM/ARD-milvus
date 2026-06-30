"""API 기반 임베딩 벤치마크 러너.

사용법:
    python -m milvus_migration.bench.runner --data-root /data
    python -m milvus_migration.bench.runner --data-root /data --vector-mode sparse
    python -m milvus_migration.bench.runner --data-root /data --search-concurrency 1 4 8 16

환경변수 (pod.yaml에서 주입):
    MILVUS_URI, MILVUS_TOKEN
    EMBEDDING_API_ENDPOINT, EMBEDDING_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time

import numpy as np

from milvus_migration.bench.data_loader import load_from_dir
from milvus_migration.bench.evaluator import evaluate
from milvus_migration.config import Config
from milvus_migration.embedding import EmbeddingClient, SparseUnsupported, ColBERTUnsupported
from milvus_migration.milvus_store import MilvusStore


def _trunc_bytes(text: str, max_bytes: int = 2000) -> str:
    b = text.encode("utf-8")
    if len(b) <= max_bytes:
        return text
    return b[:max_bytes].decode("utf-8", errors="ignore")


def _safe_name(model_id: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", model_id.lower())[:200]


def _json_default(obj):
    try:
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return str(obj)


def run_bench(
    cfg:              Config,
    docs:             dict,
    queries:          dict,
    qrels:            dict,
    vector_mode:      str = "dense",
    with_colbert_rerank: bool = False,
    colbert_top_n:    int = 100,
    search_concurrency: list[int] | None = None,
    search_duration:    int = 30,
) -> dict:
    store  = MilvusStore(cfg.milvus.uri, cfg.milvus.token)
    client = EmbeddingClient(cfg.embedding)
    model_id   = cfg.embedding.model
    dim        = cfg.embedding.dim
    batch_size = cfg.embedding.batch_size
    collection = _safe_name(model_id) + (f"_{vector_mode}" if vector_mode != "dense" else "")

    print(f"\n{'='*64}")
    print(f"  모델: {model_id}  dim={dim}  mode={vector_mode}  colbert={with_colbert_rerank}")
    print(f"  컬렉션: {collection}  Milvus: {cfg.milvus.uri}")
    print(f"{'='*64}")

    # ── 인덱싱 ───────────────────────────────────────────────────────────────
    n_docs = len(docs)
    index_build_sec = index_docs_per_sec = None

    if store.has_collection(collection):
        n_pts = store.collection_size(collection)
        if n_pts == n_docs:
            print(f"  [스킵] 인덱스 존재 ({n_pts:,}건)", flush=True)
        else:
            print(f"  [재색인] 불완전 ({n_pts:,}/{n_docs:,})", flush=True)
            store.drop_collection(collection)
            index_build_sec, index_docs_per_sec = _build_index(
                store, client, collection, docs, dim, vector_mode, batch_size
            )
    else:
        index_build_sec, index_docs_per_sec = _build_index(
            store, client, collection, docs, dim, vector_mode, batch_size
        )

    # ── query 인코딩 ─────────────────────────────────────────────────────────
    q_ids   = list(queries.keys())
    q_texts = [queries[qid] for qid in q_ids]

    print(f"  query 인코딩 ({len(q_ids):,}건)...", flush=True)
    t0 = time.time()
    if vector_mode == "sparse":
        q_embs = client.encode_sparse(q_texts, batch_size=batch_size)
    else:
        q_embs = client.encode(q_texts, batch_size=batch_size)
    query_encode_sec = round(time.time() - t0, 2)
    query_encode_qps = round(len(q_ids) / query_encode_sec, 1)
    print(f"  query 인코딩: {query_encode_sec}s  ({query_encode_qps} q/s)", flush=True)

    # ── 검색 (+ ColBERT 리랭킹) ──────────────────────────────────────────────
    fetch_n = max(100, colbert_top_n) if with_colbert_rerank else 100
    print(f"  검색 (top-{fetch_n})...", flush=True)
    t0 = time.time()

    if with_colbert_rerank:
        candidates_per_q = store.search_with_text(collection, q_embs, top_k=fetch_n, vector_mode=vector_mode)
        search_sec = round(time.time() - t0, 2)
        search_qps = round(len(q_ids) / search_sec, 1)
        print(f"  검색: {search_sec}s  ({search_qps} q/s)", flush=True)

        print(f"  ColBERT 리랭킹...", flush=True)
        t_cb = time.time()
        try:
            cand_texts = [text for cands in candidates_per_q for _, _, text in cands]
            all_tok_vecs = client.encode_colbert(q_texts + cand_texts, batch_size=batch_size)
            q_tok_vecs = all_tok_vecs[:len(q_texts)]
            d_tok_vecs = all_tok_vecs[len(q_texts):]

            raw_results = []
            offset = 0
            for i, cands in enumerate(candidates_per_q):
                n = len(cands)
                reranked = MilvusStore.colbert_rerank(
                    cands, q_tok_vecs[i], d_tok_vecs[offset: offset + n], top_k=100
                )
                raw_results.append(reranked)
                offset += n
            colbert_sec = round(time.time() - t_cb, 2)
            print(f"  ColBERT 리랭킹: {colbert_sec}s", flush=True)
        except ColBERTUnsupported as exc:
            colbert_sec = None
            print(f"  ColBERT 미지원 → ANN 결과 그대로 사용: {exc}", flush=True)
            raw_results = [
                [(doc_id, score) for doc_id, score, _ in cands]
                for cands in candidates_per_q
            ]
    else:
        colbert_sec = None
        raw_results = store.search(collection, q_embs, top_k=100, vector_mode=vector_mode)
        search_sec = round(time.time() - t0, 2)
        search_qps = round(len(q_ids) / search_sec, 1)
        print(f"  검색: {search_sec}s  ({search_qps} q/s)", flush=True)

    run = {q_ids[i]: {doc_id: score for doc_id, score in hits}
           for i, hits in enumerate(raw_results)}

    # ── 평가 ─────────────────────────────────────────────────────────────────
    metrics = evaluate(run, qrels)
    print(
        f"\n  NDCG@10={metrics.get('ndcg_at_10')}  @20={metrics.get('ndcg_at_20')}  "
        f"@50={metrics.get('ndcg_at_50')}  @100={metrics.get('ndcg_at_100')}"
    )
    print(
        f"  MRR@10={metrics.get('mrr_at_10')}  "
        f"Recall@10={metrics.get('recall_at_10')}  @50={metrics.get('recall_at_50')}  @100={metrics.get('recall_at_100')}"
    )

    # ── Concurrent QPS ───────────────────────────────────────────────────────
    concurrent_results: dict = {}
    if search_concurrency and vector_mode == "dense":
        print(f"\n  [concurrent 벤치마크] duration={search_duration}s per level", flush=True)
        for c in search_concurrency:
            print(f"  concurrency={c} 시작...", flush=True)
            stats = _search_concurrent(store, collection, q_embs, top_k=100,
                                       concurrency=c, duration_sec=search_duration)
            concurrent_results[str(c)] = stats
            print(
                f"  concurrency={c}  QPS={stats['qps']}  "
                f"p50={stats['p50_ms']}ms  p95={stats['p95_ms']}ms  p99={stats['p99_ms']}ms  "
                f"(n={stats['total']})",
                flush=True,
            )

    return {
        "model":               model_id,
        "vector_mode":         vector_mode,
        "dim":                 dim,
        "api_endpoint":        cfg.embedding.endpoint,
        "batch_size":          batch_size,
        "timeout_sec":         cfg.embedding.timeout,
        "n_docs":              len(docs),
        "n_queries":           len(queries),
        "index_build_sec":     index_build_sec,
        "index_docs_per_sec":  index_docs_per_sec,
        "query_encode_sec":    query_encode_sec,
        "query_encode_qps":    query_encode_qps,
        "search_sec":          search_sec,
        "search_qps":          search_qps,
        "colbert_rerank":      with_colbert_rerank,
        "colbert_sec":         colbert_sec,
        "concurrent":          concurrent_results,
        **metrics,
    }


MILVUS_INSERT_BATCH = 5_000


def _build_index(
    store:       MilvusStore,
    client:      EmbeddingClient,
    collection:  str,
    docs:        dict,
    dim:         int,
    vector_mode: str,
    batch_size:  int,
) -> tuple[float, float]:
    store.create_collection(collection, dim=dim, vector_mode=vector_mode)
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

        if vector_mode == "sparse":
            embs = client.encode_sparse(texts, batch_size=batch_size)
            for doc_id, text, vec in zip(chunk_ids, texts, embs):
                pending.append({
                    "id": pid, "doc_id": doc_id,
                    "text": _trunc_bytes(text),
                    "vector": {int(k): float(v) for k, v in vec.items()},
                })
                pid += 1
        else:
            embs = client.encode(texts, batch_size=batch_size)
            for doc_id, text, vec in zip(chunk_ids, texts, embs):
                pending.append({
                    "id": pid, "doc_id": doc_id,
                    "text": _trunc_bytes(text),
                    "vector": vec.tolist(),
                })
                pid += 1

        if len(pending) >= MILVUS_INSERT_BATCH:
            store._insert_with_retry(collection, pending)
            pending.clear()
            elapsed = round(time.time() - t0, 1)
            dps_now = round(pid / elapsed, 1) if elapsed > 0 else 0
            print(f"  인덱싱: {pid:,}/{n_total:,}  elapsed={elapsed}s  ({dps_now} docs/s)", flush=True)

    if pending:
        store._insert_with_retry(collection, pending)
        print(f"  인덱싱: {pid:,}/{n_total:,}", flush=True)

    store.finalize(collection)
    elapsed = round(time.time() - t0, 2)
    dps     = round(n_total / elapsed, 1)
    print(f"  인덱싱 완료: {elapsed}s  ({dps} docs/s)", flush=True)
    return elapsed, dps


def _search_concurrent(
    store:       MilvusStore,
    collection:  str,
    vectors:     np.ndarray,
    top_k:       int,
    concurrency: int,
    duration_sec: int,
) -> dict:
    import threading, random
    from pymilvus import MilvusClient

    latencies: list[float] = []
    lock = threading.Lock()
    stop_event = threading.Event()
    n = len(vectors)

    def _worker():
        c = MilvusClient(uri=store._uri, timeout=60)
        if store._token:
            c = MilvusClient(uri=store._uri, token=store._token, timeout=60)
        while not stop_event.is_set():
            idx = random.randint(0, n - 1)
            t0 = time.perf_counter()
            c.search(
                collection_name=collection,
                data=[vectors[idx].tolist()],
                anns_field="vector",
                search_params={"metric_type": "COSINE", "params": {"ef": 100}},
                limit=top_k,
                output_fields=[],
            )
            lat = time.perf_counter() - t0
            with lock:
                latencies.append(lat)

    threads = [threading.Thread(target=_worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    time.sleep(duration_sec)
    stop_event.set()
    for t in threads:
        t.join(timeout=10)

    if not latencies:
        return {"qps": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "total": 0}

    latencies.sort()
    m = len(latencies)
    return {
        "qps":    round(m / duration_sec, 1),
        "p50_ms": round(latencies[int(m * 0.50)] * 1000, 1),
        "p95_ms": round(latencies[int(m * 0.95)] * 1000, 1),
        "p99_ms": round(latencies[int(m * 0.99)] * 1000, 1),
        "total":  m,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Milvus API 기반 임베딩 벤치마크")
    ap.add_argument("--data-root",          default=os.getenv("DATA_ROOT", "/data"))
    ap.add_argument("--vector-mode",        default="dense", choices=["dense", "sparse"])
    ap.add_argument("--colbert-rerank",     action="store_true", help="dense/sparse 100개 후보 → ColBERT 리랭킹")
    ap.add_argument("--colbert-top-n",      type=int, default=100)
    ap.add_argument("--search-concurrency", type=int, nargs="+", default=None, metavar="N")
    ap.add_argument("--search-duration",    type=int, default=30)
    ap.add_argument("--out",                default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = Config.from_env()

    bench_start = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*64}")
    print(f"  벤치마크 시작: {bench_start}")
    print(f"  모델:          {cfg.embedding.model}")
    print(f"  dim:           {cfg.embedding.dim}")
    print(f"  vector_mode:   {args.vector_mode}")
    print(f"  API endpoint:  {cfg.embedding.endpoint}")
    print(f"  batch_size:    {cfg.embedding.batch_size}")
    print(f"  timeout:       {cfg.embedding.timeout}s")
    print(f"  Milvus URI:    {cfg.milvus.uri}")
    print(f"  data_root:     {args.data_root}")
    print(f"  colbert:       {args.colbert_rerank}")
    print(f"  concurrency:   {args.search_concurrency}")
    print(f"{'='*64}\n")

    docs, queries, qrels = load_from_dir(args.data_root)

    result = run_bench(
        cfg, docs, queries, qrels,
        vector_mode=args.vector_mode,
        with_colbert_rerank=args.colbert_rerank,
        colbert_top_n=args.colbert_top_n,
        search_concurrency=args.search_concurrency,
        search_duration=args.search_duration,
    )

    colbert_tag = "_colbert" if args.colbert_rerank else ""
    model_tag = re.sub(r"[^a-z0-9_]", "_", cfg.embedding.model.lower())
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path  = os.path.join(args.out, f"{model_tag}_{args.vector_mode}{colbert_tag}_{timestamp}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"\n결과 저장: {out_path}")

    # ── stdout 출력 ───────────────────────────────────────────────────────────
    print("\n===== BENCH RESULT JSON =====")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    print("===== END =====")

    # ── MinIO 업로드 ──────────────────────────────────────────────────────────
    minio_endpoint   = os.getenv("MINIO_ENDPOINT", "minio:9000")
    minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    bucket           = os.getenv("MINIO_BUCKET", "bench-results")
    try:
        from minio import Minio
        mc = Minio(minio_endpoint, access_key=minio_access_key,
                   secret_key=minio_secret_key, secure=False)
        if not mc.bucket_exists(bucket):
            mc.make_bucket(bucket)
        object_name = os.path.basename(out_path)
        mc.fput_object(bucket, object_name, out_path, content_type="application/json")
        print(f"MinIO 업로드 완료: {minio_endpoint}/{bucket}/{object_name}")
    except Exception as e:
        print(f"MinIO 업로드 실패 (무시): {e}")


if __name__ == "__main__":
    main()
