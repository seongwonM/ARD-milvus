"""저장된 retrieval 결과(top-100 후보) 기반 리랭킹 벤치마크.

retrieval_runner.py가 저장한 top100 JSON을 입력으로 받아, top_n(5/10/20/50/100)개의
후보만 잘라 리랭커에 넣고 소요 시간과 성능 지표(evaluator.py 동일 지표)를 측정합니다.
Milvus 재검색이나 재임베딩은 하지 않습니다 — 이미 저장된 후보 id만 사용합니다.

사용법:
    python -m milvus_migration.bench.rerank_runner \
        --retrieval-result results/retrieval/hcp_llm_latest_top100.json \
        --rerank-model qwen3-reranker \
        --data-root /data \
        --out results/rerank

환경변수:
    EMBEDDING_API_ENDPOINT, EMBEDDING_API_KEY  (rerank API도 동일 endpoint/헤더 사용)
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
from milvus_migration.bench.reranker import RerankClient
from milvus_migration.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_TOP_N = [5, 10, 20, 50, 100]


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", s.lower())[:200]


def _doc_text(docs: dict, doc_id: str) -> str:
    doc = docs[doc_id]
    return f"{doc.get('title', '')} {doc['chunk']}".strip()


def run_rerank(
    client: RerankClient,
    rerank_model: str,
    retrieval_model: str,
    candidates: dict[str, list[str]],
    docs: dict,
    queries: dict,
    qrels: dict,
    top_n: int,
) -> dict:
    run: dict[str, dict[str, float]] = {}
    latencies: list[float] = []

    t0 = time.time()
    for qid, q_text in queries.items():
        cand_ids = candidates.get(qid, [])[:top_n]
        if not cand_ids:
            continue
        cand_texts = [_doc_text(docs, d) for d in cand_ids]

        t_q = time.time()
        reranked = client.rerank(rerank_model, q_text, cand_texts, top_n=len(cand_ids))
        latencies.append(time.time() - t_q)

        run[qid] = {cand_ids[idx]: score for idx, score in reranked}

    elapsed = round(time.time() - t0, 2)
    n = len(run)
    qps = round(n / elapsed, 2) if elapsed > 0 else 0.0
    latencies.sort()
    p50_ms = round(latencies[int(len(latencies) * 0.50)] * 1000, 1) if latencies else 0.0
    p95_ms = round(latencies[int(len(latencies) * 0.95)] * 1000, 1) if latencies else 0.0

    metrics = evaluate(run, qrels)
    logger.info(
        f"  retrieval={retrieval_model}  rerank={rerank_model}  top_n={top_n}  "
        f"elapsed={elapsed}s  qps={qps}  p50={p50_ms}ms  p95={p95_ms}ms"
    )
    logger.info(
        f"  NDCG@10={metrics.get('ndcg_at_10')}  @20={metrics.get('ndcg_at_20')}  "
        f"@50={metrics.get('ndcg_at_50')}  @100={metrics.get('ndcg_at_100')}"
    )
    logger.info(
        f"  MRR@10={metrics.get('mrr_at_10')}  "
        f"Recall@10={metrics.get('recall_at_10')}  @50={metrics.get('recall_at_50')}  @100={metrics.get('recall_at_100')}"
    )

    return {
        "retrieval_model": retrieval_model,
        "rerank_model": rerank_model,
        "top_n": top_n,
        "n_queries": n,
        "rerank_sec": elapsed,
        "rerank_qps": qps,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        **metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="저장된 retrieval 결과 기반 리랭킹 벤치마크")
    ap.add_argument("--retrieval-result", required=True, nargs="+",
                     help="retrieval_runner.py가 생성한 top100 JSON 경로(들)")
    ap.add_argument("--rerank-model", required=True, nargs="+", help="리랭커 모델명(들)")
    ap.add_argument("--top-n", type=int, nargs="+", default=_DEFAULT_TOP_N)
    ap.add_argument("--data-root", default=os.getenv("DATA_ROOT", "/data"))
    ap.add_argument("--out", default="results/rerank")
    args = ap.parse_args()

    cfg = Config.from_env()
    client = RerankClient(cfg.embedding)
    docs, queries, qrels = load_from_dir(args.data_root)
    os.makedirs(args.out, exist_ok=True)

    all_results: list[dict] = []
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    for retrieval_path in args.retrieval_result:
        with open(retrieval_path, encoding="utf-8") as f:
            retrieval_data = json.load(f)
        retrieval_model = retrieval_data["model"]
        candidates = retrieval_data["results"]

        for rerank_model in args.rerank_model:
            for n in args.top_n:
                logger.info(f"\n{'='*64}")
                logger.info(f"retrieval={retrieval_model}  rerank={rerank_model}  top_n={n}")
                logger.info(f"{'='*64}")

                result = run_rerank(
                    client, rerank_model, retrieval_model, candidates, docs, queries, qrels, n,
                )
                all_results.append(result)

                out_name = (
                    f"{_safe_name(retrieval_model)}__{_safe_name(rerank_model)}"
                    f"__top{n}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                )
                out_path = os.path.join(args.out, out_name)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                logger.info(f"저장: {out_path}")

    summary_path = os.path.join(args.out, f"summary_{run_timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n요약 저장 ({len(all_results)}개 조합): {summary_path}")

    print("\n===== RERANK SUMMARY JSON =====")
    print(json.dumps(all_results, ensure_ascii=False, indent=2))
    print("===== END =====")


if __name__ == "__main__":
    main()
