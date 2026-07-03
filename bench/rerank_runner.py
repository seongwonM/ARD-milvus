"""저장된 retrieval 결과(top-100 후보) 기반 리랭킹 벤치마크.

retrieval_runner.py가 저장한 top100 JSON을 입력으로 받아, top_n(5/10/20/50/100)개의
후보만 잘라 리랭커에 넣고 소요 시간과 성능 지표(evaluator.py 동일 지표)를 측정합니다.
Milvus 재검색은 하지 않습니다 — 이미 저장된 후보 id만 사용합니다.

리랭커 종류 2가지:
    --rerank-model            전용 rerank API 사용 (예: bge-reranker-v2-m3).
                               endpoint/헤더는 embedding API와 동일, body는
                               {model, query, documents, top_n} 형식 (reranker.py 참고).
    --embedding-rerank-model  retrieval 때 쓰던 임베딩 API 그대로 재사용해
                               query/후보를 다시 인코딩 후 cosine 유사도로 재정렬
                               (예: Qwen3-Embedding-8B — "qwen3 reranker"는 별도 API가
                               아니라 이 방식).

사용법:
    python -m milvus_migration.bench.rerank_runner \
        --retrieval-result results/retrieval/hcp_llm_latest_top100.json \
        --rerank-model bge-reranker-v2-m3 \
        --embedding-rerank-model Qwen3-Embedding-8B \
        --data-root /data \
        --out results/rerank

환경변수:
    EMBEDDING_API_ENDPOINT, EMBEDDING_API_KEY  (rerank API/embedding 재인코딩 둘 다 동일 endpoint/헤더 사용)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np

from milvus_migration.bench.data_loader import load_from_dir
from milvus_migration.bench.evaluator import evaluate
from milvus_migration.bench.reranker import RerankClient
from milvus_migration.config import Config, EmbeddingConfig
from milvus_migration.embedding import EmbeddingClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_TOP_N = [5, 10, 20, 50, 100]

ScoreFn = Callable[[str, list[str], list[str]], dict[str, float]]


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", s.lower())[:200]


def _doc_text(docs: dict, doc_id: str) -> str:
    doc = docs[doc_id]
    return f"{doc.get('title', '')} {doc['chunk']}".strip()


def _cosine_scores(query_vec: np.ndarray, cand_vecs: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    d = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-9)
    return d @ q


def make_rerank_api_score_fn(client: RerankClient, model: str) -> ScoreFn:
    def score(q_text: str, cand_ids: list[str], cand_texts: list[str]) -> dict[str, float]:
        reranked = client.rerank(model, q_text, cand_texts, top_n=len(cand_ids))
        return {cand_ids[idx]: rel_score for idx, rel_score in reranked}
    return score


def _retry_snapshot(client: RerankClient | EmbeddingClient) -> tuple[int, float, int]:
    return client.retry_count, client.failed_time_sec, len(client.retry_success_at)


def make_embedding_score_fn(client: EmbeddingClient) -> ScoreFn:
    def score(q_text: str, cand_ids: list[str], cand_texts: list[str]) -> dict[str, float]:
        vecs = client.encode([q_text] + cand_texts)
        scores = _cosine_scores(vecs[0], vecs[1:])
        return {doc_id: float(s) for doc_id, s in zip(cand_ids, scores)}
    return score


def run_rerank(
    score_fn: ScoreFn,
    client: RerankClient | EmbeddingClient,
    rerank_model: str,
    rerank_method: str,
    retrieval_model: str,
    candidates: dict[str, list[str]],
    docs: dict,
    queries: dict,
    qrels: dict,
    top_n: int,
    request_interval_sec: float = 0.0,
    concurrency: int = 32,
) -> dict:
    """쿼리 하나씩 API 응답을 기다리는 대신, 워커 풀(큐)로 여러 요청을 동시에 흘려보낸다.

    QPS는 총 소요시간(n/wall_sec)이 아니라 API 응답 속도(latency)로부터 역산한다 —
    wall_sec 기반으로 계산하면 동시에 띄운 워커(큐) 개수 자체가 QPS에 그대로 반영돼서
    "API가 실제로 얼마나 빠른지"가 아니라 "워커를 몇 개 띄웠는지"를 재는 꼴이 되기 때문.
    Little's Law: 정상상태에서 처리량 = 동시성(concurrency) / 평균 응답시간(latency).
    """
    run: dict[str, dict[str, float]] = {}
    run_lock = threading.Lock()
    latencies: list[float] = []  # 스레드-로컬로 잰 순수 응답 시간(재시도/백오프 제외)
    latencies_lock = threading.Lock()
    cand_counts: list[int] = []
    stats_lock = threading.Lock()
    progress = {"done": 0}
    _LOG_SAMPLE = 5

    total = len(candidates)
    log_every = max(1, round(total * 0.025))  # 2.5% 단위(예: 12,000개면 300개 = concurrency 32 기준 약 10배치마다)

    retry_count_before, failed_time_before, retry_success_before = _retry_snapshot(client)

    def process_one(qid: str, cand_ids_full: list[str]) -> None:
        q_text = queries.get(qid)
        if q_text is None:
            logger.warning(
                f"  [스킵] qid={qid} 가 현재 queries_all.parquet에 없음 "
                f"(retrieval 시점과 data_root 불일치 가능)"
            )
            return
        cand_ids = cand_ids_full[:top_n]
        if not cand_ids:
            return
        cand_texts = [_doc_text(docs, d) for d in cand_ids]

        with stats_lock:
            cand_counts.append(len(cand_ids))
            sample_idx = len(cand_counts)
        if sample_idx <= _LOG_SAMPLE:
            logger.info(f"  [샘플 {sample_idx}/{_LOG_SAMPLE}] [{qid}] rerank API로 보내는 청크 수={len(cand_ids)}")

        result = score_fn(q_text, cand_ids, cand_texts)
        # client.last_latency_sec는 스레드-로컬 — 이 스레드가 방금 성공한 호출의 순수 응답
        # 시간(재시도/백오프 제외)만 담고 있어서, 다른 워커 스레드와 섞일 걱정 없이 바로 읽으면 됨.
        latency = client.last_latency_sec

        with run_lock:
            run[qid] = result
        with latencies_lock:
            latencies.append(latency)
        with stats_lock:
            progress["done"] += 1
            done = progress["done"]
        if done % log_every == 0 or done == total:
            logger.info(f"  진행: {done:,}/{total:,} ({round(100*done/total)}%)")

    interval_wait_sec = 0.0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for qid, cand_ids_full in candidates.items():
            futures.append(executor.submit(process_one, qid, cand_ids_full))
            if request_interval_sec > 0:
                time.sleep(request_interval_sec)  # 큐에 넣는(요청 제출) 속도 조절용
                interval_wait_sec += request_interval_sec
        for fut in as_completed(futures):
            fut.result()  # 워커 스레드에서 발생한 예외를 여기서 다시 던짐

    wall_sec = round(time.time() - t0, 2)

    if cand_counts:
        logger.info(
            f"  쿼리당 청크 수: min={min(cand_counts)}  max={max(cand_counts)}  "
            f"avg={round(sum(cand_counts) / len(cand_counts), 1)}  (top_n={top_n} 상한)"
        )

    retry_count_after, failed_time_after, retry_success_after = _retry_snapshot(client)
    retry_count = retry_count_after - retry_count_before
    retry_failed_sec = round(failed_time_after - failed_time_before, 2)
    retry_calls = retry_success_after - retry_success_before  # 재시도 끝에 결국 성공한 호출 수

    n = len(run)
    latencies.sort()
    mean_latency_sec = statistics.mean(latencies) if latencies else 0.0
    qps = round(concurrency / mean_latency_sec, 2) if mean_latency_sec > 0 else 0.0
    p50_ms = round(latencies[int(len(latencies) * 0.50)] * 1000, 1) if latencies else 0.0
    p95_ms = round(latencies[int(len(latencies) * 0.95)] * 1000, 1) if latencies else 0.0

    metrics = evaluate(run, qrels)
    logger.info(
        f"  retrieval={retrieval_model}  rerank={rerank_model}({rerank_method})  top_n={top_n}  "
        f"concurrency={concurrency}  wall={wall_sec}s  qps={qps}(=concurrency/평균응답시간)  "
        f"평균응답={round(mean_latency_sec*1000, 1)}ms  p50={p50_ms}ms  p95={p95_ms}ms"
    )
    if retry_count:
        logger.info(
            f"  재시도: 실패한 시도={retry_count}회  재시도 끝에 성공한 호출={retry_calls}건  "
            f"실패(+백오프)에 쓴 시간={retry_failed_sec}s (latency 집계에서 제외됨)"
        )
    if interval_wait_sec:
        logger.info(f"  요청 제출 간 대기: {request_interval_sec}s x {n}건 = {round(interval_wait_sec, 1)}s")
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
        "rerank_method": rerank_method,
        "top_n": top_n,
        "n_queries": n,
        "concurrency": concurrency,
        "wall_sec": wall_sec,
        "rerank_qps": qps,
        "mean_latency_ms": round(mean_latency_sec * 1000, 1),
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "retry_count": retry_count,
        "retry_calls": retry_calls,
        "retry_failed_sec": retry_failed_sec,
        "interval_wait_sec": round(interval_wait_sec, 1),
        **metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="저장된 retrieval 결과 기반 리랭킹 벤치마크")
    ap.add_argument("--retrieval-result", required=True, nargs="+",
                     help="retrieval_runner.py가 생성한 top100 JSON 경로(들)")
    ap.add_argument("--rerank-model", nargs="+", default=[],
                     help="전용 rerank API를 쓰는 리랭커 모델명(들) — 예: bge-reranker-v2-m3")
    ap.add_argument("--embedding-rerank-model", nargs="+", default=[],
                     help="임베딩 API 재사용(cosine 재정렬) 리랭커 모델명(들) — 예: Qwen3-Embedding-8B")
    ap.add_argument("--top-n", type=int, nargs="+", default=_DEFAULT_TOP_N)
    ap.add_argument("--data-root", default=os.getenv("DATA_ROOT", "/data"))
    ap.add_argument("--out", default="results/rerank")
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("RERANK_CONCURRENCY", "32")),
                     help="동시에 흘려보낼 rerank 요청 수(워커 풀 크기) — 순차 호출 대신 큐로 흘려서 QPS 측정")
    args = ap.parse_args()

    if not args.rerank_model and not args.embedding_rerank_model:
        ap.error("--rerank-model 또는 --embedding-rerank-model 중 하나는 지정해야 합니다")

    startup_delay = float(os.environ.get("STARTUP_DELAY_SEC", "0"))
    if startup_delay > 0:
        logger.info(
            f"시작 지연 {startup_delay}s 대기 중 (동시에 뜨는 다른 rerank Job들과 API 요청 "
            f"엇박 처리용 — 총 소요시간에는 미포함)..."
        )
        time.sleep(startup_delay)

    request_interval_sec = float(os.environ.get("REQUEST_INTERVAL_SEC", "0"))
    if request_interval_sec > 0:
        logger.info(f"요청 제출 간 대기 {request_interval_sec}s 적용 (큐에 새 요청 넣는 속도 조절)")
    logger.info(f"동시성(concurrency)={args.concurrency} — 응답을 기다리지 않고 큐로 흘려보냄")

    main_t0 = time.time()

    cfg = Config.from_env()
    rerank_client = RerankClient(cfg.embedding)
    docs, queries, qrels = load_from_dir(args.data_root)
    os.makedirs(args.out, exist_ok=True)

    # (표시용 모델명, 방식, score 함수, client) 조합 목록 — client는 재시도 통계 집계용
    rerankers: list[tuple[str, str, ScoreFn, RerankClient | EmbeddingClient]] = []
    for model in args.rerank_model:
        rerankers.append((model, "rerank_api", make_rerank_api_score_fn(rerank_client, model), rerank_client))
    for model in args.embedding_rerank_model:
        emb_cfg = EmbeddingConfig(
            endpoint=cfg.embedding.endpoint,
            api_key=cfg.embedding.api_key,
            model=model,
            batch_size=cfg.embedding.batch_size,
            timeout=cfg.embedding.timeout,
        )
        emb_client = EmbeddingClient(emb_cfg)
        rerankers.append((model, "embedding_cosine", make_embedding_score_fn(emb_client), emb_client))

    all_results: list[dict] = []
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    for retrieval_path in args.retrieval_result:
        with open(retrieval_path, encoding="utf-8") as f:
            retrieval_data = json.load(f)
        retrieval_model = retrieval_data["model"]
        candidates = retrieval_data["results"]

        for rerank_model, rerank_method, score_fn, client in rerankers:
            for n in args.top_n:
                logger.info(f"\n{'='*64}")
                logger.info(f"retrieval={retrieval_model}  rerank={rerank_model}({rerank_method})  top_n={n}")
                logger.info(f"{'='*64}")

                result = run_rerank(
                    score_fn, client, rerank_model, rerank_method, retrieval_model,
                    candidates, docs, queries, qrels, n, request_interval_sec,
                    args.concurrency,
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
    total_elapsed = round(time.time() - main_t0, 1)
    logger.info(f"\n요약 저장 ({len(all_results)}개 조합, 총 소요={total_elapsed}s): {summary_path}")

    print("\n===== RERANK SUMMARY JSON =====")
    print(json.dumps(all_results, ensure_ascii=False, indent=2))
    print("===== END =====")


if __name__ == "__main__":
    main()
