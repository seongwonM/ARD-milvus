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
from milvus_migration.core.config import Config, EmbeddingConfig
from milvus_migration.core.embedding import EmbeddingClient

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


def _unique_doc_ids(candidates: dict[str, list[str]], top_n: int) -> list[str]:
    """모든 쿼리의 top_n 후보를 합친 유니크 문서 id 목록.

    코퍼스는 고정 크기라 같은 문서가 여러 쿼리의 top_n에 반복 등장한다 — 쿼리마다 후보를
    다시 인코딩하면 쿼리 수 x top_n번 호출하는 꼴이 되지만, 유니크 집합만 한 번씩 인코딩하면
    코퍼스 크기만큼만 호출하면 된다.
    """
    seen: set[str] = set()
    order: list[str] = []
    for cand_ids_full in candidates.values():
        for d in cand_ids_full[:top_n]:
            if d not in seen:
                seen.add(d)
                order.append(d)
    return order


def make_precomputed_embedding_score_fn(
    client: EmbeddingClient, candidates: dict[str, list[str]], docs: dict, top_n: int, concurrency: int,
) -> ScoreFn:
    """문서 벡터만 미리 한 번씩 배치 인코딩해서 캐싱한다 (쿼리는 캐싱하지 않음).

    실제 운영에서도 문서는 인덱싱 시점에 한 번만 임베딩해서 벡터 저장소에 넣어두고,
    쿼리는 요청이 올 때마다 그때그때 임베딩한다 — 문서를 쿼리별로 다시 인코딩하는 건
    애초에 운영 구조와 안 맞는 낭비였을 뿐이다. 반대로 쿼리까지 미리 다 모아 배치
    인코딩해버리면 "요청 1건당 API 응답 시간"이라는 벤치마크 latency/QPS의 의미 자체가
    사라지므로, 쿼리는 기존처럼 score() 호출 시점(=쿼리별 워커)에 API로 임베딩한다.

    문서 배치는 concurrency로 동시에 흘려보낸다 — 한때 top_n별 pod 5개를 동시에 띄우면서
    "pod 안에서까지 동시에 쏘면 서버 부하가 이중으로 겹친다"는 이유로 순차 호출로 바꿨었는데,
    이제 pod는 k8s/scripts/run-rerank-qwen3-sequential.ps1로 한 번에 하나씩만 실행되므로(2026-07-06)
    그 전제가 사라졌다. pod 하나만 서버를 쓰는 상황에서 문서 배치까지 순차로 가는 건
    불필요한 낭비라 다시 동시 처리로 되돌린다.
    """
    unique_doc_ids = _unique_doc_ids(candidates, top_n)
    doc_texts = [_doc_text(docs, d) for d in unique_doc_ids]
    n_docs = len(doc_texts)
    bs = client.batch_size
    doc_id_chunks = [unique_doc_ids[i: i + bs] for i in range(0, n_docs, bs)]
    text_chunks = [doc_texts[i: i + bs] for i in range(0, n_docs, bs)]
    n_chunks = len(text_chunks)
    # concurrency는 쿼리 처리(수천~수만 건)용 큐 크기를 그대로 재사용한 값이라, top_n이
    # 작아 문서 배치 수가 그보다 적으면 어차피 배치 수만큼만 동시에 나간다 — 헷갈리지 않게
    # 실제 쓰는 값을 따로 계산해서 로그에 남김.
    doc_concurrency = min(concurrency, n_chunks) if n_chunks else 1
    logger.info(
        f"  [사전 인코딩 - 문서만] 유니크 후보 문서 {n_docs:,}개, {n_chunks:,}배치(batch_size={bs}), "
        f"동시 요청={doc_concurrency}(설정 concurrency={concurrency}) — pod 1개만 서버를 쓰므로 동시 처리"
    )

    doc_vec_map: dict[str, np.ndarray] = {}
    doc_vec_lock = threading.Lock()
    progress = {"done": 0}
    progress_lock = threading.Lock()
    log_every = max(1, round(n_chunks * 0.05))
    t_pre = time.time()

    def encode_chunk(idx: int) -> None:
        vecs = client.encode(text_chunks[idx], batch_size=len(text_chunks[idx]))
        with doc_vec_lock:
            doc_vec_map.update(zip(doc_id_chunks[idx], vecs))
        with progress_lock:
            progress["done"] += 1
            done = progress["done"]
        if done % log_every == 0 or done == n_chunks:
            logger.info(
                f"    문서 인코딩 진행: 배치 {done:,}/{n_chunks:,} "
                f"(문서 약 {min(done * bs, n_docs):,}/{n_docs:,})  경과={round(time.time() - t_pre, 1)}s"
            )

    with ThreadPoolExecutor(max_workers=doc_concurrency) as executor:
        futures = [executor.submit(encode_chunk, i) for i in range(n_chunks)]
        for fut in as_completed(futures):
            fut.result()

    def score(q_text: str, cand_ids: list[str], cand_texts: list[str]) -> dict[str, float]:
        q_vec = client.encode([q_text])[0]
        cand_vecs = np.stack([doc_vec_map[d] for d in cand_ids])
        scores = _cosine_scores(q_vec, cand_vecs)
        return {doc_id: float(s) for doc_id, s in zip(cand_ids, scores)}
    return score


def run_rerank(
    score_fn: ScoreFn | None,
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
    concurrency: int = 16,
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

    precompute_sec = 0.0
    if rerank_method == "embedding_cosine":
        assert isinstance(client, EmbeddingClient)
        t_pre = time.time()
        score_fn = make_precomputed_embedding_score_fn(client, candidates, docs, top_n, concurrency)
        precompute_sec = round(time.time() - t_pre, 2)
        logger.info(
            f"  문서 사전 인코딩 완료: {precompute_sec}s — 쿼리는 기존처럼 워커마다 API로 임베딩 "
            f"(latency/QPS는 이 쿼리 임베딩 호출 기준)"
        )
    assert score_fn is not None

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
    if rerank_method == "embedding_cosine" and precompute_sec:
        logger.info(
            f"  (문서 사전 인코딩 {precompute_sec}s는 실제 서비스의 인덱싱 단계에 해당 — 위 wall/평균응답/"
            f"p50/p95/qps는 쿼리 임베딩 API 호출 기준이라 그대로 유효함)"
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
        "precompute_sec": precompute_sec,
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
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("RERANK_CONCURRENCY", "16")),
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

    # (표시용 모델명, 방식, score 함수, client) 조합 목록 — client는 재시도 통계 집계용.
    # embedding_cosine 방식은 score_fn을 여기서 만들지 않고 None으로 둔다 — 문서/쿼리 벡터를
    # candidates/top_n 기준으로 사전 인코딩해서 캐싱해야 하는데, candidates와 top_n은 이 시점엔
    # 아직 모르고 run_rerank() 안에서만 알 수 있기 때문 (make_precomputed_embedding_score_fn 참고).
    rerankers: list[tuple[str, str, ScoreFn | None, RerankClient | EmbeddingClient]] = []
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
        rerankers.append((model, "embedding_cosine", None, emb_client))

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
