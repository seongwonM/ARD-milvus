"""6가지 조합 전체 실행.

    dense                (ColBERT 없음)
    dense  + ColBERT     리랭킹
    sparse               (ColBERT 없음)  — API 미지원 시 SKIP
    sparse + ColBERT     리랭킹          — API 미지원 시 SKIP
    8B dense             (ColBERT 없음)  — EMBEDDING_MODEL_8B 미설정 시 SKIP
    8B dense + ColBERT   리랭킹          — EMBEDDING_MODEL_8B 미설정 시 SKIP

사용법:
    python -m milvus_migration.bench.run_all --data-root /data [--out results]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_COMBOS = [
    # (label,                vector_mode, is_8b, with_colbert)
    ("dense",                "dense",  False, False),
    ("dense+ColBERT",        "dense",  False, True),
    ("sparse",               "sparse", False, False),
    ("sparse+ColBERT",       "sparse", False, True),
    ("8B dense",             "dense",  True,  False),
    ("8B dense+ColBERT",     "dense",  True,  True),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="6-combo 벤치마크")
    ap.add_argument("--data-root",         default=os.getenv("DATA_ROOT", "/data"))
    ap.add_argument("--colbert-top-n",     type=int, default=100)
    ap.add_argument("--search-concurrency", type=int, nargs="+", default=None)
    ap.add_argument("--search-duration",   type=int, default=30)
    ap.add_argument("--out",               default="results")
    args = ap.parse_args()

    from milvus_migration.bench.data_loader import load_from_dir
    from milvus_migration.bench.runner import run_bench, _json_default
    from milvus_migration.config import Config, EmbeddingConfig
    from milvus_migration.embedding import SparseUnsupported

    os.makedirs(args.out, exist_ok=True)
    cfg = Config.from_env()

    # 8B config (가능하면)
    if cfg.embedding.model_8b:
        emb_8b = EmbeddingConfig(
            endpoint=cfg.embedding.endpoint,
            api_key=cfg.embedding.api_key,
            model=cfg.embedding.model_8b,
            dim=cfg.embedding.dim_8b,
            batch_size=cfg.embedding.batch_size,
            timeout=cfg.embedding.timeout,
        )
        cfg_8b = Config(milvus=cfg.milvus, embedding=emb_8b)
    else:
        cfg_8b = None

    print(f"\n데이터 로드: {args.data_root}")
    docs, queries, qrels = load_from_dir(args.data_root)
    print(f"  docs={len(docs):,}  queries={len(queries):,}")

    summary: list[dict] = []
    rows: list[tuple[str, str, str]] = []

    for label, mode, is_8b, colbert in _COMBOS:
        print(f"\n{'─'*64}")
        print(f"  [{label}]")

        if is_8b and cfg_8b is None:
            print("  → SKIP: EMBEDDING_MODEL_8B 미설정")
            rows.append((label, "SKIP", "EMBEDDING_MODEL_8B 미설정"))
            continue

        run_cfg = cfg_8b if is_8b else cfg

        try:
            result = run_bench(
                run_cfg, docs, queries, qrels,
                vector_mode=mode,
                with_colbert_rerank=colbert,
                colbert_top_n=args.colbert_top_n,
                search_concurrency=args.search_concurrency,
                search_duration=args.search_duration,
            )
        except SparseUnsupported as exc:
            print(f"  → SKIP: sparse 미지원 — {exc}")
            rows.append((label, "SKIP", f"sparse 미지원"))
            continue
        except Exception as exc:
            print(f"  → ERROR: {exc}")
            rows.append((label, "ERROR", str(exc)))
            continue

        model_tag = re.sub(r"[^a-z0-9_]", "_", run_cfg.embedding.model.lower())
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{model_tag}_{label.replace('+', '_').replace(' ', '_')}_{ts}.json"
        with open(os.path.join(args.out, fname), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=_json_default)

        ndcg = result.get("ndcg_at_10", "?")
        rows.append((label, "OK", f"NDCG@10={ndcg}"))
        summary.append(result)

    # ── 요약 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  {'조합':<22}  {'결과':<6}  메모")
    print(f"  {'-'*60}")
    for label, status, info in rows:
        print(f"  {label:<22}  {status:<6}  {info}")
    print(f"{'='*64}")

    if summary:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(args.out, f"summary_all_{ts}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
        print(f"\n전체 결과: {out}")


if __name__ == "__main__":
    main()
