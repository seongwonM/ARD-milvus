# Milvus Migration

Cloud Platform Embedding API + Milvus VectorDB 기반 문서 인덱싱 및 검색 패키지.
`bench/`에는 별도로 실행 가능한 2단계 벤치마크(retrieval → reranking)가 포함되어 있습니다.

## 설치

```bash
pip install -r requirements.txt
```

## 환경변수 설정

`.env.example`을 복사해서 `.env`로 만들고 값을 채웁니다.

```bash
cp .env.example .env
```

| 변수명 | 필수 | 기본값 | 설명 |
|---|---|---|---|
| `EMBEDDING_API_ENDPOINT` | ✅ | - | Cloud Platform 임베딩 API URL |
| `EMBEDDING_API_KEY` | - | `""` | API 인증 키 |
| `EMBEDDING_MODEL` | - | `text-embedding-3-small` | 모델명 |
| `EMBEDDING_BATCH_SIZE` | - | `64` | API 호출당 최대 텍스트 수 |
| `EMBEDDING_TIMEOUT` | - | `60` | API 타임아웃 (초) |
| `MILVUS_URI` | - | `http://localhost:19530` | Milvus 서버 주소 |
| `MILVUS_TOKEN` | - | `""` | Milvus 인증 토큰 |
| `MILVUS_COLLECTION` | - | `documents` | 기본 컬렉션 이름 |

임베딩 차원(dim)은 설정하지 않아도 됩니다 — 첫 API 응답으로 자동 감지합니다.

## 사용법

### Python 코드

```python
from milvus_migration import Pipeline

pipe = Pipeline.from_env()

# 인덱싱
docs = [
    {"id": "doc-1", "text": "Milvus는 오픈소스 벡터 데이터베이스입니다."},
    {"id": "doc-2", "title": "PyMilvus", "text": "Milvus Python SDK입니다."},
]
pipe.index(docs, recreate=True)

# 검색
results = pipe.search("벡터 데이터베이스 추천", top_k=5)
for doc_id, score in results:
    print(f"{doc_id}: {score:.4f}")

# 복수 쿼리 한 번에
results = pipe.search(["쿼리1", "쿼리2"], top_k=3)
```

#### `index()` 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `docs` | `list[dict]` | - | `{"id": ..., "text": ...}` 리스트 |
| `collection` | `str \| None` | env 값 | 컬렉션 이름 |
| `recreate` | `bool` | `False` | 기존 컬렉션 삭제 후 재생성 |
| `batch_size` | `int \| None` | env 값 | 임베딩 배치 크기 |

#### `search()` 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `query` | `str \| list[str]` | - | 검색 쿼리 |
| `top_k` | `int` | `10` | 반환할 최대 결과 수 |
| `collection` | `str \| None` | env 값 | 컬렉션 이름 |

### CLI

**인덱싱** — JSON 파일을 읽어 Milvus에 저장

```bash
python -m milvus_migration.main index --file docs.json
python -m milvus_migration.main index --file docs.json --recreate          # 기존 컬렉션 초기화
```

JSON 파일 형식:

```json
[
    {"id": "doc-1", "text": "내용"},
    {"id": "doc-2", "title": "제목", "text": "내용"}
]
```

**검색**

```bash
python -m milvus_migration.main search --query "벡터 데이터베이스 추천"
python -m milvus_migration.main search --query "검색어" --top-k 20
```

**컬렉션 정보 확인**

```bash
python -m milvus_migration.main info
```

## Embedding API 형식 변경

기본값은 OpenAI-compatible 형식입니다. Cloud Platform의 응답 형식이 다를 경우 `embedding.py`의 두 메서드를 수정합니다.

```python
# embedding.py

def _build_payload(self, texts: list[str]) -> dict:
    # 요청 바디 커스터마이징
    return {"input": texts, "model": self._config.model}

def _parse_response(self, data: dict) -> list[list[float]]:
    # 응답 파싱 커스터마이징
    return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
```

## 벤치마크 (`bench/`)

Retrieval(임베딩 모델별 top-100 후보 검색)과 Reranking(리랭커 성능 측정)을 완전히 분리했습니다.
Retrieval 결과는 한 번 계산되면 재사용되며, Reranking은 그 결과를 그대로 읽어서만 동작하므로
Milvus나 임베딩 API를 다시 호출하지 않습니다.

### 1단계 — Retrieval

모델별로 문서를 인덱싱하고 쿼리마다 top-100 후보 doc-id를 검색해 로그에 출력하고 JSON으로 저장합니다.

```bash
python -m milvus_migration.bench.retrieval_runner --data-root /data --out results/retrieval
```

- 모델 선택은 `EMBEDDING_MODEL` 환경변수로 합니다 (모델마다 스크립트를 따로 실행).
- 결과 파일이 이미 있으면 재검색하지 않고 스킵합니다 (`--force`로 강제 재실행).
- raw dense 검색(리랭킹 전) 자체의 NDCG/MRR/Recall/MAP과 query_encode_qps/search_qps도 같이 계산해 저장합니다.
- 결과 형식: `{"model": ..., "ndcg_at_10": ..., "search_qps": ..., "results": {"query_id": ["doc_id", ...100개], ...}}`

### 2단계 — Reranking

1단계에서 저장된 top-100 후보에서 top_n(5/10/20/50/100)개만 잘라 리랭커에 넣고
소요 시간과 성능 지표(NDCG/MRR/Recall/MAP — `evaluator.py`)를 함께 측정합니다.

리랭커는 두 가지 방식을 지원합니다:

| 옵션 | 방식 | 예시 모델 |
|---|---|---|
| `--rerank-model` | 전용 rerank API (`{model, query, documents, top_n}` → `{results: [{index, relevance_score}]}`) | `bge-reranker-v2-m3` |
| `--embedding-rerank-model` | retrieval 때 쓰던 임베딩 API를 그대로 재사용해 query/후보를 재인코딩 후 cosine 유사도로 재정렬 | `Qwen3-Embedding-8B` |

```bash
python -m milvus_migration.bench.rerank_runner \
    --retrieval-result results/retrieval/bge_m3_top100.json \
    --rerank-model bge-reranker-v2-m3 \
    --embedding-rerank-model Qwen3-Embedding-8B \
    --data-root /data \
    --out results/rerank
```

- `--retrieval-result`, `--rerank-model`, `--embedding-rerank-model`, `--top-n`은 여러 개를 넘기면 모든 조합을 순회합니다.
- 두 방식 모두 endpoint/헤더는 Embedding API와 동일합니다 (`RerankClient`/`EmbeddingClient`, `bench/reranker.py`).

### k8s

- `k8s/milvus.yaml` — Milvus standalone (embedded etcd) + PVC
- `k8s/minio.yaml` — Milvus의 blob storage 백엔드. `milvus.yaml`이 `minio:9000` / `minioadmin`/`minioadmin`으로
  참조하므로 서비스 이름·계정을 바꾸면 `milvus.yaml`도 같이 맞춰야 합니다.
- `k8s/bench-jobs.yaml` — retrieval 3개 모델(HCP-LLM-Latest / bge-m3 / Qwen3-Embedding-8B) Job + 결과 공유용 PVC(`bench-results`)
- `k8s/rerank-job.yaml` — retrieval Job들이 끝난 뒤 적용하는 reranking Job
- `k8s/pod.yaml` — 수동 디버그용 Pod (`kubectl exec`로 들어가 직접 실행)
- `k8s/fetch-results.ps1` — Job 완료를 기다렸다가 결과 JSON/로그를 로컬 `results/`로 자동 복사
  (`./k8s/fetch-results.ps1 -Stage retrieval`, `-Stage rerank`, `-Stage all`)

> retrieval 결과 PVC(`bench-results`)는 3개 Job이 동시에 마운트하므로 `ReadWriteMany`(nfs.csi.k8s.io
> 기반 StorageClass)로 만들어져 있습니다. `ReadWriteOnce`로 바꾸면 동시 실행 시 volume attach 대기로 멈춥니다.
> 클러스터마다 StorageClass 이름이 다르니 `k8s/bench-jobs.yaml`의 `storageClassName`을 환경에 맞게 바꾸세요.

## 파일 구조

```
milvus_migration/
├── config.py             # 환경변수 설정
├── embedding.py          # Cloud Platform Embedding API 클라이언트
├── milvus_store.py       # Milvus CRUD
├── pipeline.py           # 인덱싱 + 검색 파이프라인
├── main.py                # CLI 진입점
├── bench/
│   ├── data_loader.py     # parquet 데이터 로더
│   ├── evaluator.py       # NDCG / MRR / Recall / MAP
│   ├── reranker.py        # Rerank API 클라이언트
│   ├── retrieval_runner.py  # 1단계: 임베딩 모델별 top-100 검색 및 저장
│   └── rerank_runner.py     # 2단계: 저장된 후보 기반 리랭킹 벤치마크
├── k8s/
│   ├── milvus.yaml        # Milvus standalone
│   ├── minio.yaml         # MinIO (Milvus blob storage 백엔드)
│   ├── bench-jobs.yaml    # retrieval Job × 3 + PVC(ReadWriteMany)
│   ├── rerank-job.yaml    # reranking Job
│   ├── pod.yaml           # 디버그용 Pod
│   └── fetch-results.ps1  # 결과 로컬 자동 복사 스크립트
├── requirements.txt
└── .env.example
```
