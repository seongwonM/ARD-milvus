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
| `VECTOR_BACKEND` | - | `milvus` | 벡터 백엔드 선택 (`milvus` \| `starrocks`) |
| `STARROCKS_HOST` | `starrocks` 사용 시 | `localhost` | StarRocks FE 주소 (MySQL 프로토콜) |
| `STARROCKS_PORT` | - | `9030` | StarRocks FE 쿼리 포트 |
| `STARROCKS_USER` / `STARROCKS_PASSWORD` | - | `root` / `""` | StarRocks 접속 계정 |
| `STARROCKS_DATABASE` | - | `milvus_migration` | StarRocks 데이터베이스 이름 |

임베딩 차원(dim)은 설정하지 않아도 됩니다 — 첫 API 응답으로 자동 감지합니다.

`VECTOR_BACKEND=starrocks`로 설정하면 `Pipeline`/`retrieval_runner`/`loadtest`가 모두
Milvus 대신 StarRocks(벡터 인덱스 HNSW, `core/starrocks_store.py`)를 사용합니다 — 코드/CLI
사용법은 완전히 동일하고 백엔드만 바뀝니다.

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
python -m milvus_migration.core.main index --file docs.json
python -m milvus_migration.core.main index --file docs.json --recreate          # 기존 컬렉션 초기화
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
python -m milvus_migration.core.main search --query "벡터 데이터베이스 추천"
python -m milvus_migration.core.main search --query "검색어" --top-k 20
```

**컬렉션 정보 확인**

```bash
python -m milvus_migration.core.main info
```

## Embedding API 형식 변경

기본값은 OpenAI-compatible 형식입니다. Cloud Platform의 응답 형식이 다를 경우 `core/embedding.py`의 두 메서드를 수정합니다.

```python
# core/embedding.py

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
- Milvus에 인덱스가 이미 있으면 기본적으로 재사용/스킵합니다 — 인덱싱 속도를 다시 재려면 `--reindex`로
  기존 컬렉션을 삭제하고 강제로 재색인합니다 (`--force`도 자동으로 적용됨).
- raw dense 검색(리랭킹 전) 자체의 NDCG/MRR/Recall/MAP과 query_encode_qps/search_qps도 같이 계산해 저장합니다.
- API가 배치 단위로 호출되기 때문에 항목 하나하나의 시간은 잴 수 없어, 실제로 측정 가능한 가장 작은
  단위인 "배치 1회 호출" 단위로 아래 4가지를 각각 기록합니다 (해당 배치에 포함된 id 목록도 같이 저장):
  - `chunk_timings` — 청크 임베딩(인덱싱), `sec_per_chunk`
  - `insert_timings` — Milvus insert(차원별 동적 배치 크기), `sec_per_row`
  - `query_timings` — 쿼리 임베딩, `sec_per_query`
  - `search_timings` — Milvus 검색(128개 쿼리씩 청크), `sec_per_query`
- 결과 형식: `{"model": ..., "ndcg_at_10": ..., "search_qps": ..., "chunk_timings": [{"batch_index": 0, "n_chunks": 512, "sec_per_chunk": ..., "chunk_ids": [...]}, ...], "query_timings": [...], "results": {"query_id": ["doc_id", ...100개], ...}}`

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
- `--embedding-rerank-model`은 Embedding API를 그대로 씁니다. `--rerank-model`(전용 rerank API)은 헤더/인증은
  같지만 경로가 다릅니다 (`.../embeddings` 대신 `.../rerank`) — `RERANK_API_ENDPOINT` 환경변수로 명시하거나,
  미설정 시 `EMBEDDING_API_ENDPOINT`의 마지막 경로만 `rerank`로 바꿔서 자동 유도합니다 (`bench/reranker.py`).

### k8s

인프라 매니페스트는 `k8s/` 바로 아래, Job 정의는 `k8s/jobs/`, 사람이 직접 실행하는 헬퍼 스크립트는
`k8s/scripts/`로 나뉘어 있습니다.

- `k8s/milvus.yaml` — Milvus standalone (embedded etcd) + PVC
- `k8s/minio.yaml` — Milvus의 blob storage 백엔드. `milvus.yaml`이 `minio:9000` / `minioadmin`/`minioadmin`으로
  참조하므로 서비스 이름·계정을 바꾸면 `milvus.yaml`도 같이 맞춰야 합니다.
- `k8s/starrocks.yaml` — StarRocks(allin1: FE+BE 단일 컨테이너, 벡터 인덱스 HNSW) standalone + PVC.
  `k8s/milvus.yaml`과 동일한 리소스 requests/limits로 맞춰 공정하게 비교되도록 함.
- `k8s/bench-results-pvc.yaml` — retrieval/rerank Job들이 공유하는 결과용 PVC(`bench-results`).
  `k8s/jobs/bench-jobs.yaml`에 같이 있던 걸 분리한 것 — retrieval Job 없이 rerank만(로컬에 이미 있는
  top100 결과를 주입해서) 돌릴 때도 이 파일만 먼저 적용하면 됩니다.
- `k8s/pod.yaml` — 수동 디버그용 Pod (`kubectl exec`로 들어가 직접 실행). `bench-results` PVC를
  마운트한 채 계속 떠있어서, `k8s/scripts/fetch-results.ps1`이 이 pod를 PVC 읽기 통로로도 사용하고,
  로컬 top100 결과를 PVC에 주입할 때도 `kubectl cp`의 대상으로 씁니다
  (Job의 pod는 완료되면 컨테이너가 종료돼서 그 pod로는 kubectl cp/exec가 안 됨 — 데이터는
  PVC에 그대로 남아있지만 읽어올/써넣을 살아있는 pod가 따로 필요함).
- `k8s/jobs/bench-jobs.yaml` — retrieval 3개 모델(HCP-LLM-Latest / bge-m3 / Qwen3-Embedding-8B) Job.
  각 Job의 `REINDEX` env 값을 `"true"`로 바꾸면(Job은 재적용 전 delete 필요) 인덱스가 있어도 강제로
  재색인해서 인덱싱 속도를 다시 측정합니다 (기본값 `"false"` — 있으면 재사용).
- `k8s/jobs/bench-jobs-starrocks.yaml` — `bench-jobs.yaml`과 동일한 retrieval 3모델 벤치마크를
  StarRocks로 돌리는 Job (`VECTOR_BACKEND=starrocks`). 결과는 `/results/retrieval-starrocks`에
  저장되어 Milvus 결과(`/results/retrieval`)와 섞이지 않습니다. `k8s/starrocks.yaml`을
  먼저 적용/Ready 대기한 뒤 적용하세요.
- `k8s/jobs/rerank-job-bge.yaml`, `k8s/jobs/rerank-job-qwen3-top{5,10,20,50,100}.yaml` — retrieval Job들이
  끝난 뒤 적용하는 reranking Job. top_n(5/10/20/50/100)별로 Job(pod)을 분리합니다. bge는
  5개를 동시에 적용해도 되지만, qwen3는 `./k8s/scripts/run-rerank-qwen3-sequential.ps1`로 하나씩
  순서대로 실행해야 합니다(2026-07-06) — 5개를 동시에 띄우면 서로 같은 API를 두들겨 밀리고
  코퍼스/retrieval JSON이 pod마다 중복으로 메모리에 올라가 OOM이 났습니다. retrieval 없이
  로컬 top100 결과만으로 돌리는 방법은 각 파일 헤더의 "PVC 준비" 절 참고.
- `k8s/scripts/fetch-results.ps1` — 위 디버그 pod을 자동으로 띄우고, Job 완료를 기다렸다가 결과 JSON/로그를
  로컬 `results/`로 자동 복사 (`./k8s/scripts/fetch-results.ps1 -Stage retrieval`, `-Stage rerank`, `-Stage all`)
- `k8s/scripts/create-secret-from-env.ps1` — 로컬 `.env`에서 `milvus-migration-secret`을 새 네임스페이스에 생성.

> retrieval 결과 PVC(`bench-results`)는 3개 Job이 동시에 마운트하므로 `ReadWriteMany`(nfs.csi.k8s.io
> 기반 StorageClass)로 만들어져 있습니다. `ReadWriteOnce`로 바꾸면 동시 실행 시 volume attach 대기로 멈춥니다.
> 클러스터마다 StorageClass 이름이 다르니 `k8s/bench-results-pvc.yaml`의 `storageClassName`을 환경에 맞게 바꾸세요.

## 부하테스트 (`loadtest/`)

Locust 기반 Milvus/StarRocks 부하테스트. `bench/`(정확도·성능 벤치마크)와 별개로, 백엔드별
동시성 한계(기준선/램프)를 측정합니다.

- 사전 준비(네임스페이스당 최초 1회): `kubectl apply -f loadtest/k8s/rbac.yaml`, `pvc.yaml`,
  그리고 `milvus-migration-secret`(EMBEDDING_API_ENDPOINT/EMBEDDING_API_KEY/MILVUS_TOKEN).
- 실행: `kubectl apply -f loadtest/k8s/locust-job.yaml -n <namespace>` 한 번이면 끝 —
  `entrypoint.sh`가 `BACKENDS`(기본 `milvus starrocks`) 각각에 대해 배포 → corpus 색인 →
  기준선(N값별)/램프 테스트 → 정리까지 전체 생애주기를 스스로 반복합니다. **테스트 1회는
  항상 독립된 milvus/minio(또는 starrocks) 인스턴스 위에서 실행**되어 이전 테스트의 잔여
  상태가 다음 결과에 섞이지 않습니다.
- 결과 가져오기: `./loadtest/k8s/scripts/fetch-results.sh <namespace>` (또는 `.ps1`) — Job 완료를
  기다리지 않고 그 시점의 PVC 상태를 디버그 pod(`loadtest/k8s/debug/debug-pod.yaml`)를 통해 로컬로 복사합니다.
- 수동/대화형 확인이 필요하면 `loadtest/k8s/debug/locust-pod.yaml`을 띄워 locust 웹 UI
  (`kubectl port-forward` 후 브라우저 접속) 또는 셸에서 직접 명령을 실행할 수 있습니다.
- `loadtest/cache_query_vectors.py`/`cache_chunk_vectors.py`가 쿼리·corpus 임베딩을 PVC에
  1회만 캐싱해두므로, 테스트마다 backend를 재배포해도 Embedding API를 다시 호출하지 않습니다.

## 파일 구조

```
milvus_migration/
├── __init__.py              # from milvus_migration import Pipeline, Config
├── core/                    # 핵심 라이브러리 (Docker 이미지 루트가 이 레포 자체라 패키지명과 겹치지 않게 분리)
│   ├── config.py               # 환경변수 설정
│   ├── embedding.py            # Cloud Platform Embedding API 클라이언트
│   ├── milvus_store.py         # Milvus CRUD
│   ├── starrocks_store.py      # StarRocks CRUD (벡터 인덱스 HNSW) — MilvusStore와 동일 인터페이스
│   ├── vector_store_common.py  # 두 store가 공유하는 상수/유틸 (텍스트 truncate 등)
│   ├── store_factory.py        # VECTOR_BACKEND로 MilvusStore/StarRocksStore 선택
│   ├── pipeline.py             # 인덱싱 + 검색 파이프라인 (백엔드 무관)
│   └── main.py                 # CLI 진입점 (`python -m milvus_migration.core.main`)
├── bench/                   # 정확도·성능 벤치마크 (retrieval → reranking)
│   ├── data_loader.py     # parquet 데이터 로더
│   ├── evaluator.py       # NDCG / MRR / Recall / MAP
│   ├── reranker.py        # Rerank API 클라이언트
│   ├── retrieval_runner.py  # 1단계: 임베딩 모델별 top-100 검색 및 저장
│   └── rerank_runner.py     # 2단계: 저장된 후보 기반 리랭킹 벤치마크
├── k8s/                     # bench용 배포/운영 매니페스트
│   ├── milvus.yaml, minio.yaml, starrocks.yaml   # 인프라 standalone + PVC
│   ├── bench-results-pvc.yaml  # 결과 공유용 PVC(ReadWriteMany) — retrieval/rerank Job 공용
│   ├── pod.yaml            # 수동 디버그용 Pod (fetch-results의 PVC 읽기 통로)
│   ├── jobs/                  # kubectl apply로 실행하는 Job 정의
│   │   ├── bench-jobs.yaml         # retrieval Job × 3 — Milvus
│   │   ├── bench-jobs-starrocks.yaml  # 동일 retrieval Job × 3 — StarRocks
│   │   ├── rerank-job-bge.yaml      # reranking Job — bge-reranker-v2-m3 (top_n별 5개, 동시 적용 가능)
│   │   └── rerank-job-qwen3-top{5,10,20,50,100}.yaml  # reranking Job — Qwen3-Embedding-8B (top_n별 5개)
│   └── scripts/                # 사람이 직접 실행하는 헬퍼 스크립트
│       ├── run-rerank-qwen3-sequential.ps1  # qwen3 rerank Job 5개를 순서대로 하나씩 실행
│       ├── fetch-results.ps1        # 결과 로컬 자동 복사 스크립트
│       └── create-secret-from-env.ps1  # 로컬 .env로 milvus-migration-secret 생성
├── loadtest/                # Locust 기반 동시성 부하테스트
│   ├── common.py            # gevent-safe pymilvus 호출 헬퍼 등 locustfile 공용 유틸
│   ├── locustfile_baseline.py  # 테스트 A: 고정 interval 기준선
│   ├── locustfile_ramp.py      # 테스트 B: 동시 사용자 램프업
│   ├── seed_collection.py      # 캐시된 벡터를 backend에 삽입(재임베딩 없음)
│   ├── cache_query_vectors.py  # 쿼리 임베딩 1회 캐싱(PVC)
│   ├── cache_chunk_vectors.py  # corpus 임베딩 1회 캐싱(PVC)
│   ├── report.py            # 기준선/램프 결과 리포트 생성
│   ├── entrypoint.sh         # Job의 전체 생애주기(배포→색인→테스트→정리) 관리
│   ├── Dockerfile            # 부하테스트 전용 이미지(kubectl 포함)
│   ├── requirements.txt
│   └── k8s/
│       ├── rbac.yaml         # Job이 자기 네임스페이스의 milvus/minio/starrocks를 스스로 apply/delete
│       ├── pvc.yaml          # 결과+캐시 공유 PVC(loadtest-results, ReadWriteMany)
│       ├── milvus.yaml, minio.yaml, starrocks.yaml  # bench용 k8s/*와 이름 충돌 방지(-loadtest 접미사)
│       ├── locust-job.yaml   # 전체 자동화 Job — 평소엔 이것만 apply
│       ├── debug/
│       │   ├── locust-pod.yaml   # 수동/대화형(웹 UI) 디버그 Pod
│       │   └── debug-pod.yaml    # fetch-results 전용 PVC 읽기 통로 Pod
│       └── scripts/
│           ├── fetch-results.sh  # 결과 로컬 자동 복사 스크립트 (bash)
│           └── fetch-results.ps1 # 결과 로컬 자동 복사 스크립트 (PowerShell)
├── data/                    # corpus/queries/qrels parquet
├── results/                 # 로컬 벤치마크 결과 출력 (gitignore)
├── Dockerfile               # bench용 메인 이미지
├── requirements.txt
└── .env.example
```
