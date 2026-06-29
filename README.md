# Milvus Migration

Cloud Platform Embedding API + Milvus VectorDB 기반 문서 인덱싱 및 검색 패키지.

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
| `EMBEDDING_DIM` | - | `1536` | 모델 출력 벡터 차원 |
| `EMBEDDING_BATCH_SIZE` | - | `64` | API 호출당 최대 텍스트 수 |
| `EMBEDDING_TIMEOUT` | - | `60` | API 타임아웃 (초) |
| `MILVUS_URI` | - | `http://localhost:19530` | Milvus 서버 주소 |
| `MILVUS_TOKEN` | - | `""` | Milvus 인증 토큰 |
| `MILVUS_COLLECTION` | - | `documents` | 기본 컬렉션 이름 |

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
| `vector_mode` | `"dense" \| "sparse"` | `"dense"` | 벡터 타입 |
| `recreate` | `bool` | `False` | 기존 컬렉션 삭제 후 재생성 |
| `batch_size` | `int \| None` | env 값 | 임베딩 배치 크기 |

#### `search()` 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `query` | `str \| list[str]` | - | 검색 쿼리 |
| `top_k` | `int` | `10` | 반환할 최대 결과 수 |
| `collection` | `str \| None` | env 값 | 컬렉션 이름 |
| `vector_mode` | `"dense" \| "sparse"` | `"dense"` | 벡터 타입 |

### CLI

**인덱싱** — JSON 파일을 읽어 Milvus에 저장

```bash
python -m milvus_migration.main index --file docs.json
python -m milvus_migration.main index --file docs.json --recreate          # 기존 컬렉션 초기화
python -m milvus_migration.main index --file docs.json --vector-mode sparse
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

## 파일 구조

```
milvus_migration/
├── config.py        # 환경변수 설정
├── embedding.py     # Cloud Platform API 클라이언트
├── milvus_store.py  # Milvus CRUD (dense / sparse)
├── pipeline.py      # 인덱싱 + 검색 파이프라인
├── main.py          # CLI 진입점
├── requirements.txt
└── .env.example
```
