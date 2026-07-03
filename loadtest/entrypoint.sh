#!/bin/sh
# milvus-loadtest Job의 진입점 — 사람이 kubectl로 milvus/minio(또는 starrocks)를
# 매번 지우고 다시 띄울 필요 없이, 이 스크립트가 전체 생애주기를 스스로 관리한다.
#
# BACKENDS(기본 "milvus starrocks")에 나열된 각 백엔드에 대해 동일한 baseline/ramp
# 테스트를 그대로 반복해서, 같은 조건으로 속도를 비교할 수 있게 한다.
#
# 테스트 1회(=baseline의 interval 값 하나, 또는 ramp 테스트 1회)는 항상 독립적으로
# 실행되어야 하므로, 매 테스트마다 해당 백엔드의 pod를 내렸다가 새로 올린다 —
# 이전 테스트가 남긴 캐시/커넥션/부하 잔여 상태가 다음 테스트 결과에 섞이지 않게 하기 위함.
#
#   1) 쿼리 벡터 캐싱 / corpus(청크) 벡터 캐싱 — 백엔드와 무관한 캐시, 1회성만.
#      "텍스트 → 벡터" 변환 결과일 뿐이라 milvus든 starrocks든 완전히 동일한 입력을
#      쓴다 — 재배포마다 6만 건을 Embedding API로 다시 임베딩하지 않고 캐시된 벡터를
#      삽입만 하면 된다 (재배포 비용을 줄이는 핵심 포인트). LOADTEST_CACHE_DIR=/results/cache로
#      PVC에 저장되므로, Job을 지우고 값 바꿔 다시 apply해도(파드가 새로 떠도) 유지된다.
#   2) 백엔드마다:
#      2-1) 테스트 A(기준선): N값마다 [배포 → corpus 색인(캐시 삽입) → 테스트 → 정리] 반복
#      2-2) 테스트 B(동시성 램프): [배포 → corpus 색인(캐시 삽입) → 테스트 → 정리] 1회
#      2-3) 리포트 생성 ($RESULTS_DIR/report_<backend>.txt, 통합은 report.txt)
#
# loadtest/k8s/rbac.yaml이 부여한 ServiceAccount 권한으로 자기 네임스페이스 안의
# 리소스를 직접 apply/delete한다.
set -e

: "${MILVUS_NAMESPACE:?MILVUS_NAMESPACE가 비어있습니다 — locust-job.yaml의 downward API(metadata.namespace) 설정을 확인하세요}"

K8S_DIR="${K8S_DIR:-/app/milvus_migration/loadtest/k8s}"  # bench용 k8s/*.yaml과 별개 (이름 충돌 방지)
RESULTS_DIR="${RESULTS_DIR:-/results}"
BACKENDS="${BACKENDS:-milvus starrocks}"  # 공백 구분 — 순서대로 순회하며 동일 테스트를 각각 독립 실행

deploy_backend() {
  case "$1" in
    milvus)
      kubectl apply -f "$K8S_DIR/minio.yaml" -n "$MILVUS_NAMESPACE"
      kubectl apply -f "$K8S_DIR/milvus.yaml" -n "$MILVUS_NAMESPACE"
      # milvus-loadtest-data PVC는 테스트마다 새로 생성되는데, 이 네임스페이스의
      # 기본 스토리지클래스(디스크 attach 기반)는 NFS보다 프로비저닝/attach가 오래
      # 걸려 300s로는 부족한 경우가 있었음(2026-07-03 Job Failed 실측) → 600s로 완화.
      kubectl wait --for=condition=Ready pod/milvus-loadtest -n "$MILVUS_NAMESPACE" --timeout=600s
      kubectl rollout status deploy/minio-loadtest -n "$MILVUS_NAMESPACE" --timeout=300s
      ;;
    starrocks)
      kubectl apply -f "$K8S_DIR/starrocks.yaml" -n "$MILVUS_NAMESPACE"
      kubectl wait --for=condition=Ready pod/starrocks-loadtest -n "$MILVUS_NAMESPACE" --timeout=300s
      ;;
    *)
      echo "알 수 없는 backend: $1 (milvus|starrocks만 지원)" >&2; exit 1 ;;
  esac
}

cleanup_backend() {
  case "$1" in
    milvus)
      kubectl delete -f "$K8S_DIR/milvus.yaml" -n "$MILVUS_NAMESPACE" --ignore-not-found --wait=true
      kubectl delete -f "$K8S_DIR/minio.yaml" -n "$MILVUS_NAMESPACE" --ignore-not-found --wait=true
      ;;
    starrocks)
      kubectl delete -f "$K8S_DIR/starrocks.yaml" -n "$MILVUS_NAMESPACE" --ignore-not-found --wait=true
      ;;
  esac
}

# 매번 새로 배포된 빈 DB에 corpus를 다시 색인한다 (재배포했으므로 --recreate 불필요 —
# 컬렉션/테이블 자체가 없는 상태에서 시작). 임베딩은 1)에서 캐싱된 벡터를 그대로
# 삽입만 하므로 Embedding API를 다시 호출하지 않는다. VECTOR_BACKEND는 호출자가 export.
seed_corpus() {
  echo "----- corpus 색인 (backend=$VECTOR_BACKEND, 캐시된 벡터 삽입) -----"
  python -m milvus_migration.loadtest.seed_collection --data-root /data
}

# 테스트 1회를 항상 깨끗한 DB 위에서 실행: 배포 → 색인 → 테스트 → 정리.
# 정리를 맨 앞에도 넣는 건 이전 실행이 비정상 종료해 리소스가 남아있을 경우의 안전장치.
run_isolated_test() {
  cleanup_backend "$VECTOR_BACKEND"
  deploy_backend "$VECTOR_BACKEND"
  seed_corpus
  "$@"
  cleanup_backend "$VECTOR_BACKEND"
}

echo "===== 1) 쿼리/corpus 벡터 캐싱 (백엔드 무관 공유 로컬 캐시, 1회성) ====="
python -m milvus_migration.loadtest.cache_query_vectors --data-root /data
python -m milvus_migration.loadtest.cache_chunk_vectors --data-root /data

: > "$RESULTS_DIR/report.txt"  # 통합 리포트 초기화 (백엔드별 리포트를 이어붙임)

for BACKEND in $BACKENDS; do
  export VECTOR_BACKEND="$BACKEND"
  echo "===== [$BACKEND] 테스트 시작 ====="

  echo "----- [$BACKEND] 2) 테스트 A: 기준선 (N값마다 독립 실행) -----"
  BASELINE_STATS=""
  for N in $BASELINE_INTERVALS; do
    echo "  [$BACKEND] interval=${N}s"
    run_isolated_test locust -f milvus_migration/loadtest/locustfile_baseline.py --headless --only-summary \
      -u 1 -r 1 --run-time "$BASELINE_RUNTIME" --interval "$N" \
      --csv="$RESULTS_DIR/baseline_${BACKEND}_N${N}" --logfile="$RESULTS_DIR/baseline_${BACKEND}_N${N}.log"
    BASELINE_STATS="$BASELINE_STATS $RESULTS_DIR/baseline_${BACKEND}_N${N}_stats.csv"
  done

  echo "----- [$BACKEND] 3) 테스트 B: 동시성 램프 (독립 실행) -----"
  run_isolated_test locust -f milvus_migration/loadtest/locustfile_ramp.py --headless --only-summary \
    --logfile="$RESULTS_DIR/ramp_${BACKEND}.log" --html="$RESULTS_DIR/ramp_${BACKEND}.html" \
    --csv="$RESULTS_DIR/ramp_${BACKEND}" --request-log="$RESULTS_DIR/ramp_${BACKEND}_requests.csv"

  echo "----- [$BACKEND] 4) 리포트 ($RESULTS_DIR/report_${BACKEND}.txt) -----"
  {
    echo "===== BACKEND: $BACKEND ====="
    python -m milvus_migration.loadtest.report baseline --stats $BASELINE_STATS
    python -m milvus_migration.loadtest.report ramp \
      --request-log "$RESULTS_DIR/ramp_${BACKEND}_requests.csv" --logfile "$RESULTS_DIR/ramp_${BACKEND}.log"
  } | tee "$RESULTS_DIR/report_${BACKEND}.txt"
  cat "$RESULTS_DIR/report_${BACKEND}.txt" >> "$RESULTS_DIR/report.txt"
done

echo "===== 완료 (백엔드별: report_<backend>.txt, 통합: report.txt) ====="
