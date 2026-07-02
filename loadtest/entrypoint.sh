#!/bin/sh
# milvus-loadtest Job의 진입점 — 사람이 kubectl로 milvus/minio를 매번 지우고 다시
# 띄울 필요 없이, 이 스크립트가 전체 생애주기를 스스로 관리한다.
#
# 테스트 1회(=baseline의 interval 값 하나, 또는 ramp 테스트 1회)는 항상 독립적으로
# 실행되어야 하므로, 매 테스트마다 milvus/minio pod를 내렸다가 새로 올린다 —
# 이전 테스트가 남긴 캐시/커넥션/부하 잔여 상태가 다음 테스트 결과에 섞이지 않게 하기 위함.
#
#   1) 쿼리 벡터 캐싱 / corpus(청크) 벡터 캐싱 — 둘 다 로컬 파일 캐시라 milvus/minio
#      생애주기와 무관, 1회성. corpus 내용은 테스트 사이에 바뀌지 않으므로 이렇게
#      한 번만 임베딩해두면, 재배포마다 6만 건을 Embedding API로 다시 임베딩하지
#      않고 캐시된 벡터를 삽입만 하면 된다 (재배포 비용을 줄이는 핵심 포인트).
#   2) 테스트 A(기준선): N값마다 [milvus/minio 배포 → corpus 색인(캐시 삽입) → 테스트 → 정리] 반복
#   3) 테스트 B(동시성 램프): [milvus/minio 배포 → corpus 색인(캐시 삽입) → 테스트 → 정리] 1회
#   4) 리포트 생성 + $RESULTS_DIR/report.txt로 저장
#
# loadtest/k8s/rbac.yaml이 부여한 ServiceAccount 권한으로 자기 네임스페이스 안의
# milvus/minio 리소스를 직접 apply/delete한다.
set -e

: "${MILVUS_NAMESPACE:?MILVUS_NAMESPACE가 비어있습니다 — locust-job.yaml의 downward API(metadata.namespace) 설정을 확인하세요}"

K8S_DIR="${K8S_DIR:-/app/milvus_migration/k8s}"
RESULTS_DIR="${RESULTS_DIR:-/results}"

cleanup_milvus() {
  echo "----- milvus/minio 정리 (ns=$MILVUS_NAMESPACE) -----"
  kubectl delete -f "$K8S_DIR/milvus.yaml" -n "$MILVUS_NAMESPACE" --ignore-not-found --wait=true
  kubectl delete -f "$K8S_DIR/minio.yaml" -n "$MILVUS_NAMESPACE" --ignore-not-found --wait=true
}

deploy_milvus() {
  echo "----- milvus/minio 배포 (ns=$MILVUS_NAMESPACE) -----"
  kubectl apply -f "$K8S_DIR/minio.yaml" -n "$MILVUS_NAMESPACE"
  kubectl apply -f "$K8S_DIR/milvus.yaml" -n "$MILVUS_NAMESPACE"
  kubectl wait --for=condition=Ready pod/milvus -n "$MILVUS_NAMESPACE" --timeout=300s
  kubectl rollout status deploy/minio -n "$MILVUS_NAMESPACE" --timeout=300s
}

# 매번 새로 배포된 빈 milvus에 corpus를 다시 색인한다 (milvus/minio를 지웠다 올렸으므로
# --recreate 불필요 — 컬렉션 자체가 없는 상태에서 시작). 임베딩은 1)에서 캐싱된 벡터를
# 그대로 삽입만 하므로 Embedding API를 다시 호출하지 않는다.
seed_corpus() {
  echo "----- corpus 색인 (캐시된 벡터 삽입) -----"
  python -m milvus_migration.loadtest.seed_collection --data-root /data
}

# 테스트 1회를 항상 깨끗한 milvus/minio 위에서 실행: 배포 → 색인 → 테스트 → 정리.
# 정리를 맨 앞에도 넣는 건 이전 실행이 비정상 종료해 리소스가 남아있을 경우의 안전장치.
run_isolated_test() {
  cleanup_milvus
  deploy_milvus
  seed_corpus
  "$@"
  cleanup_milvus
}

echo "===== 1) 쿼리/corpus 벡터 캐싱 (milvus/minio와 무관한 로컬 캐시, 1회성) ====="
python -m milvus_migration.loadtest.cache_query_vectors --data-root /data
python -m milvus_migration.loadtest.cache_chunk_vectors --data-root /data

echo "===== 2) 테스트 A: 기준선 (N값마다 독립 실행) ====="
BASELINE_STATS=""
for N in $BASELINE_INTERVALS; do
  echo "----- interval=${N}s -----"
  run_isolated_test locust -f milvus_migration/loadtest/locustfile_baseline.py --headless --only-summary \
    -u 1 -r 1 --run-time "$BASELINE_RUNTIME" --interval "$N" \
    --csv="$RESULTS_DIR/baseline_N${N}" --logfile="$RESULTS_DIR/baseline_N${N}.log"
  BASELINE_STATS="$BASELINE_STATS $RESULTS_DIR/baseline_N${N}_stats.csv"
done

echo "===== 3) 테스트 B: 동시성 램프 (독립 실행) ====="
run_isolated_test locust -f milvus_migration/loadtest/locustfile_ramp.py --headless --only-summary \
  --logfile="$RESULTS_DIR/ramp.log" --html="$RESULTS_DIR/ramp.html" \
  --csv="$RESULTS_DIR/ramp" --request-log="$RESULTS_DIR/ramp_requests.csv"

echo "===== 4) 리포트 (결과는 $RESULTS_DIR/report.txt 에도 저장됨) ====="
{
  python -m milvus_migration.loadtest.report baseline --stats $BASELINE_STATS
  python -m milvus_migration.loadtest.report ramp \
    --request-log "$RESULTS_DIR/ramp_requests.csv" --logfile "$RESULTS_DIR/ramp.log"
} | tee "$RESULTS_DIR/report.txt"

echo "===== 완료 ====="
