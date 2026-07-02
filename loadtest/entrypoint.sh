#!/bin/sh
# milvus-loadtest Job의 진입점 — 사람이 kubectl로 milvus/minio를 매번 지우고 다시
# 띄울 필요 없이, 이 스크립트가 전체 생애주기를 스스로 관리한다.
#
#   0) 이전 milvus/minio 정리(남아있다면)  — 빈 상태 보장
#   1) milvus/minio 새로 배포 + Ready 대기
#   2) corpus 색인 / 3) 쿼리 벡터 캐싱      — 1회성 준비
#   4) 테스트 A(기준선, N값별 반복)
#   5) 테스트 B(동시성 램프)
#   6) 리포트 생성 + $RESULTS_DIR/report.txt로 저장
#   7) milvus/minio 정리                  — 다음 실행을 위해 다시 빈 상태로
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

echo "===== 0) 이전 milvus/minio 정리 (빈 상태 보장) ====="
cleanup_milvus

echo "===== 1) milvus/minio 새로 배포 ====="
kubectl apply -f "$K8S_DIR/minio.yaml" -n "$MILVUS_NAMESPACE"
kubectl apply -f "$K8S_DIR/milvus.yaml" -n "$MILVUS_NAMESPACE"
kubectl wait --for=condition=Ready pod/milvus -n "$MILVUS_NAMESPACE" --timeout=300s
kubectl rollout status deploy/minio -n "$MILVUS_NAMESPACE" --timeout=300s

echo "===== 2) corpus 색인 ====="
python -m milvus_migration.loadtest.seed_collection --data-root /data

echo "===== 3) 쿼리 벡터 캐싱 ====="
python -m milvus_migration.loadtest.cache_query_vectors --data-root /data

BASELINE_STATS=""
for N in $BASELINE_INTERVALS; do
  echo "===== 테스트 A: interval=${N}s ====="
  locust -f milvus_migration/loadtest/locustfile_baseline.py --headless --only-summary \
    -u 1 -r 1 --run-time "$BASELINE_RUNTIME" --interval "$N" \
    --csv="$RESULTS_DIR/baseline_N${N}" --logfile="$RESULTS_DIR/baseline_N${N}.log"
  BASELINE_STATS="$BASELINE_STATS $RESULTS_DIR/baseline_N${N}_stats.csv"
done

echo "===== 테스트 B: 동시성 램프 ====="
locust -f milvus_migration/loadtest/locustfile_ramp.py --headless --only-summary \
  --logfile="$RESULTS_DIR/ramp.log" --html="$RESULTS_DIR/ramp.html" \
  --csv="$RESULTS_DIR/ramp" --request-log="$RESULTS_DIR/ramp_requests.csv"

echo "===== 리포트 (결과는 $RESULTS_DIR/report.txt 에도 저장됨) ====="
{
  python -m milvus_migration.loadtest.report baseline --stats $BASELINE_STATS
  python -m milvus_migration.loadtest.report ramp \
    --request-log "$RESULTS_DIR/ramp_requests.csv" --logfile "$RESULTS_DIR/ramp.log"
} | tee "$RESULTS_DIR/report.txt"

echo "===== 4) milvus/minio 정리 (다음 실행을 위해 빈 상태로 되돌림) ====="
cleanup_milvus

echo "===== 완료 ====="
