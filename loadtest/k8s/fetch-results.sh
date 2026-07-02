#!/usr/bin/env bash
# k8s/fetch-results.ps1과 동일한 목적/구조 — milvus-loadtest Job 완료를 기다렸다가
# 결과(csv/html/log/report.txt)를 로컬로 자동 복사한다.
#
# Job의 pod는 완료(Completed)되면 컨테이너가 종료돼서 kubectl cp/exec가 안 되므로,
# 같은 PVC(loadtest-results, RWX)를 마운트한 채 계속 떠있는 디버그 pod
# (loadtest/k8s/locust-pod.yaml)를 통해 복사한다. 데이터 자체는 PVC에 남아있고,
# 이 pod는 그걸 읽어오는 통로 역할만 한다.
#
# 사용법:
#   ./loadtest/k8s/fetch-results.sh <namespace> [출력디렉토리] [timeout]
#   ./loadtest/k8s/fetch-results.sh milvus-loadtest
#   ./loadtest/k8s/fetch-results.sh milvus-loadtest results-loadtest 21600s
#
# 기본 timeout(21600s=6h)은 BACKENDS(기본 milvus+starrocks 둘 다) 순회 기준 —
# locust-job.yaml의 activeDeadlineSeconds와 맞춰뒀다. 백엔드 하나만 돌린다면 줄여도 됨.
set -euo pipefail

NAMESPACE="${1:?namespace를 첫 번째 인자로 넘겨주세요 (예: ./loadtest/k8s/fetch-results.sh milvus-loadtest)}"
OUT_DIR="${2:-results-loadtest}"
TIMEOUT="${3:-21600s}"
JOB_NAME="milvus-loadtest"
DEBUG_POD="locust-loadtest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "디버그 pod($DEBUG_POD) 준비 중 (PVC 읽기 통로)..."
kubectl apply -f "$SCRIPT_DIR/locust-pod.yaml" -n "$NAMESPACE"
kubectl wait --for=condition=Ready "pod/$DEBUG_POD" -n "$NAMESPACE" --timeout=180s

echo "[$JOB_NAME] 완료 대기 중 (timeout=$TIMEOUT)..."
kubectl wait --for=condition=complete "job/$JOB_NAME" -n "$NAMESPACE" --timeout="$TIMEOUT"

mkdir -p "$OUT_DIR"
echo "[$JOB_NAME] $DEBUG_POD(같은 PVC) → $OUT_DIR 로 결과 복사 중..."
kubectl cp "$NAMESPACE/$DEBUG_POD:/results" "$OUT_DIR"

kubectl logs "job/$JOB_NAME" -n "$NAMESPACE" > "$OUT_DIR/${JOB_NAME}.log" 2>&1 || true
echo "완료: $OUT_DIR  (report.txt에 breakpoint/sentinel/CPU 경고 요약 포함)"
