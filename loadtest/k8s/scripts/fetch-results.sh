#!/usr/bin/env bash
# k8s/scripts/fetch-results.ps1과 동일한 목적/구조 — 결과(csv/html/log/report.txt)를 로컬로
# 자동 복사한다. LOADTEST_CACHE_DIR=/results/cache로 저장되는 쿼리/corpus 임베딩
# 캐시도 /results 밑이라 이 스크립트로 같이 빠져나온다(캐시를 따로 빼는 스크립트는
# 없음 — /results 전체를 복사하는 것으로 충분).
#
# Job 완료를 기다리지 않고 그 시점에 PVC에 있는 상태를 그대로 복사한다 — 중간
# 진행 상황을 보고 싶을 때 여러 번 실행해도 됨(2026-07-07, 사용자 요청으로 완료
# 대기 제거). Job의 pod는 완료(Completed)되면 컨테이너가 종료돼서 kubectl cp/exec가
# 안 되므로, 같은 PVC(loadtest-results, RWX)를 마운트한 채 sleep infinity만 하는
# 순수 디버그 pod(loadtest/k8s/debug/debug-pod.yaml)를 통해 복사한다. 데이터 자체는 PVC에
# 남아있고, 이 pod는 그걸 읽어오는 통로 역할만 한다 — locust-pod.yaml은 캐싱/색인까지
# 직접 실행하는 파드라 fetch 용도로는 부적절해서 안 씀.
#
# 사용법:
#   ./loadtest/k8s/scripts/fetch-results.sh <namespace> [출력디렉토리]
#   ./loadtest/k8s/scripts/fetch-results.sh user-x0179564-loadtest
#   ./loadtest/k8s/scripts/fetch-results.sh user-x0179564-loadtest results-loadtest
set -euo pipefail

NAMESPACE="${1:?namespace를 첫 번째 인자로 넘겨주세요 (예: ./loadtest/k8s/scripts/fetch-results.sh user-x0179564-loadtest)}"
OUT_DIR="${2:-results-loadtest}"
JOB_NAME="milvus-loadtest"
DEBUG_POD="loadtest-debug"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "디버그 pod($DEBUG_POD) 준비 중 (PVC 읽기 통로)..."
kubectl apply -f "$SCRIPT_DIR/../debug/debug-pod.yaml" -n "$NAMESPACE"
kubectl wait --for=condition=Ready "pod/$DEBUG_POD" -n "$NAMESPACE" --timeout=180s

mkdir -p "$OUT_DIR"
echo "[$JOB_NAME] $DEBUG_POD(같은 PVC) → $OUT_DIR 로 결과 복사 중..."
kubectl cp "$NAMESPACE/$DEBUG_POD:/results" "$OUT_DIR"

kubectl logs "job/$JOB_NAME" -n "$NAMESPACE" > "$OUT_DIR/${JOB_NAME}.log" 2>&1 || true
echo "완료: $OUT_DIR  (report.txt에 breakpoint/sentinel/CPU 경고 요약 포함)"
