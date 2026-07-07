<#
.SYNOPSIS
  milvus-loadtest Job의 결과(csv/html/log/report.txt)를 로컬로 복사합니다.
  Job 완료를 기다리지 않고 그 시점에 PVC에 있는 상태를 그대로 가져옵니다 —
  진행 중에 여러 번 실행해도 됩니다.

  Job의 pod는 완료(Completed)되면 컨테이너가 종료돼서 kubectl cp/exec가 안 되므로,
  같은 PVC(loadtest-results, RWX)를 마운트한 채 계속 떠있는 디버그 pod
  (loadtest/k8s/debug-pod.yaml)를 통해 복사합니다.

.EXAMPLE
  ./loadtest/k8s/fetch-results.ps1
  ./loadtest/k8s/fetch-results.ps1 -Namespace user-x0179564-loadtest -OutDir results-loadtest
#>
param(
    [string]$Namespace = "user-x0179564-loadtest",
    [string]$OutDir = "results-loadtest"
)

$ErrorActionPreference = "Stop"
$JobName = "milvus-loadtest"
$DebugPod = "loadtest-debug"

Write-Host "디버그 pod($DebugPod) 준비 중 (PVC 읽기 통로, namespace=$Namespace)..."
kubectl -n $Namespace apply -f "$PSScriptRoot/debug-pod.yaml" | Out-Null
kubectl -n $Namespace wait --for=condition=Ready "pod/$DebugPod" --timeout=180s
if ($LASTEXITCODE -ne 0) {
    throw "디버그 pod($DebugPod)가 준비되지 않았습니다. 'kubectl describe pod $DebugPod -n $Namespace'로 확인하세요."
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Host "[$JobName] $DebugPod(같은 PVC) -> $OutDir 로 결과 복사 중..."
kubectl -n $Namespace cp "${DebugPod}:/results" $OutDir

kubectl -n $Namespace logs "job/$JobName" | Out-File -Encoding utf8 "$OutDir\$JobName.log"
Write-Host "완료: $OutDir  (report.txt에 breakpoint/sentinel/CPU 경고 요약 포함)"
