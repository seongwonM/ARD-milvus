<#
.SYNOPSIS
  k8s/jobs/rerank-job-qwen3-top{5,10,20,50,100}.yaml을 동시에 띄우지 않고 하나씩 순서대로
  실행합니다: 적용 -> 완료 대기(결과는 Job 안에서 PVC에 저장됨) -> 다음 top_n 적용.

  top_n별로 Job(pod)을 계속 분리하는 이유는 k8s/jobs/rerank-job-bge.yaml과 같습니다 — 한 pod에서
  top_n을 순차로 돌리면 뒤 top_n일수록 느려지는 게 캐시/서버 상태 누적 때문인지 top_n
  자체의 속도 차이인지 구분이 안 됩니다. 대신 5개를 동시에 띄우면 서로 같은 API를 동시에
  두드려 먼저 시작한 pod가 서버를 붙잡는 동안 뒤 pod가 밀리고, 코퍼스/retrieval JSON도
  pod마다 중복으로 메모리에 올라가 OOM으로 이어졌다(2026-07-06) — 그래서 pod는 분리하되
  실행은 이 스크립트로 순차화한다.

  완료된 Job은 일부러 삭제하지 않는다 — Pod는 Succeeded가 되는 순간 더 이상 리소스를
  안 쓰므로("파드 내림") 다음 top_n 시작 전에 삭제할 필요가 없고, Job을 남겨둬야
  k8s/scripts/fetch-results.ps1이 나중에 로그(kubectl logs job/...)를 읽어올 수 있다.

.EXAMPLE
  ./k8s/scripts/run-rerank-qwen3-sequential.ps1
  ./k8s/scripts/run-rerank-qwen3-sequential.ps1 -TimeoutSec 1800
#>
param(
    [string]$Namespace = "user-x0179564",
    [int]$TimeoutSec = 3600
)

$ErrorActionPreference = "Stop"
$nsArgs = @()
if ($Namespace) { $nsArgs = @("-n", $Namespace) }

$topNs = @(5, 10, 20, 50, 100)

foreach ($n in $topNs) {
    $job = "bench-rerank-qwen3-top$n"
    $file = "$PSScriptRoot/../jobs/rerank-job-qwen3-top$n.yaml"

    Write-Host "`n=== [$job] 적용 (이전 top_n의 pod는 이미 완료되어 리소스를 안 쓰는 상태) ==="
    kubectl @nsArgs apply -f $file

    Write-Host "[$job] 완료 대기 중 (timeout=${TimeoutSec}s)..."
    kubectl @nsArgs wait --for=condition=complete "job/$job" --timeout="${TimeoutSec}s"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "[$job] 완료 대기 실패 — 'kubectl logs job/$job'로 확인하세요. 다음 top_n으로 계속 진행합니다."
    }
}

Write-Host "`n전체 완료. 결과는 PVC(bench-results)에 저장돼 있습니다 — ./k8s/scripts/fetch-results.ps1 -Stage rerank 로 로컬에 복사하세요."
