<#
.SYNOPSIS
  retrieval/rerank Job 완료를 기다렸다가 결과 JSON(top-100 id, rerank 성능)과 로그를
  로컬 results/ 폴더로 자동 복사합니다. GitHub push 없이 로컬로 결과를 빼는 용도.

.EXAMPLE
  ./k8s/fetch-results.ps1 -Stage retrieval
  ./k8s/fetch-results.ps1 -Stage rerank
  ./k8s/fetch-results.ps1 -Stage all
#>
param(
    [ValidateSet("retrieval", "rerank", "all")]
    [string]$Stage = "all",
    [string]$OutDir = "results",
    [string]$Namespace = "",
    [int]$TimeoutSec = 3600
)

$ErrorActionPreference = "Stop"
$nsArgs = @()
if ($Namespace) { $nsArgs = @("-n", $Namespace) }

function Wait-AndFetch {
    param([string]$JobName, [string]$RemotePath, [string]$LocalPath)

    Write-Host "[$JobName] 완료 대기 중 (timeout=${TimeoutSec}s)..."
    kubectl wait @nsArgs --for=condition=complete "job/$JobName" --timeout="${TimeoutSec}s"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "[$JobName] 완료 대기 실패 — 'kubectl logs job/$JobName' 로 상태를 확인하세요."
        return
    }

    $pod = kubectl @nsArgs get pods --selector="job-name=$JobName" -o jsonpath="{.items[0].metadata.name}"
    if (-not $pod) {
        Write-Warning "[$JobName] pod를 찾을 수 없습니다."
        return
    }

    New-Item -ItemType Directory -Force -Path $LocalPath | Out-Null
    Write-Host "[$JobName] $pod → $LocalPath 로 결과 복사 중..."
    kubectl @nsArgs cp "${pod}:${RemotePath}" $LocalPath

    kubectl @nsArgs logs "job/$JobName" | Out-File -Encoding utf8 "$OutDir\$JobName.log"
    Write-Host "[$JobName] 완료: $LocalPath  (로그: $OutDir\$JobName.log)"
}

if ($Stage -eq "retrieval" -or $Stage -eq "all") {
    foreach ($job in @("bench-retrieval-hcp", "bench-retrieval-m3", "bench-retrieval-qwen3")) {
        Wait-AndFetch -JobName $job -RemotePath "/results/retrieval" -LocalPath "$OutDir\retrieval"
    }
}

if ($Stage -eq "rerank" -or $Stage -eq "all") {
    Wait-AndFetch -JobName "bench-rerank" -RemotePath "/results/rerank" -LocalPath "$OutDir\rerank"
}
