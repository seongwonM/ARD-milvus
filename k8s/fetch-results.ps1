<#
.SYNOPSIS
  retrieval/rerank Job 완료를 기다렸다가 결과 JSON(top-100 id, rerank 성능)과 로그를
  로컬 results/ 폴더로 자동 복사합니다. GitHub push 없이 로컬로 결과를 빼는 용도.

  Job의 pod는 완료(Completed)되면 컨테이너가 종료돼서 kubectl cp/exec가 안 되므로,
  같은 PVC(bench-results, RWX)를 마운트한 채 계속 떠있는 디버그 pod(k8s/pod.yaml,
  ard-milvus)를 통해 복사합니다. 데이터 자체는 PVC에 그대로 남아있고, 이 pod는
  그걸 읽어오는 통로 역할만 합니다.

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
$DebugPod = "ard-milvus"

function Ensure-DebugPod {
    Write-Host "디버그 pod($DebugPod) 준비 중 (PVC 읽기 통로)..."
    kubectl @nsArgs apply -f "$PSScriptRoot/pod.yaml" | Out-Null
    kubectl @nsArgs wait --for=condition=Ready "pod/$DebugPod" --timeout=180s
    if ($LASTEXITCODE -ne 0) {
        throw "디버그 pod($DebugPod)가 준비되지 않았습니다. 'kubectl describe pod $DebugPod'로 확인하세요."
    }
}

function Wait-AndFetch {
    param([string]$JobName, [string]$RemotePath, [string]$LocalPath)

    Write-Host "[$JobName] 완료 대기 중 (timeout=${TimeoutSec}s)..."
    kubectl wait @nsArgs --for=condition=complete "job/$JobName" --timeout="${TimeoutSec}s"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "[$JobName] 완료 대기 실패 — 'kubectl logs job/$JobName' 로 상태를 확인하세요."
        return
    }

    New-Item -ItemType Directory -Force -Path $LocalPath | Out-Null
    Write-Host "[$JobName] $DebugPod(같은 PVC) → $LocalPath 로 결과 복사 중..."
    kubectl @nsArgs cp "${DebugPod}:${RemotePath}" $LocalPath

    kubectl @nsArgs logs "job/$JobName" | Out-File -Encoding utf8 "$OutDir\$JobName.log"
    Write-Host "[$JobName] 완료: $LocalPath  (로그: $OutDir\$JobName.log)"
}

Ensure-DebugPod

if ($Stage -eq "retrieval" -or $Stage -eq "all") {
    foreach ($job in @("bench-retrieval-hcp", "bench-retrieval-m3", "bench-retrieval-qwen3")) {
        Wait-AndFetch -JobName $job -RemotePath "/results/retrieval" -LocalPath "$OutDir\retrieval"
    }
}

if ($Stage -eq "rerank" -or $Stage -eq "all") {
    foreach ($job in @("bench-rerank-bge", "bench-rerank-qwen3")) {
        Wait-AndFetch -JobName $job -RemotePath "/results/rerank" -LocalPath "$OutDir\rerank"
    }
}
