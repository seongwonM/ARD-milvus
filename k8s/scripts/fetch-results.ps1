<#
.SYNOPSIS
  retrieval/rerank Job 완료를 기다렸다가 결과 JSON(top-100 id, rerank 성능)과 로그를
  로컬 results/ 폴더로 자동 복사합니다. GitHub push 없이 로컬로 결과를 빼는 용도.

  Job의 pod는 완료(Completed)되면 컨테이너가 종료돼서 kubectl cp/exec가 안 되므로,
  같은 PVC(bench-results, RWX)를 마운트한 채 계속 떠있는 디버그 pod(k8s/pod.yaml,
  ard-milvus)를 통해 복사합니다. 데이터 자체는 PVC에 그대로 남아있고, 이 pod는
  그걸 읽어오는 통로 역할만 합니다.

.EXAMPLE
  ./k8s/scripts/fetch-results.ps1 -Stage retrieval
  ./k8s/scripts/fetch-results.ps1 -Stage retrieval-starrocks
  ./k8s/scripts/fetch-results.ps1 -Stage rerank
  ./k8s/scripts/fetch-results.ps1 -Stage all
#>
param(
    [ValidateSet("retrieval", "retrieval-starrocks", "rerank", "all")]
    [string]$Stage = "all",
    [string]$OutDir = "results",
    [string]$Namespace = "user-x0179564",
    [string]$StarrocksNamespace = "user-x0179564-starrocks",
    [int]$TimeoutSec = 3600
)

$ErrorActionPreference = "Stop"
$nsArgs = @()
if ($Namespace) { $nsArgs = @("-n", $Namespace) }
$nsArgsStarrocks = @()
if ($StarrocksNamespace) { $nsArgsStarrocks = @("-n", $StarrocksNamespace) }
$DebugPod = "ard-milvus"

function Ensure-DebugPod {
    param([string[]]$NsArgs)

    Write-Host "디버그 pod($DebugPod) 준비 중 (PVC 읽기 통로, namespace=$($NsArgs[1]))..."
    kubectl @NsArgs apply -f "$PSScriptRoot/../pod.yaml" | Out-Null
    kubectl @NsArgs wait --for=condition=Ready "pod/$DebugPod" --timeout=180s
    if ($LASTEXITCODE -ne 0) {
        throw "디버그 pod($DebugPod)가 준비되지 않았습니다. 'kubectl describe pod $DebugPod -n $($NsArgs[1])'로 확인하세요."
    }
}

function Wait-AndFetch {
    param([string]$JobName, [string]$RemotePath, [string]$LocalPath, [string[]]$NsArgs)

    Write-Host "[$JobName] 완료 대기 중 (namespace=$($NsArgs[1]), timeout=${TimeoutSec}s)..."
    kubectl wait @NsArgs --for=condition=complete "job/$JobName" --timeout="${TimeoutSec}s"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "[$JobName] 완료 대기 실패 — 'kubectl logs job/$JobName -n $($NsArgs[1])' 로 상태를 확인하세요."
        return
    }

    New-Item -ItemType Directory -Force -Path $LocalPath | Out-Null
    Write-Host "[$JobName] $DebugPod(같은 PVC) → $LocalPath 로 결과 복사 중..."
    kubectl @NsArgs cp "${DebugPod}:${RemotePath}" $LocalPath

    kubectl @NsArgs logs "job/$JobName" | Out-File -Encoding utf8 "$OutDir\$JobName.log"
    Write-Host "[$JobName] 완료: $LocalPath  (로그: $OutDir\$JobName.log)"
}

if ($Stage -eq "retrieval" -or $Stage -eq "rerank" -or $Stage -eq "all") {
    Ensure-DebugPod -NsArgs $nsArgs
}

if ($Stage -eq "retrieval" -or $Stage -eq "all") {
    foreach ($job in @("bench-retrieval-hcp", "bench-retrieval-m3", "bench-retrieval-qwen3")) {
        Wait-AndFetch -JobName $job -RemotePath "/results/retrieval" -LocalPath "$OutDir\retrieval" -NsArgs $nsArgs
    }
}

if ($Stage -eq "retrieval-starrocks" -or $Stage -eq "all") {
    # k8s/jobs/bench-jobs-starrocks.yaml Job들은 별도 네임스페이스($StarrocksNamespace)에서 돎.
    # 결과는 별도 경로(/results/retrieval-starrocks)에 저장됨(Milvus 쪽 /results/retrieval과
    # 파일명이 겹치지 않도록).
    Ensure-DebugPod -NsArgs $nsArgsStarrocks
    foreach ($job in @("bench-retrieval-hcp-starrocks", "bench-retrieval-m3-starrocks", "bench-retrieval-qwen3-starrocks")) {
        Wait-AndFetch -JobName $job -RemotePath "/results/retrieval-starrocks" -LocalPath "$OutDir\retrieval-starrocks" -NsArgs $nsArgsStarrocks
    }
}

if ($Stage -eq "rerank" -or $Stage -eq "all") {
    # bge/qwen3 둘 다 top_n(5/10/20/50/100)별로 Job/pod가 분리되어 있음. bge는 5개를
    # 동시에 적용해도 되지만, qwen3는 ./k8s/scripts/run-rerank-qwen3-sequential.ps1로 하나씩
    # 순차 실행해야 함(2026-07-06 — 5개 동시 실행 시 서로 API를 두들겨 밀리는 현상 +
    # 코퍼스/retrieval JSON 중복 로딩으로 인한 OOM 때문). 이 스크립트는 Job이 이미
    # 완료된 뒤 결과를 fetch만 하므로 순서는 상관없음.
    $topNs = @(5, 10, 20, 50, 100)
    $rerankJobs = @("bge", "qwen3") | ForEach-Object { $r = $_; $topNs | ForEach-Object { "bench-rerank-$r-top$_" } }
    foreach ($job in $rerankJobs) {
        Wait-AndFetch -JobName $job -RemotePath "/results/rerank" -LocalPath "$OutDir\rerank" -NsArgs $nsArgs
    }
}
