<#
.SYNOPSIS
  로컬 .env에서 EMBEDDING_API_ENDPOINT / EMBEDDING_API_KEY / MILVUS_TOKEN 값만 뽑아
  milvus-migration-secret을 새 네임스페이스에 생성합니다(네임스페이스가 없으면 같이 만듦).

  kubectl --from-env-file=.env을 바로 쓰면 .env.example처럼 값 뒤에 인라인 주석
  (# 설명...)이 붙어 있는 줄을 그대로 시크릿 값에 넣어버리는 문제가 있어서, 값만
  정확히 파싱해서 --from-literal로 넣습니다. --dry-run=client -o yaml | kubectl apply -f -
  방식이라 이미 있는 시크릿/네임스페이스에 다시 실행해도 안전합니다(덮어쓰기).

.EXAMPLE
  ./k8s/scripts/create-secret-from-env.ps1 -Namespace user-x0179564-new
  ./k8s/scripts/create-secret-from-env.ps1 -Namespace user-x0179564-new -EnvFile .env.prod
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$Namespace,
    [string]$EnvFile = ".env"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $EnvFile)) {
    throw "$EnvFile 을 찾을 수 없습니다. 프로젝트 루트에서 실행하거나 -EnvFile로 경로를 지정하세요."
}

function Get-EnvValue([string]$Key) {
    $line = Get-Content $EnvFile | Where-Object { $_ -match "^\s*$Key\s*=" } | Select-Object -First 1
    if (-not $line) { return "" }
    $value = $line -replace "^\s*$Key\s*=", ""
    $value = $value -replace "#.*$", ""  # 인라인 주석 제거
    return $value.Trim()
}

$endpoint = Get-EnvValue "EMBEDDING_API_ENDPOINT"
$apiKey   = Get-EnvValue "EMBEDDING_API_KEY"
$token    = Get-EnvValue "MILVUS_TOKEN"

if (-not $endpoint -or -not $apiKey) {
    throw "EMBEDDING_API_ENDPOINT 또는 EMBEDDING_API_KEY가 $EnvFile 에 없습니다."
}

Write-Host "네임스페이스($Namespace) 준비 중..."
kubectl create namespace $Namespace --dry-run=client -o yaml | kubectl apply -f - | Out-Null

Write-Host "milvus-migration-secret 생성/갱신 중..."
kubectl -n $Namespace create secret generic milvus-migration-secret `
    --from-literal=EMBEDDING_API_ENDPOINT=$endpoint `
    --from-literal=EMBEDDING_API_KEY=$apiKey `
    --from-literal=MILVUS_TOKEN=$token `
    --dry-run=client -o yaml | kubectl apply -f -

Write-Host "완료: kubectl -n $Namespace get secret milvus-migration-secret 으로 확인하세요."
