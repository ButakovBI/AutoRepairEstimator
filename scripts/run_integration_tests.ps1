<#
.SYNOPSIS
    Run the integration test suite against a real Postgres started via
    docker-compose. Windows PowerShell counterpart to run_integration_tests.sh.

.DESCRIPTION
    Starts ONLY the postgres service from docker/docker-compose.yml (no Kafka/
    MinIO overhead), waits for its healthcheck, sets INTEGRATION_DB_URL, runs
    pytest, and tears the container down on exit.

.PARAMETER TestPath
    Optional path(s) to restrict the test run. Defaults to tests/integration.

.EXAMPLE
    .\scripts\run_integration_tests.ps1
    .\scripts\run_integration_tests.ps1 tests/integration/backend/api
#>

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TestPath = @("tests/integration")
)

$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ComposeFile = Join-Path $RootDir "docker/docker-compose.yml"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker CLI not found. Install Docker Desktop or set `$env:INTEGRATION_DB_URL directly."
    exit 1
}

Write-Host ">> starting postgres container..."
docker compose -f $ComposeFile up -d postgres | Out-Null

try {
    Write-Host ">> waiting for postgres healthcheck..."
    $healthy = $false
    for ($i = 0; $i -lt 30; $i++) {
        $status = docker inspect -f '{{.State.Health.Status}}' auto-repair-postgres 2>$null
        if ($status -eq "healthy") {
            $healthy = $true
            break
        }
        Start-Sleep -Seconds 1
    }

    if (-not $healthy) {
        Write-Error "Postgres did not become healthy within 30s"
        exit 1
    }

    $env:INTEGRATION_DB_URL = "postgresql://auto_repair:auto_repair@localhost:5432/auto_repair"
    Write-Host ">> INTEGRATION_DB_URL=$($env:INTEGRATION_DB_URL)"

    Push-Location $RootDir
    try {
        python -m pytest @TestPath -v
    } finally {
        Pop-Location
    }
}
finally {
    Write-Host ">> stopping postgres container..."
    docker compose -f $ComposeFile down | Out-Null
}
