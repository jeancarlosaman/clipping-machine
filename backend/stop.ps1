<#
.SYNOPSIS
    Stops what .\start.ps1 started.

.DESCRIPTION
    Kills the API, worker and (if start.ps1 launched it) Ollama windows using
    the PIDs start.ps1 recorded in .run\pids.json, falling back to a
    command-line scan when that file is missing or stale.

    Containers are left running by default -- they are cheap, healthy across
    reboots, and starting them is the slowest part of start.ps1. Pass
    -Containers to stop those too.

.EXAMPLE
    .\stop.ps1
    .\stop.ps1 -Containers
#>
[CmdletBinding()]
param(
    [switch]$Containers,
    [switch]$KeepOllama
)

$ErrorActionPreference = "Stop"

function Write-Step { param([string]$Message) Write-Host ""; Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Good { param([string]$Message) Write-Host "    ok   $Message" -ForegroundColor Green }
function Write-Note { param([string]$Message) Write-Host "    note $Message" -ForegroundColor Yellow }

$Root = $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location -LiteralPath $Root

# Kills the whole tree: uvicorn --reload runs a reloader parent plus a worker
# child, so stopping only the PowerShell host would orphan the child that
# actually holds port 8000.
function Stop-Tree {
    param([int]$ProcessId, [string]$Label)
    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $proc) {
        Write-Note "$Label (PID $ProcessId) was not running."
        return $false
    }
    taskkill /PID $ProcessId /T /F *> $null
    Write-Good "Stopped $Label (PID $ProcessId)."
    return $true
}

Write-Step "Stopping app processes"

$pidFile = Join-Path $Root '.run\pids.json'
$stoppedAny = $false

if (Test-Path -LiteralPath $pidFile) {
    $record = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json

    foreach ($name in @('api', 'worker', 'ollama')) {
        if ($name -eq 'ollama' -and $KeepOllama) { continue }
        $value = $record.PSObject.Properties[$name]
        if ($value -and $value.Value) {
            if (Stop-Tree -ProcessId ([int]$value.Value) -Label $name) { $stoppedAny = $true }
        }
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
} else {
    Write-Note "No .run\pids.json - falling back to a command-line scan."
}

# Fallback / belt-and-braces: catch anything started by hand or left over from
# a previous run whose pid file was deleted.
$patterns = @(
    @{ Match = 'uvicorn';              Label = 'uvicorn' },
    @{ Match = 'worker_entrypoint.py'; Label = 'worker' }
)
if (-not $KeepOllama) { $patterns += @{ Match = 'ollama serve'; Label = 'ollama' } }

foreach ($pattern in $patterns) {
    $escaped = $pattern.Match.Replace('\', '\\').Replace("'", "''").Replace('%', '[%]').Replace('_', '[_]')
    $query = "SELECT ProcessId, CommandLine FROM Win32_Process WHERE CommandLine LIKE '%$escaped%'"
    $found = @(Get-CimInstance -Query $query -ErrorAction SilentlyContinue)
    foreach ($item in $found) {
        if ($item.ProcessId -eq $PID) { continue }
        if (Stop-Tree -ProcessId ([int]$item.ProcessId) -Label $pattern.Label) { $stoppedAny = $true }
    }
}

if (-not $stoppedAny) { Write-Note "Nothing was running." }

if ($Containers) {
    Write-Step "Stopping containers"
    docker compose down
    if ($LASTEXITCODE -eq 0) {
        Write-Good "Postgres + Redis stopped."
    } else {
        Write-Note "docker compose down reported an error."
    }
} else {
    Write-Step "Containers"
    Write-Note "Left running (use -Containers to stop them too)."
}

Write-Host ""
