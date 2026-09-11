<#
.SYNOPSIS
    One-command launcher for Clipping Machine (Windows).

.DESCRIPTION
    Brings up everything the app needs, in order, and stops with a clear
    error if a step fails:

      1. Docker Desktop  (started if not already running)
      2. Postgres + Redis containers (waits until both answer, not just "created")
      3. Python venv     (created + dependencies installed on first run)
      4. .env            (copied from .env.example if missing)
      5. Alembic migrations -> head
      6. Ollama          (started if installed and not already listening)
      7. API server      (own window, titled "Clipping Machine - API")
      8. RQ worker       (own window, titled "Clipping Machine - Worker")
      9. Dev bearer token, copied to the clipboard
     10. Browser opened on the dev console

    Run .\stop.ps1 to shut the pieces back down.

.EXAMPLE
    .\start.ps1
    .\start.ps1 -SkipOllama -NoBrowser
#>
[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$SkipOllama,
    [switch]$SkipMigrations,
    [string]$DevUserEmail = "you@example.com",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------- helpers --

function Write-Step { param([string]$Message) Write-Host ""; Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Good { param([string]$Message) Write-Host "    ok   $Message" -ForegroundColor Green }
function Write-Note { param([string]$Message) Write-Host "    note $Message" -ForegroundColor Yellow }
function Write-Bad  { param([string]$Message) Write-Host "    FAIL $Message" -ForegroundColor Red }

function Stop-WithError {
    param([string]$Message, [string]$Hint)
    Write-Bad $Message
    if ($Hint) { Write-Host ""; Write-Host "  How to fix: $Hint" -ForegroundColor Yellow }
    Write-Host ""
    exit 1
}

# True when something is listening on a local TCP port. Used instead of
# Test-NetConnection, which takes seconds per call when the port is closed.
function Test-LocalPort {
    param([int]$PortNumber, [int]$TimeoutMs = 400)
    $client = $null
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $async = $client.BeginConnect('127.0.0.1', $PortNumber, $null, $null)
        $connected = $async.AsyncWaitHandle.WaitOne($TimeoutMs)
        if ($connected) {
            try { $client.EndConnect($async) } catch { $connected = $false }
        }
        return $connected
    } catch {
        return $false
    } finally {
        if ($client) { $client.Close() }
    }
}

function Test-CommandExists {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# Runs a native command and reports BOTH its exit code and its combined
# output.
#
# The $ErrorActionPreference dance is load-bearing, not defensive noise.
# Windows PowerShell 5.1 turns a native command's stderr into
# NativeCommandError records when the stream is redirected; with
# $ErrorActionPreference='Stop' (set at the top of this script) those become
# TERMINATING errors. So a command that merely prints a warning and exits 0
# -- 'docker info' and 'docker compose up' both do this routinely on Windows
# -- would throw, get swallowed by a catch, and be reported as failure. That
# is exactly what made an earlier version of this script wait forever on a
# healthy Docker engine. PowerShell 7 does not behave this way, so this can
# not be reproduced on pwsh: leave the preference handling alone.
function Invoke-Native {
    param([scriptblock]$Command)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $global:LASTEXITCODE = $null
    $output = ''
    try {
        $output = (& $Command 2>&1 | Out-String)
    } catch {
        $ErrorActionPreference = $previousPreference
        return @{ Ok = $false; Output = $_.Exception.Message }
    }
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    # $null means the command never ran (not found); only 0 is success.
    return @{ Ok = ($exitCode -eq 0); Output = $output.Trim() }
}

function Invoke-Quiet {
    param([scriptblock]$Command)
    return (Invoke-Native $Command).Ok
}

function Wait-Until {
    param(
        [scriptblock]$Condition,
        [int]$TimeoutSeconds,
        [string]$WaitingFor
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $spinner = @('|', '/', '-', '\')
    $clearLine = "`r" + (' ' * 72) + "`r"
    $i = 0
    while ((Get-Date) -lt $deadline) {
        if (& $Condition) {
            Write-Host $clearLine -NoNewline
            return $true
        }
        $elapsed = [int]((Get-Date) - $deadline.AddSeconds(-$TimeoutSeconds)).TotalSeconds
        Write-Host "`r    .... waiting for $WaitingFor $($spinner[$i % 4]) ${elapsed}s/${TimeoutSeconds}s" -NoNewline
        $i++
        Start-Sleep -Milliseconds 700
    }
    Write-Host $clearLine -NoNewline
    return $false
}

# Relaunch child windows with the same PowerShell host this script runs in.
$PsExe = if ($PSVersionTable.PSEdition -eq 'Core') { 'pwsh.exe' } else { 'powershell.exe' }

$Root = $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }
$RootEscaped = $Root.Replace("'", "''")

Set-Location -LiteralPath $Root

Write-Host ""
Write-Host "  Clipping Machine" -ForegroundColor White
Write-Host "  $Root" -ForegroundColor DarkGray

# ------------------------------------------------------------- 1. docker --

Write-Step "Docker"

if (-not (Test-CommandExists 'docker')) {
    Stop-WithError "docker command not found." "Install Docker Desktop, then re-run this script."
}

# 'docker version' is a quieter probe than 'docker info': it asks only whether
# the daemon answers, without printing the pile of configuration warnings that
# made the old check noisy.
function Test-DockerEngine {
    return (Invoke-Native { docker version --format '{{.Server.Version}}' })
}

$docker = Test-DockerEngine

if ($docker.Ok) {
    Write-Good "Docker engine is running (server $($docker.Output))."
} else {
    Write-Note "Docker engine is not answering - starting Docker Desktop."
    Write-Host "         reason: $($docker.Output)" -ForegroundColor DarkGray

    $dockerDesktop = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
    if (Test-Path -LiteralPath $dockerDesktop) {
        Start-Process -FilePath $dockerDesktop | Out-Null
    } else {
        Write-Note "Docker Desktop.exe not at its default path - start it yourself; this will pick it up."
    }

    $dockerUp = Wait-Until -Condition { (Test-DockerEngine).Ok } -TimeoutSeconds 240 -WaitingFor "Docker engine"
    if (-not $dockerUp) {
        $last = Test-DockerEngine
        Write-Host ""
        Write-Host "  docker says:" -ForegroundColor DarkGray
        Write-Host "  $($last.Output)" -ForegroundColor DarkGray
        Stop-WithError "Docker engine did not answer within 240s." @"
Open Docker Desktop and check it finished starting (the whale stops animating).
A first start after a reboot, a pending update prompt, or a WSL2 restart can all
hold it here. Once 'docker version' works in a normal terminal, re-run this script.
"@
    }
    Write-Good "Docker engine is running."
}

# --------------------------------------------------- 2. postgres + redis --

Write-Step "Postgres + Redis containers"

if (-not (Invoke-Quiet { docker compose up -d })) {
    Stop-WithError "docker compose up -d failed." "Run 'docker compose up -d' by hand in $Root to see the error."
}

$pgReady = Wait-Until -Condition {
    Invoke-Quiet { docker compose exec -T postgres pg_isready -U clipping_machine }
} -TimeoutSeconds 120 -WaitingFor "Postgres"

if (-not $pgReady) {
    Stop-WithError "Postgres did not become ready within 120s." "Check 'docker compose logs postgres'."
}
Write-Good "Postgres is accepting connections."

$redisReady = Wait-Until -Condition {
    Invoke-Quiet { docker compose exec -T redis redis-cli ping }
} -TimeoutSeconds 60 -WaitingFor "Redis"

if (-not $redisReady) {
    Stop-WithError "Redis did not become ready within 60s." "Check 'docker compose logs redis'."
}
Write-Good "Redis is answering PING."

# ---------------------------------------------------------------- 3. venv --

Write-Step "Python environment"

$activate = Join-Path $Root 'venv\Scripts\Activate.ps1'

if (-not (Test-Path -LiteralPath $activate)) {
    if (-not (Test-CommandExists 'python')) {
        Stop-WithError "python command not found." "Install Python 3.11+ and make sure it is on PATH."
    }
    Write-Note "No venv yet - creating one (this takes a minute)."
    python -m venv venv
    if (-not (Test-Path -LiteralPath $activate)) {
        Stop-WithError "Failed to create the venv." "Run 'python -m venv venv' by hand to see the error."
    }
}

# Dot-sourced so PATH/VIRTUAL_ENV apply to this script AND to the child
# processes it spawns below.
. $activate
Write-Good "venv activated."

$env:PYTHONPATH = "."

# scripts\ has no __init__.py, so Python puts scripts\ (not the repo root) on
# sys.path when running them by path -- PYTHONPATH="." is what makes
# 'import app' resolve. Same reason it is set for the child windows below.

if (-not (Invoke-Quiet { python -c "import fastapi, uvicorn, rq, alembic" })) {
    Write-Note "Dependencies missing - installing from requirements.txt (first run takes a few minutes)."
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Stop-WithError "pip install failed." "Run 'pip install -r requirements.txt' by hand to see the error."
    }
}
Write-Good "Dependencies present."

# ----------------------------------------------------------------- 4. env --

$envFile = Join-Path $Root '.env'
if (-not (Test-Path -LiteralPath $envFile)) {
    Copy-Item -LiteralPath (Join-Path $Root '.env.example') -Destination $envFile
    Write-Note ".env was missing - copied from .env.example. Add your OPENAI_API_KEY before transcribing."
}

# ---------------------------------------------------------- 5. migrations --

if (-not $SkipMigrations) {
    Write-Step "Database migrations"
    alembic upgrade head
    if ($LASTEXITCODE -ne 0) {
        Stop-WithError "alembic upgrade head failed." "Run 'alembic upgrade head' by hand to see the error."
    }
    Write-Good "Schema is at head."
}

# -------------------------------------------------------------- 6. ollama --

$ollamaProc = $null

if (-not $SkipOllama) {
    Write-Step "Ollama"

    if (Test-LocalPort -PortNumber 11434) {
        Write-Good "Ollama already listening on 11434."
    } elseif (Test-CommandExists 'ollama') {
        $ollamaCmd = "`$host.UI.RawUI.WindowTitle='Clipping Machine - Ollama'; ollama serve"
        $ollamaProc = Start-Process -FilePath $PsExe `
            -ArgumentList @('-NoExit', '-NoProfile', '-Command', $ollamaCmd) -PassThru
        $ollamaUp = Wait-Until -Condition { Test-LocalPort -PortNumber 11434 } -TimeoutSeconds 60 -WaitingFor "Ollama"
        if ($ollamaUp) {
            Write-Good "Ollama started (own window)."
        } else {
            Write-Note "Ollama did not start within 60s - LLM captions/segment suggestions will fall back."
        }
    } else {
        Write-Note "ollama not installed - LLM captions and segment suggestions will fall back to heuristics."
    }

    # Warn early if the configured model was never pulled: the pipeline would
    # otherwise only discover this at the captions stage, minutes in.
    if (Test-LocalPort -PortNumber 11434) {
        $model = 'llama3.1:8b'
        $modelLine = Select-String -Path $envFile -Pattern '^CAPTION_OLLAMA_MODEL=' -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($modelLine) { $model = ($modelLine.Line -split '=', 2)[1].Trim() }

        $installed = ''
        try { $installed = (ollama list 2>$null | Out-String) } catch { }
        $modelBase = ($model -split ':')[0]
        if ($installed -and ($installed -notmatch [regex]::Escape($modelBase))) {
            Write-Note "Model '$model' does not look pulled yet. Run: ollama pull $model"
        }
    }
}

# ----------------------------------------------------------------- 7. api --

Write-Step "API server"

if (Test-LocalPort -PortNumber $Port) {
    Write-Note "Port $Port is already in use - assuming the API is already running, not starting a second one."
    $apiProc = $null
} else {
    $apiCmd = "Set-Location -LiteralPath '$RootEscaped'; . '.\venv\Scripts\Activate.ps1'; " +
              "`$env:PYTHONPATH='.'; `$host.UI.RawUI.WindowTitle='Clipping Machine - API'; " +
              "uvicorn app.main:app --reload --port $Port"
    $apiProc = Start-Process -FilePath $PsExe `
        -ArgumentList @('-NoExit', '-NoProfile', '-Command', $apiCmd) -PassThru

    $apiUp = Wait-Until -Condition {
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 3
            return ($r.StatusCode -eq 200)
        } catch { return $false }
    } -TimeoutSeconds 90 -WaitingFor "API /health"

    if (-not $apiUp) {
        Stop-WithError "API did not answer /health within 90s." "Check the 'Clipping Machine - API' window for a traceback."
    }
    Write-Good "API healthy on http://localhost:$Port"
}

# -------------------------------------------------------------- 8. worker --

Write-Step "Worker"

$workerCmd = "Set-Location -LiteralPath '$RootEscaped'; . '.\venv\Scripts\Activate.ps1'; " +
             "`$env:PYTHONPATH='.'; `$host.UI.RawUI.WindowTitle='Clipping Machine - Worker'; " +
             "python worker_entrypoint.py"
$workerProc = Start-Process -FilePath $PsExe `
    -ArgumentList @('-NoExit', '-NoProfile', '-Command', $workerCmd) -PassThru
Write-Good "Worker started (own window, PID $($workerProc.Id))."

# --------------------------------------------------------------- 9. token --

Write-Step "Dev bearer token"

$token = ''
try {
    $tokenOutput = python scripts\create_dev_user.py $DevUserEmail 2>&1 | Out-String
    $tokenLines = $tokenOutput -split "`r?`n" | Where-Object { $_.Trim() -ne '' }
    if ($tokenLines) {
        $candidate = $tokenLines[-1].Trim()
        # A JWT is three dot-separated base64url chunks; anything else means
        # the script printed an error instead and should be shown as-is.
        if ($candidate -match '^[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+$') { $token = $candidate }
    }
} catch {
    Write-Note "Could not create/fetch the dev user: $($_.Exception.Message)"
}

if ($token) {
    try {
        Set-Clipboard -Value $token
        Write-Good "Token copied to clipboard (user: $DevUserEmail)."
    } catch {
        Write-Good "Token ready (clipboard unavailable)."
    }
    Write-Host ""
    Write-Host $token -ForegroundColor DarkGray
} else {
    Write-Note "No token produced - run 'python scripts\create_dev_user.py $DevUserEmail' by hand."
}

# ------------------------------------------------------------- pid record --

$runDir = Join-Path $Root '.run'
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$pidRecord = [ordered]@{ startedAt = (Get-Date).ToString('s'); port = $Port }
if ($apiProc)    { $pidRecord.api    = $apiProc.Id }
if ($workerProc) { $pidRecord.worker = $workerProc.Id }
if ($ollamaProc) { $pidRecord.ollama = $ollamaProc.Id }
$pidRecord | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runDir 'pids.json') -Encoding UTF8

# ------------------------------------------------------------- 10. browser --

if (-not $NoBrowser) {
    Start-Process "http://localhost:$Port" | Out-Null
}

Write-Host ""
Write-Host "  Ready." -ForegroundColor Green
Write-Host "  Console:  http://localhost:$Port" -ForegroundColor White
Write-Host "  Paste the token above into the console's token field (already on your clipboard)." -ForegroundColor DarkGray
Write-Host "  Stop everything:  .\stop.ps1" -ForegroundColor DarkGray
Write-Host ""
