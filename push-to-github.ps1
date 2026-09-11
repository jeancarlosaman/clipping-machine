<#
.SYNOPSIS
    Commit Clipping Machine and push it to a new PRIVATE GitHub repo.

.DESCRIPTION
    Run this from the repo root (Desktop\clipping-machine). It:
      1. clears a stale git index.lock if one is present
      2. stages everything the .gitignore allows
      3. shows you exactly what will be committed, and refuses to continue
         if anything that looks like a secret or a huge media file slipped in
      4. commits
      5. creates the private GitHub repo and pushes (needs the GitHub CLI)

    Your GitHub credentials stay with you: this script never sees a token,
    it shells out to `gh`, which uses the login you already did.

.EXAMPLE
    .\push-to-github.ps1
    .\push-to-github.ps1 -RepoName clipping-machine -Public:$false
#>
[CmdletBinding()]
param(
    [string]$RepoName = "clipping-machine",
    [string]$Message  = "Clipping Machine MVP: VOD-first AI clipping pipeline",
    [switch]$Public
)

$ErrorActionPreference = "Stop"

function Write-Step { param([string]$m) Write-Host ""; Write-Host "==> $m" -ForegroundColor Cyan }
function Write-Good { param([string]$m) Write-Host "    ok   $m" -ForegroundColor Green }
function Write-Note { param([string]$m) Write-Host "    note $m" -ForegroundColor Yellow }
function Write-Bad  { param([string]$m) Write-Host "    FAIL $m" -ForegroundColor Red }

$Root = $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location -LiteralPath $Root

if (-not (Test-Path -LiteralPath (Join-Path $Root ".gitignore"))) {
    Write-Bad "No .gitignore here. Run this from the repo root (Desktop\clipping-machine)."
    exit 1
}

# --------------------------------------------------------------- 1. lock --

Write-Step "Repository"

$lock = Join-Path $Root ".git\index.lock"
if (Test-Path -LiteralPath $lock) {
    # Left behind by an interrupted `git add`. Safe to remove once no git
    # process is running -- it is a mutex file, not data.
    if (Get-Process git -ErrorAction SilentlyContinue) {
        Write-Bad "A git process is still running. Wait for it to finish, then re-run."
        exit 1
    }
    Remove-Item -LiteralPath $lock -Force
    Write-Note "Removed a stale .git\index.lock"
}

if (-not (Test-Path -LiteralPath (Join-Path $Root ".git"))) {
    git init -q
    Write-Good "Initialised a new repository."
} else {
    Write-Good "Using the existing repository."
}

git config user.name  "Jean Carlo"      | Out-Null
git config user.email "jeancarlosaman@gmail.com" | Out-Null

# ------------------------------------------------------------- 2. stage --

Write-Step "Staging (this walks the tree once -- give it a moment)"
git add -A
if ($LASTEXITCODE -ne 0) { Write-Bad "git add failed."; exit 1 }

$staged = @(git diff --cached --name-only)

# "Nothing staged" is NOT a reason to stop: the usual way to get here is a
# previous run that committed fine but could not reach GitHub (gh missing or
# not logged in). Exiting here would mean re-running the script after
# installing gh does nothing at all, which is precisely the case this script
# most needs to handle.
$hasCommits = $false
git rev-parse --verify HEAD *> $null
if ($LASTEXITCODE -eq 0) { $hasCommits = $true }

if ($staged.Count -eq 0) {
    if (-not $hasCommits) {
        Write-Bad "Nothing to commit and no commits exist -- is this the right folder?"
        exit 1
    }
    Write-Note "Nothing new to commit; the existing commit will be pushed."
} else {
    Write-Good "$($staged.Count) files staged."
}

# ------------------------------------------------- 3. refuse to leak stuff --

Write-Step "Safety check"

# A .gitignore mistake here publishes real credentials, so this is a hard
# stop rather than a warning. Patterns match PATHS, not contents -- the
# contents scan below catches key-shaped strings separately.
$forbidden = $staged | Where-Object {
    $_ -match '(^|/)\.env$' -or
    $_ -match '(^|/)venv/' -or
    $_ -match '(^|/)\.venv/' -or
    $_ -match 'data/objects/' -or
    $_ -match '\.(mp4|mkv|mov|webm|wav|mp3|m4a|pem|key)$' -or
    $_ -match '(^|/)_to_delete/'
}
if ($forbidden) {
    Write-Bad "These should not be committed:"
    $forbidden | Select-Object -First 20 | ForEach-Object { Write-Host "      $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "  Fix .gitignore, then run: git reset" -ForegroundColor Yellow
    exit 1
}
Write-Good "No secrets, virtualenvs or media staged."

# Contents scan: an API key pasted into a tracked source file would pass the
# path check above.
$suspects = @()
foreach ($f in $staged) {
    if (-not (Test-Path -LiteralPath $f)) { continue }          # deletions
    if ((Get-Item -LiteralPath $f).Length -gt 2MB) { continue }  # not source
    $hit = Select-String -LiteralPath $f -Pattern 'sk-[A-Za-z0-9]{20,}', 'ghp_[A-Za-z0-9]{20,}', 'BEGIN (RSA|OPENSSH) PRIVATE KEY' -ErrorAction SilentlyContinue
    if ($hit) { $suspects += "$f : $($hit[0].Line.Trim())" }
}
if ($suspects) {
    Write-Bad "Possible credentials inside tracked files:"
    $suspects | Select-Object -First 10 | ForEach-Object { Write-Host "      $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "  Remove them, then run: git reset" -ForegroundColor Yellow
    exit 1
}
Write-Good "No key-shaped strings inside tracked files."

$bytes = ($staged | Where-Object { Test-Path -LiteralPath $_ } |
          ForEach-Object { (Get-Item -LiteralPath $_).Length } | Measure-Object -Sum).Sum
Write-Good ("Total size: {0:N1} MB" -f ($bytes / 1MB))

# ------------------------------------------------------------ 4. commit --

Write-Step "Commit"

$body = @"
$Message

VOD-first AI clipping tool: ingest -> transcribe -> segment -> score ->
render -> review -> TikTok draft upload. FastAPI + Postgres + Redis/RQ +
FFmpeg, with a plain HTML/JS dev console.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014JoqdWouAZm1pqceAGBS48
"@

if ($staged.Count -eq 0) {
    Write-Note "Skipped (nothing new)."
} else {
    git commit -q -m $body
    if ($LASTEXITCODE -ne 0) { Write-Bad "git commit failed."; exit 1 }
    Write-Good "Committed."
}

# -------------------------------------------------------------- 5. push --

Write-Step "GitHub"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Note "GitHub CLI (gh) is not installed, so the repo was NOT created."
    Write-Host ""
    Write-Host "  EASIEST -- install it, then just re-run this script:" -ForegroundColor Yellow
    Write-Host "    winget install GitHub.cli" -ForegroundColor White
    Write-Host "    (close and reopen PowerShell so gh lands on PATH)" -ForegroundColor DarkGray
    Write-Host "    gh auth login" -ForegroundColor White
    Write-Host "    .\push-to-github.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "  MANUAL -- create the repo at https://github.com/new (tick Private, add no" -ForegroundColor Yellow
    Write-Host "  README/.gitignore), then run these, replacing YOUR-USERNAME with your actual" -ForegroundColor Yellow
    Write-Host "  GitHub username (the one in your profile URL):" -ForegroundColor Yellow
    Write-Host "    git branch -M main" -ForegroundColor White
    Write-Host "    git remote add origin https://github.com/YOUR-USERNAME/$RepoName.git" -ForegroundColor White
    Write-Host "    git push -u origin main" -ForegroundColor White
    Write-Host ""
    Write-Host "  (Your commit is already saved locally either way -- nothing is lost.)" -ForegroundColor DarkGray
    exit 0
}

gh auth status 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Note "gh is installed but not logged in. Run 'gh auth login', then re-run this script."
    exit 0
}

$visibility = if ($Public) { "--public" } else { "--private" }
Write-Host "    creating $RepoName ($(if ($Public) { 'PUBLIC' } else { 'private' }))..." -ForegroundColor DarkGray

git branch -M main
gh repo create $RepoName $visibility --source=. --remote=origin --push
if ($LASTEXITCODE -ne 0) {
    Write-Bad "gh repo create failed (a repo with that name may already exist)."
    Write-Host "  If it exists already:  git remote add origin https://github.com/<you>/$RepoName.git; git push -u origin main" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "  Pushed." -ForegroundColor Green
gh repo view --json url --jq .url 2>$null
Write-Host ""
