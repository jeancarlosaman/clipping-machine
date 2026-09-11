@echo off
REM Double-click launcher for Clipping Machine.
REM Runs start.ps1 with the execution policy bypassed for this process only,
REM so no system-wide policy change is needed. The window stays open (-NoExit)
REM so the dev token and any error stay readable.
powershell.exe -NoExit -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
