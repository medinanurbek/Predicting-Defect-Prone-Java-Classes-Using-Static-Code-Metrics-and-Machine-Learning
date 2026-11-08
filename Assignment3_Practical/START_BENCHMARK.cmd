@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_research.ps1" -Mode Benchmark
set "RMT_EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %RMT_EXIT_CODE%
