@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_research.ps1" -Mode All
set "RMT_EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%RMT_EXIT_CODE%"=="0" echo Assignment 4 failed. Read the error message above.
pause
exit /b %RMT_EXIT_CODE%
