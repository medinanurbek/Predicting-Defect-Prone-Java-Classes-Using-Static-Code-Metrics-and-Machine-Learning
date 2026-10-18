@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_demo.ps1"
set "RMT_EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%RMT_EXIT_CODE%"=="0" echo Demo failed. Read the error message above.
pause
exit /b %RMT_EXIT_CODE%
