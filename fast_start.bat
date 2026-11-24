@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\fast_start_win.ps1" %*
exit /b %ERRORLEVEL%
