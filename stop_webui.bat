@echo off
setlocal

rem Stop the detached WebUI backend started by start_webui.bat.
rem This uses taskkill to terminate the backend and its child processes
rem (Playwright Chromium, app-window UI, etc.).

cd /d "%~dp0"
chcp 65001 >nul

set "PID_FILE=%cd%\.tmp\webui.pid"

if not exist "%PID_FILE%" (
  echo [INFO] No PID file found: %PID_FILE%
  exit /b 0
)

for /f "usebackq delims=" %%p in ("%PID_FILE%") do set "PID=%%p"
if "%PID%"=="" (
  echo [WARN] PID file is empty: %PID_FILE%
  del /q "%PID_FILE%" >nul 2>nul
  exit /b 0
)

echo [INFO] Killing WebUI backend (pid=%PID%)...
taskkill /PID %PID% /T /F >nul 2>nul

del /q "%PID_FILE%" >nul 2>nul
echo [INFO] Done.
exit /b 0

