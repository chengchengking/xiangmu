@echo off
setlocal enabledelayedexpansion

rem Stop the detached WebUI backend started by start_webui.bat.

cd /d "%~dp0"
chcp 65001 >nul

set "HOST=127.0.0.1"
set "PORT=8765"
set "URL=http://%HOST%:%PORT%/"
set "TMPDIR=%cd%\.tmp"
set "PID_FILE=%TMPDIR%\webui.pid"

echo [INFO] Trying to stop WebUI via HTTP (%URL%api/stop)...
powershell -NoProfile -Command "try{Invoke-WebRequest -UseBasicParsing -Method POST -ContentType 'application/json' -Body '{}' '%URL%api/stop' -TimeoutSec 2 | Out-Null; 'ok'}catch{'failed'}" >nul 2>nul

rem Give it a moment to close Playwright and the HTTP server.
timeout /t 2 /nobreak >nul

if exist "%PID_FILE%" (
  set /p PID=<"%PID_FILE%"
  if not "!PID!"=="" (
    echo [INFO] Ensuring process is stopped (pid=!PID!)...
    powershell -NoProfile -Command "try{Stop-Process -Id !PID! -Force -ErrorAction Stop; 'killed'}catch{'not running'}" >nul 2>nul
  )
  del /q "%PID_FILE%" >nul 2>nul
)

echo [INFO] Done.
exit /b 0

