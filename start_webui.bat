@echo off
setlocal enabledelayedexpansion

rem One-click launcher for Windows.
rem - Ensures deps are installed (best-effort).
rem - Ensures Playwright Chromium is installed (best-effort).
rem - Starts the visible WebUI + Playwright persistent browser session.

cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1

echo [INFO] Working dir: %cd%

echo [INFO] Checking Python deps (playwright)...
python -c "from playwright.sync_api import sync_playwright" >nul 2>nul
if errorlevel 1 (
  echo [WARN] playwright not found. Installing requirements.txt...
  python -m pip install -r requirements.txt
  if errorlevel 1 goto :fail
)

echo [INFO] Checking Playwright Chromium installation...
python -c "from playwright.sync_api import sync_playwright; import os, sys; p=sync_playwright().start(); path=p.chromium.executable_path; p.stop(); sys.exit(0 if os.path.exists(path) else 1)" >nul 2>nul
if errorlevel 1 (
  echo [WARN] Chromium not found. Installing via mirror+retry script...
  powershell -ExecutionPolicy Bypass -File "%~dp0install_playwright_chromium_fast.ps1"
  if errorlevel 1 (
    echo [WARN] Mirror install failed. Trying default installer...
    python -m playwright install chromium
    if errorlevel 1 goto :fail
  )
)

set "HOST=127.0.0.1"
set "PORT=8765"
set "URL=http://%HOST%:%PORT%/"
set "TMPDIR=%cd%\.tmp"
set "PID_FILE=%TMPDIR%\webui.pid"
set "OUT_LOG=%TMPDIR%\webui.out.log"
set "ERR_LOG=%TMPDIR%\webui.err.log"

if not exist "%TMPDIR%" mkdir "%TMPDIR%" >nul 2>nul

rem If WebUI is already running, just open the window and exit.
set "CODE=0"
for /f %%c in ('powershell -NoProfile -Command "try{(Invoke-WebRequest -UseBasicParsing '%URL%api/state' -TimeoutSec 2).StatusCode}catch{0}"') do set "CODE=%%c"
if "!CODE!"=="200" (
  echo [INFO] WebUI already running: %URL%
  python -c "import ai_duel_webui as w; w._open_webui_window('%URL%')"
  exit /b 0
)

echo [INFO] Starting WebUI backend (detached)...
echo [INFO] Logs: %OUT_LOG% and %ERR_LOG%

del /q "%OUT_LOG%" >nul 2>nul
del /q "%ERR_LOG%" >nul 2>nul
del /q "%PID_FILE%" >nul 2>nul

powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Start-Process -FilePath 'python' -ArgumentList @('ai_duel_webui.py','--host','%HOST%','--port','%PORT%') -WorkingDirectory '%cd%' -WindowStyle Hidden -RedirectStandardOutput '%OUT_LOG%' -RedirectStandardError '%ERR_LOG%' -PassThru; $p.Id | Out-File -Encoding ASCII '%PID_FILE%'"
if errorlevel 1 goto :fail

rem Wait for the backend to become ready (up to ~40s).
set "CODE=0"
for /l %%i in (1,1,40) do (
  for /f %%c in ('powershell -NoProfile -Command "try{(Invoke-WebRequest -UseBasicParsing '%URL%api/state' -TimeoutSec 2).StatusCode}catch{0}"') do set "CODE=%%c"
  if "!CODE!"=="200" goto :started
  powershell -NoProfile -Command "Start-Sleep -Seconds 1" >nul 2>nul
)

echo [ERROR] WebUI backend did not become ready.
if exist "%ERR_LOG%" (
  echo [INFO] Last error log lines:
  powershell -NoProfile -Command "Get-Content -Path '%ERR_LOG%' -Tail 120"
)
goto :fail

:started
echo [INFO] WebUI is running: %URL%
echo [INFO] First message is from you in the WebUI input box (no auto seed).
echo [INFO] To stop: click Stop in the WebUI, or run stop_webui.bat.
exit /b 0

:fail
echo [ERROR] Startup failed. See the messages above.
pause
exit /b 1
