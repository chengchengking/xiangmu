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

echo [INFO] Starting WebUI (visible browser windows)...
echo [INFO] First message is from you in the WebUI input box (no auto seed).
python ai_duel_webui.py --host 127.0.0.1 --port 8765
exit /b 0

:fail
echo [ERROR] Startup failed. See the messages above.
pause
exit /b 1

