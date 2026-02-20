@echo off
setlocal

rem Stop detached WebUI backend and related browser processes.
rem Robust cleanup order:
rem 1) PID from .tmp\webui.pid
rem 2) Any listener on port 8765
rem 3) Any python/pythonw commandline containing ai_duel_webui.py

cd /d "%~dp0"
chcp 65001 >nul

set "PORT=8765"
set "PID_FILE=%cd%\.tmp\webui.pid"
set "TMP_RES=%TEMP%\_webui_stop_result.txt"

del /q "%TMP_RES%" >nul 2>nul

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='SilentlyContinue';" ^
  "$killed=$false;" ^
  "$pidFile='%PID_FILE%';" ^
  "if(Test-Path $pidFile){ $txt=(Get-Content -Path $pidFile -Raw).Trim(); if($txt){ try{ Stop-Process -Id ([int]$txt) -Force -ErrorAction SilentlyContinue; $killed=$true }catch{} } };" ^
  "try{ $pids=Get-NetTCPConnection -LocalPort %PORT% -State Listen | Select-Object -ExpandProperty OwningProcess -Unique; foreach($p in $pids){ if($p){ try{ Stop-Process -Id $p -Force -ErrorAction SilentlyContinue; $killed=$true }catch{} } } }catch{};" ^
  "try{ Get-CimInstance Win32_Process | Where-Object { ($_.Name -match '^pythonw?\\.exe$') -and ($_.CommandLine -match 'ai_duel_webui\\.py') } | ForEach-Object { try{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $killed=$true }catch{} } }catch{};" ^
  "try{ Remove-Item -Path $pidFile -Force -ErrorAction SilentlyContinue }catch{};" ^
  "if($killed){ 'KILLED' } else { 'NONE' }" > "%TMP_RES%"

set "RES=NONE"
if exist "%TMP_RES%" (
  set /p RES=<"%TMP_RES%"
  del /q "%TMP_RES%" >nul 2>nul
)

if /I "%RES%"=="KILLED" (
  echo [INFO] Done.
) else (
  echo [INFO] No running WebUI backend found.
)
exit /b 0

