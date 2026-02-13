# xiangmu

## Quick Start

1. Install Python deps:

```powershell
python -m pip install -r requirements.txt
```

2. Install Playwright browsers (Chromium/ffmpeg/winldd) with mirror probing + retry:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_playwright_chromium_fast.ps1
```

3. Run the demo (opens visible browser windows):

```powershell
python .\ai_duel.py
```

## Notes

- `install_playwright_chromium_fast.ps1` will probe multiple download hosts (including `npmmirror`) and pick the fastest for Chromium.
- If download errors like `stream disconnected before completion: ... error decoding response body` occur, the script will back off and retry automatically.
- `ai_duel.py` uses a persistent Playwright profile directory at `.\user_data\` so you can log in once and reuse the session.

