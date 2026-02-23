# xiangmu

## Quick Start

### One-click (Windows)

Double click:

```text
start_webui.bat
```

It will (best-effort) install Python deps, install Playwright Chromium, then start the WebUI backend **detached** (in background).

- Logs: `.tmp\webui.out.log` / `.tmp\webui.err.log`
- Stop: click **Stop** in the WebUI, or double click `stop_webui.bat`

If you see `Failed to fetch` in the UI, it usually means the backend is not running. Re-run `start_webui.bat` and check `.tmp\webui.err.log`.

### Manual

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
python .\ai_duel_webui.py
```

## Web UI (3-person group chat)

If you don't want to stare at the two web pages, use the local Web UI:

```powershell
python .\ai_duel_webui.py
```

It will start a local page at `http://127.0.0.1:8765/` and open it in your browser. You can send messages as the "host" and watch ChatGPT/Gemini replies in one place.

On Windows, it will *prefer* opening the UI as a standalone app window (Chrome/Edge `--app=` mode) instead of a normal tab. If not available, it falls back to the default browser.

Note: the WebUI does not auto-send a seed message. The first message is your first line in the WebUI input box.

### Model Slots (1..10)

The WebUI has a left model bar (slots `1..10`):

- 3-state:
  - Locked: not integrated yet (grey + lock)
  - Available: integrated but not selected (grey)
  - Selected: enabled (color)
- Green dot means authenticated (login detected).
- Click a slot to toggle join/leave group chat. If not authenticated, it will show a login modal:
  - **Open login window** (official site)
  - **Recheck** after you finish manual login
- Send requires explicit target selection in the bottom dropdown:
  - `???` / `?? (public)` / `?? (shadow, only selected models)`

Integrated models (current):

- `1=ChatGPT` `2=Gemini` `3=DeepSeek`
- `4=??` `5=Qwen` `6=Kimi` (generic adapter template; may need selector tweaks)
- `7..10` are locked placeholders

### Model Equality

All selected models are treated equally. Use `??` to send to all selected models, or `??` to talk to one model.

Shadow semantics:

- `shadow_shared`: stored as `shadow` and may be synchronized to other selected models (compatibility mode / broader context sharing).
- `shadow_local`: stored as `shadow` but kept local to the target model (useful for game-like/private strategies; not synchronized to peers).

The default shadow behavior can be configured with `AI_DUEL_SHADOW_SCOPE_DEFAULT=shared|local`.

## Notes

- `install_playwright_chromium_fast.ps1` will probe multiple download hosts (including `npmmirror`) and pick the fastest for Chromium.
- If download errors like `stream disconnected before completion: ... error decoding response body` occur, the script will back off and retry automatically.
- `ai_duel.py` uses a persistent Playwright profile directory at `.\user_data\` so you can log in once and reuse the session.


## Docs

- Architecture (Frozen v1, root copy): `GROUPCHAT_ARCHITECTURE.md`
- Architecture (docs copy): `docs/architecture/GROUPCHAT_ARCHITECTURE.md`
