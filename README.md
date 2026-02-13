# xiangmu

## Quick Start

### One-click (Windows)

Double click:

```text
start_webui.bat
```

It will (best-effort) install Python deps, install Playwright Chromium, then start the WebUI.

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

Note: the WebUI does not auto-send a seed message. The first message is your first line in the WebUI input box.

### Session Start (Model Slots 1..10)

The WebUI now has a left model bar (slots `1..10`):

- Click a slot to enable/disable it (selected = in use; grayscale = off; disabled = not integrated yet).
- Click **启动** to start the automation. The script will only open the model web pages after you click **启动**.
- Current version only integrates `1=ChatGPT` and `2=Gemini` (other slots are placeholders).

### Model Equality (Foreground / Background)

In the Web UI dropdown, the two models are treated as equals:

- If you choose **foreground ChatGPT**, Gemini will still be synced in the background and can generate replies, but the UI will hide Gemini messages until you switch to foreground Gemini. The sidebar shows unread counts: `未读：ChatGPT X | Gemini Y`.
- If you choose **foreground Gemini**, ChatGPT behaves the same way.

## Notes

- `install_playwright_chromium_fast.ps1` will probe multiple download hosts (including `npmmirror`) and pick the fastest for Chromium.
- If download errors like `stream disconnected before completion: ... error decoding response body` occur, the script will back off and retry automatically.
- `ai_duel.py` uses a persistent Playwright profile directory at `.\user_data\` so you can log in once and reuse the session.
