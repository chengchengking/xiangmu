from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


HOST = "127.0.0.1"
PORT = 8876
BASE = f"http://{HOST}:{PORT}"
TMP = Path(".tmp")


def _api_get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _api_post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _wait_ready(timeout_s: float = 25.0) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            data = _api_get("/api/state")
            if data.get("ok"):
                return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(0.4)
    raise RuntimeError(f"WebUI not ready within {timeout_s}s: {last_err}")


def main() -> int:
    TMP.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    out_log = TMP / "selftest_webui_api_smoke.out.log"
    err_log = TMP / "selftest_webui_api_smoke.err.log"
    if out_log.exists():
        out_log.unlink()
    if err_log.exists():
        err_log.unlink()

    p = subprocess.Popen(
        [sys.executable, "ai_duel_webui.py", "--host", HOST, "--port", str(PORT), "--no-open"],
        stdout=open(out_log, "w", encoding="utf-8"),
        stderr=open(err_log, "w", encoding="utf-8"),
        env=env,
    )
    try:
        _wait_ready()
        st = _api_get("/api/state")
        assert st.get("ok") is True, st

        # Smoke send: no model selected is expected in clean startup; API should accept enqueue.
        send = _api_post("/api/send", {"target": "group", "text": "smoke test message"})
        assert send.get("ok") is True, send

        # Worker should convert this into a visible system message complaining no models are selected.
        found = False
        deadline = time.time() + 8.0
        while time.time() < deadline:
            msgs = _api_get("/api/messages?after=0")
            for m in msgs.get("messages", []):
                txt = str(m.get("text") or "")
                if "没有已启用模型" in txt:
                    found = True
                    break
            if found:
                break
            time.sleep(0.4)
        assert found, "Expected system message for no_selected_models not observed"

        # Ensure UTF-8 log is readable (best-effort)
        if out_log.exists():
            _ = out_log.read_text(encoding="utf-8", errors="replace")

        print("OK: webui api smoke passed")
        return 0
    finally:
        try:
            _api_post("/api/stop", {})
        except Exception:
            pass
        try:
            p.terminate()
        except Exception:
            pass
        try:
            p.wait(timeout=6)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

