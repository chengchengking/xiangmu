from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


HOST = "127.0.0.1"
PORT = 8877
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
    with urllib.request.urlopen(req, timeout=10) as resp:
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
    out_log = TMP / "selftest_worker_runtime_api_smoke.out.log"
    err_log = TMP / "selftest_worker_runtime_api_smoke.err.log"
    for pth in (out_log, err_log):
        if pth.exists():
            pth.unlink()

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
        wr = st.get("worker_runtime")
        assert isinstance(wr, dict), f"worker_runtime missing/invalid: {type(wr)}"
        assert wr, "worker_runtime should contain model entries"
        one = next(iter(wr.values()))
        for k in ("state", "reason", "fail_count", "cooldown_remaining_s", "updated_ts"):
            assert k in one, f"worker_runtime entry missing key: {k}"

        rec = _api_post("/api/models/recover", {"key": "not_exist"})
        assert rec.get("ok") is False, rec
        assert rec.get("error") in {"unknown_or_not_integrated", "recover_timeout"}, rec

        print("OK: worker runtime api smoke passed")
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

