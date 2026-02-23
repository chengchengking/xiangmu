from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TMP_DIR = ROOT / ".tmp"


def api_get(base: str, path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base}{path}", timeout=8) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def api_post(base: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def wait_ready(base: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            st = api_get(base, "/api/state")
            if st.get("ok"):
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("WebUI not ready")


def _row_ts_unix(obj: dict[str, Any]) -> float | None:
    ts = obj.get("ts")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts)).timestamp()
    except Exception:
        return None


def iter_jsonl(path: Path, *, since_unix: float | None = None):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if since_unix is not None:
                    row_ts = _row_ts_unix(obj)
                    if row_ts is not None and row_ts < since_unix:
                        continue
                yield obj
            except Exception:
                continue


def _pick_files(pattern: str, since_unix: float) -> list[Path]:
    files = sorted(TMP_DIR.glob(pattern))
    return [p for p in files if p.exists() and p.stat().st_mtime >= since_unix]


def analyze_turn_logs(since_unix: float) -> dict[str, Any]:
    turn_files = _pick_files("ai_group_turns_*.jsonl", since_unix)
    msg_files = _pick_files("ai_group_messages_*.jsonl", since_unix)
    rows: list[dict[str, Any]] = []
    for f in turn_files:
        rows.extend(iter_jsonl(f, since_unix=since_unix))

    per_model_extract_elapsed: dict[str, list[float]] = defaultdict(list)
    per_model_receipt_status: dict[str, Counter] = defaultdict(Counter)
    per_model_reject_reason: dict[str, Counter] = defaultdict(Counter)
    packet_by_turn: dict[str, dict[str, str]] = defaultdict(dict)
    packet_mismatch_turns: list[str] = []
    evidence_turns: set[str] = set()

    turn_id_pat = re.compile(r"turn_id=(\d+)")
    packet_pat = re.compile(r"packet_hash=([0-9a-f]{6,40})", re.I)
    mode_pat = re.compile(r"mode=([A-Z_]+)")

    for r in rows:
        mk = str(r.get("model_key") or "")
        stage = str(r.get("stage") or "")
        txt = str(r.get("text") or "")
        if stage == "extract" and r.get("elapsed_s") is not None:
            try:
                per_model_extract_elapsed[mk].append(float(r["elapsed_s"]))
            except Exception:
                pass
        if stage in {"receipt_final", "receipt"}:
            try:
                payload = json.loads(txt)
                status = str(payload.get("status") or "")
                per_model_receipt_status[mk][status] += 1
                reason = str(payload.get("reason") or "")
                if reason:
                    per_model_reject_reason[mk][reason] += 1
            except Exception:
                pass
        if stage == "broadcast_packet":
            m_turn = turn_id_pat.search(txt)
            m_pkt = packet_pat.search(txt)
            if m_turn and m_pkt:
                tid = m_turn.group(1)
                ph = m_pkt.group(1)
                packet_by_turn[tid][mk] = ph
        if "mode=EVIDENCE" in txt:
            m_turn = turn_id_pat.search(txt)
            if m_turn:
                evidence_turns.add(m_turn.group(1))
        else:
            m = mode_pat.search(txt)
            if m and m.group(1) == "EVIDENCE":
                mt = turn_id_pat.search(txt)
                if mt:
                    evidence_turns.add(mt.group(1))

    for tid, mp in packet_by_turn.items():
        vals = {v for v in mp.values() if v}
        if len(vals) > 1:
            packet_mismatch_turns.append(tid)

    leaks = 0
    forbidden = ["【系统回执】", "[[META]]", "[[PUBLIC_REPLY]]", "[[PRIVATE_REPLY]]", "evidence_hook=", "【广播包 "]
    for f in msg_files:
        for rec in iter_jsonl(f, since_unix=since_unix):
            if rec.get("visibility") != "public":
                continue
            t = str(rec.get("text") or "")
            if any(tok in t for tok in forbidden):
                leaks += 1

    per_model = {}
    for mk in sorted(set(list(per_model_extract_elapsed) + list(per_model_receipt_status))):
        el = per_model_extract_elapsed.get(mk, [])
        rs = per_model_receipt_status.get(mk, Counter())
        total_receipts = sum(rs.values()) or 1
        per_model[mk] = {
            "extract_avg_s": round(sum(el) / len(el), 3) if el else None,
            "extract_count": len(el),
            "receipt_status": dict(rs),
            "pass_ratio": round(rs.get("PASS", 0) / total_receipts, 4),
            "reject_reason": dict(per_model_reject_reason.get(mk, Counter())),
        }

    return {
        "since_unix": since_unix,
        "turn_files": [str(p) for p in turn_files],
        "message_files": [str(p) for p in msg_files],
        "turn_rows": len(rows),
        "packet_turn_count": len(packet_by_turn),
        "packet_hash_mismatch_turns": sorted(packet_mismatch_turns, key=lambda x: int(x)),
        "evidence_turn_count": len(evidence_turns),
        "public_leak_hits": leaks,
        "per_model": per_model,
    }


def run_regression(base: str, rounds: int, wait_s: int) -> None:
    models = api_get(base, "/api/models")
    selected = [m for m in models.get("models", []) if m.get("selected")]
    if len(selected) < 2:
        raise RuntimeError("Need at least 2 selected models before running 20-round regression")

    topic = (
        "Regression task: discuss whether a central bank should cut rates in a supply shock. "
        "Give assumptions and one risk. Keep it concise but evidence-oriented."
    )
    topic_b64 = base64.b64encode(topic.encode("utf-8")).decode("ascii")
    api_post(base, "/api/send", {"target": "group", "text_b64": topic_b64, "rounds": max(1, int(rounds))})
    time.sleep(3)
    interject = (
        "Host interjection: switch topic. Now solve 144/12 and explain how the topic switch should be handled."
    )
    interject_b64 = base64.b64encode(interject.encode("utf-8")).decode("ascii")
    api_post(base, "/api/send", {"target": "group", "text_b64": interject_b64, "rounds": 1})

    deadline = time.time() + wait_s
    while time.time() < deadline:
        st = api_get(base, "/api/state")
        if str(st.get("status")) == "idle":
            return
        time.sleep(1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="20-round group chat regression runner + metrics")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--wait-seconds", type=int, default=180)
    ap.add_argument("--start-server", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    server_proc = None
    since_unix = time.time() - 1

    try:
        if args.start_server:
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            server_proc = subprocess.Popen(
                [sys.executable, "ai_duel_webui.py", "--host", args.host, "--port", str(args.port), "--no-open"],
                cwd=str(ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
        wait_ready(base)
        if not args.analyze_only:
            run_regression(base, rounds=args.rounds, wait_s=args.wait_seconds)
        report = analyze_turn_logs(since_unix)
        out = TMP_DIR / f"group_regression_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"report_file={out}")
        if report["public_leak_hits"] > 0:
            return 2
        return 0
    finally:
        if server_proc is not None:
            try:
                api_post(base, "/api/stop", {})
            except Exception:
                pass
            try:
                server_proc.terminate()
                server_proc.wait(timeout=5)
            except Exception:
                try:
                    server_proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
