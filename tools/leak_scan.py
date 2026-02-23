from __future__ import annotations

import argparse
import json
from pathlib import Path


FORBIDDEN = [
    "【系统回执】",
    "[[META]]",
    "[[/META]]",
    "[[PUBLIC_REPLY]]",
    "[[/PUBLIC_REPLY]]",
    "[[PRIVATE_REPLY]]",
    "[[/PRIVATE_REPLY]]",
    "evidence_hook=",
    "【广播包 ",
    "packet_hash=",
    "ack_in=",
]


def _scan_text(text: str) -> list[str]:
    hits: list[str] = []
    raw = text or ""
    for tok in FORBIDDEN:
        if tok in raw:
            hits.append(tok)
    return hits


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for ln_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ln_no, json.loads(line)
            except Exception:
                continue


def _pick_logs(pattern: str, tmp_dir: Path, all_files: bool, since_unix: float | None) -> list[Path]:
    files = sorted(tmp_dir.glob(pattern))
    if since_unix is not None:
        files = [p for p in files if p.exists() and p.stat().st_mtime >= since_unix]
    if all_files:
        return files
    return files[-1:] if files else []


def scan_message_logs(tmp_dir: Path, *, all_files: bool, since_unix: float | None) -> list[str]:
    issues: list[str] = []
    for path in _pick_logs("ai_group_messages_*.jsonl", tmp_dir, all_files, since_unix):
        for ln, obj in _iter_jsonl(path):
            if obj.get("visibility") != "public":
                continue
            text = str(obj.get("text") or "")
            hits = _scan_text(text)
            if hits:
                issues.append(f"{path}:{ln}: public-message leak tokens={hits!r}")
    return issues


def scan_turn_logs(tmp_dir: Path, *, all_files: bool, since_unix: float | None) -> list[str]:
    issues: list[str] = []
    for path in _pick_logs("ai_group_turns_*.jsonl", tmp_dir, all_files, since_unix):
        for ln, obj in _iter_jsonl(path):
            if str(obj.get("stage") or "") != "final":
                continue
            payload = str(obj.get("text") or obj.get("payload") or "")
            # final trace often contains a header line "turn_id=... mode=..." which is internal trace, not user timeline.
            # Strip only the first line if it looks like trace metadata.
            if payload.startswith("turn_id="):
                payload = "\n".join(payload.splitlines()[1:])
            hits = _scan_text(payload)
            if hits:
                issues.append(f"{path}:{ln}: final-trace leak tokens={hits!r}")
    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan public timeline and final traces for control-plane leakage")
    ap.add_argument("--tmp-dir", default=".tmp")
    ap.add_argument("--all", action="store_true", help="scan all historical logs (default: latest files only)")
    ap.add_argument("--since-unix", type=float, default=None, help="only scan files modified at/after this unix ts")
    args = ap.parse_args()
    tmp_dir = Path(args.tmp_dir)

    issues = []
    issues.extend(scan_message_logs(tmp_dir, all_files=args.all, since_unix=args.since_unix))
    issues.extend(scan_turn_logs(tmp_dir, all_files=args.all, since_unix=args.since_unix))

    if issues:
        print("LEAK_SCAN_FAIL")
        for item in issues[:200]:
            print(item)
        if len(issues) > 200:
            print(f"... truncated {len(issues)-200} more issues")
        return 1
    print("LEAK_SCAN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
