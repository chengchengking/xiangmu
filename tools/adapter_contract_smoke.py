from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model_adapters as ma


REQUIRED_METHODS = [
    "ensure_page",
    "goto_home",
    "bring_to_front",
    "close",
    "is_authenticated_now",
    "send_user_text",
    "snapshot_conversation",
    "wait_reply_and_extract",
]


def classify_adapter_error(exc: BaseException | str) -> str:
    text = str(exc).lower()
    if any(x in text for x in ["timeout", "timed out"]):
        return "timeout"
    if any(x in text for x in ["captcha", "验证", "人机"]):
        return "captcha"
    if any(x in text for x in ["stale", "detached", "execution context was destroyed"]):
        return "stale"
    if any(x in text for x in ["locator", "selector", "strict mode violation", "not found"]):
        return "selector_miss"
    return "unknown"


def static_contract_check() -> dict[str, Any]:
    classes = [
        ma.ChatGPTAdapter,
        ma.GeminiAdapter,
        ma.DeepSeekAdapter,
        ma.QwenAdapter,
        ma.DoubaoAdapter,
        ma.GenericWebChatAdapter,
    ]
    out: list[dict[str, Any]] = []
    ok = True
    for cls in classes:
        missing = [m for m in REQUIRED_METHODS if not callable(getattr(cls, m, None))]
        row = {"adapter": cls.__name__, "ok": not missing, "missing": missing}
        ok = ok and row["ok"]
        out.append(row)
    return {"ok": ok, "mode": "static", "results": out, "failure_classes": ["selector_miss", "timeout", "captcha", "stale", "unknown"]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Adapter contract smoke checks (static by default)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    result = static_contract_check()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"adapter_contract_smoke ok={result['ok']} mode={result['mode']}")
        for row in result["results"]:
            print(f"- {row['adapter']}: {'OK' if row['ok'] else 'MISSING ' + ','.join(row['missing'])}")
        print("failure classes:", ", ".join(result["failure_classes"]))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
