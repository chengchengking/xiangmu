from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ai_duel_webui as webui
import model_adapters as ma
from playwright.sync_api import sync_playwright


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
    if any(x in text for x in ["launch_persistent_context", "browsertype.launch", "browser has been closed"]):
        return "stale"
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


def _pick_metas(keys: list[str] | None, include_disabled: bool = False) -> list[Any]:
    wanted = {str(k).strip().lower() for k in (keys or []) if str(k).strip()}
    metas = []
    for meta in webui.MODEL_METAS:
        if (not include_disabled) and (not getattr(meta, "integrated", False)):
            continue
        if wanted and str(meta.key).strip().lower() not in wanted:
            continue
        metas.append(meta)
    return metas


def live_probe(
    *,
    model_keys: list[str] | None = None,
    timeout_s: float = 20.0,
    open_home: bool = True,
    include_disabled: bool = False,
) -> dict[str, Any]:
    metas = _pick_metas(model_keys, include_disabled=include_disabled)
    if not metas:
        return {
            "ok": False,
            "mode": "live",
            "error": "no_models_selected",
            "results": [],
            "failure_classes": ["selector_miss", "timeout", "captcha", "stale", "unknown"],
        }

    out: list[dict[str, Any]] = []
    overall_ok = True
    start_ts = time.time()

    with sync_playwright() as pw:
        for meta in metas:
            ad = None
            t0 = time.time()
            row: dict[str, Any] = {
                "key": meta.key,
                "name": meta.name,
                "url": meta.url,
                "ok": False,
                "class": None,
                "steps": [],
                "authenticated": None,
                "elapsed_s": None,
            }
            try:
                ad = webui.build_adapter(meta)
                row["steps"].append("build_adapter")
                ad.ensure_page(pw)
                row["steps"].append("ensure_page")
                if open_home:
                    try:
                        ad.goto_home()
                        row["steps"].append("goto_home")
                    except Exception as exc:  # noqa: BLE001
                        row["steps"].append(f"goto_home_fail:{classify_adapter_error(exc)}")
                try:
                    authed = bool(ad.is_authenticated_now())
                    row["authenticated"] = authed
                    row["steps"].append("auth_check")
                except Exception as exc:  # noqa: BLE001
                    cls = classify_adapter_error(exc)
                    row["class"] = cls
                    row["error"] = str(exc)
                    row["steps"].append(f"auth_check_fail:{cls}")
                    raise
                row["ok"] = True
                row["class"] = "ok"
            except Exception as exc:  # noqa: BLE001
                cls = classify_adapter_error(exc)
                row["class"] = cls
                row["error"] = str(exc)
                overall_ok = False
            finally:
                row["elapsed_s"] = round(time.time() - t0, 3)
                out.append(row)
                if ad is not None:
                    try:
                        ad.close()
                    except Exception:
                        pass
            if timeout_s > 0 and (time.time() - start_ts) > timeout_s:
                out.append(
                    {
                        "key": "_global",
                        "ok": False,
                        "class": "timeout",
                        "error": f"global live probe timeout>{timeout_s}s",
                        "elapsed_s": round(time.time() - start_ts, 3),
                    }
                )
                overall_ok = False
                break

    return {
        "ok": overall_ok,
        "mode": "live",
        "results": out,
        "failure_classes": ["selector_miss", "timeout", "captcha", "stale", "unknown"],
        "count": len(out),
        "elapsed_s": round(time.time() - start_ts, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Adapter contract smoke checks (static by default)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--live", action="store_true", help="Run real Playwright adapter probes (ensure page + auth check)")
    ap.add_argument("--models", default="", help="Comma-separated model keys, e.g. chatgpt,gemini,qwen")
    ap.add_argument("--timeout", type=float, default=60.0, help="Global timeout for live probe")
    ap.add_argument("--no-goto-home", action="store_true", help="Skip adapter.goto_home() during live probe")
    ap.add_argument("--include-disabled", action="store_true", help="Include non-integrated placeholder models")
    args = ap.parse_args()
    if args.live:
        model_keys = [x.strip() for x in str(args.models or "").split(",") if x.strip()]
        result = live_probe(
            model_keys=model_keys or None,
            timeout_s=float(args.timeout or 60.0),
            open_home=not args.no_goto_home,
            include_disabled=bool(args.include_disabled),
        )
    else:
        result = static_contract_check()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"adapter_contract_smoke ok={result['ok']} mode={result['mode']}")
        if result["mode"] == "static":
            for row in result["results"]:
                print(f"- {row['adapter']}: {'OK' if row['ok'] else 'MISSING ' + ','.join(row['missing'])}")
        else:
            for row in result["results"]:
                key = row.get("key") or row.get("adapter") or "?"
                if row.get("ok"):
                    print(f"- {key}: OK auth={row.get('authenticated')} elapsed={row.get('elapsed_s')}s")
                else:
                    print(f"- {key}: {row.get('class','unknown')} elapsed={row.get('elapsed_s')}s err={row.get('error','')}")
        print("failure classes:", ", ".join(result["failure_classes"]))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
