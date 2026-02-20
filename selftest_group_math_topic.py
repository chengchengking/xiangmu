from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def norm(text: str) -> str:
    return (text or "").replace("\u200b", "").strip()


class Api:
    def __init__(self, base: str, timeout: float = 90.0) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> dict[str, Any]:
        req = urllib.request.Request(self.base + path, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.base + path,
            method="POST",
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
            return json.loads(resp.read().decode("utf-8", errors="replace"))


@dataclass
class Case:
    name: str
    prompt: str
    expected: int


@dataclass
class ModelCaseResult:
    case: str
    model_key: str
    speaker: str
    ok: bool
    expected: int
    got_numbers: list[int]
    latency_s: float
    text: str
    reason: str


def latest_message_id(api: Api) -> int:
    data = api.get("/api/messages?after=0")
    msgs = data.get("messages") or []
    if not msgs:
        return 0
    return max(int(m.get("id") or 0) for m in msgs)


def wait_idle(api: Api, timeout_s: int = 120) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = api.get("/api/state")
        if str(st.get("status") or "") == "idle":
            return True
        time.sleep(0.5)
    return False


def ensure_selected_models(api: Api, target_keys: list[str]) -> list[str]:
    keys = [k.strip().lower() for k in target_keys if k.strip()]
    wanted = set(keys)
    data = api.get("/api/models")
    models = {str(m.get("key") or "").strip().lower(): m for m in (data.get("models") or [])}

    usable: list[str] = []
    for key in keys:
        m = models.get(key)
        if not m:
            print(f"[WARN] unknown model key: {key}")
            continue
        if not m.get("integrated"):
            print(f"[WARN] model not integrated: {key}")
            continue
        if not m.get("authenticated"):
            print(f"[WARN] model not authenticated, skip: {key}")
            continue
        usable.append(key)

    # Keep test deterministic: only selected target models remain enabled.
    for key, m in models.items():
        if not m.get("integrated"):
            continue
        is_selected = bool(m.get("selected"))
        should_select = key in wanted and key in usable
        if is_selected == should_select:
            continue
        box = api.post("/api/models/toggle", {"key": key})
        if not box.get("ok") and not box.get("need_auth"):
            raise RuntimeError(f"toggle failed for {key}: {box}")

    return usable


def parse_ints(text: str) -> list[int]:
    vals: list[int] = []
    for tok in re.findall(r"-?\d+", norm(text)):
        try:
            vals.append(int(tok))
        except Exception:
            continue
    return vals


def collect_case_replies(
    api: Api,
    *,
    selected_keys: list[str],
    case: Case,
    rounds: int,
    timeout_s: int,
    trace_jsonl: Path,
) -> list[ModelCaseResult]:
    if not selected_keys:
        return []

    wait_idle(api, timeout_s=60)
    after_id = latest_message_id(api)
    sent_at = time.time()
    payload = {"target": "group", "text": case.prompt, "rounds": max(1, int(rounds))}
    api.post("/api/send", payload)

    pending = set(selected_keys)
    results: dict[str, ModelCaseResult] = {}
    cursor = after_id
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        data = api.get(f"/api/messages?after={cursor}")
        msgs = data.get("messages") or []
        for msg in msgs:
            mid = int(msg.get("id") or 0)
            if mid > cursor:
                cursor = mid
            row = dict(msg)
            row["case"] = case.name
            row["expected"] = case.expected
            trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with trace_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if str(msg.get("role") or "") != "model":
                continue
            if str(msg.get("visibility") or "") != "public":
                continue
            mk = str(msg.get("model_key") or "").strip().lower()
            if mk not in pending:
                continue
            text = norm(str(msg.get("text") or ""))
            nums = parse_ints(text)
            ok = case.expected in nums
            reason = "ok" if ok else ("no_number" if not nums else "wrong_number")
            results[mk] = ModelCaseResult(
                case=case.name,
                model_key=mk,
                speaker=str(msg.get("speaker") or mk),
                ok=ok,
                expected=case.expected,
                got_numbers=nums,
                latency_s=max(0.0, time.time() - sent_at),
                text=text,
                reason=reason,
            )
            pending.discard(mk)

        st = api.get("/api/state")
        if not pending and str(st.get("status") or "") == "idle":
            break
        if not pending:
            # Give one short settle window to avoid capturing stale late updates.
            time.sleep(0.8)
            st2 = api.get("/api/state")
            if str(st2.get("status") or "") == "idle":
                break
        time.sleep(0.6)

    out: list[ModelCaseResult] = []
    for key in selected_keys:
        if key in results:
            out.append(results[key])
            continue
        out.append(
            ModelCaseResult(
                case=case.name,
                model_key=key,
                speaker=key,
                ok=False,
                expected=case.expected,
                got_numbers=[],
                latency_s=float(timeout_s),
                text="",
                reason="missing_reply",
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic group-chat selftest (math + topic switch)")
    ap.add_argument("--base", default="http://127.0.0.1:8765", help="webui base url")
    ap.add_argument(
        "--models",
        default="chatgpt,gemini,deepseek,doubao,qwen",
        help="comma-separated model keys",
    )
    ap.add_argument("--rounds", type=int, default=1, help="group rounds per test case")
    ap.add_argument("--timeout", type=int, default=180, help="timeout seconds per test case")
    args = ap.parse_args()

    api = Api(args.base)
    tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(".tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = out_dir / f"selftest_math_messages_{tag}.jsonl"
    report_json = out_dir / f"selftest_math_report_{tag}.json"

    try:
        st = api.get("/api/state")
    except (urllib.error.URLError, ConnectionError) as exc:
        print(f"[FAIL] webui unavailable: {exc}")
        return 2
    print(f"[INFO] state={st}")

    model_keys = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    selected = ensure_selected_models(api, model_keys)
    if not selected:
        print("[FAIL] no usable selected models")
        return 2
    print(f"[INFO] selected={selected}")

    cases = [
        Case("math_a", "SELFTEST-MATH-A: New topic. Reply with only one integer. 1+1=?", 2),
        Case("math_b", "SELFTEST-MATH-B: Switch topic now. Reply with only one integer. 9+6=?", 15),
        Case("math_c", "SELFTEST-MATH-C: Switch topic again. Reply with only one integer. 100-37=?", 63),
        Case("math_d", "SELFTEST-MATH-D: Final switch. Reply with only one integer. 8*7=?", 56),
    ]

    all_rows: list[ModelCaseResult] = []
    for case in cases:
        print(f"[CASE] {case.name} expected={case.expected}")
        rows = collect_case_replies(
            api,
            selected_keys=selected,
            case=case,
            rounds=args.rounds,
            timeout_s=max(60, int(args.timeout)),
            trace_jsonl=trace_jsonl,
        )
        all_rows.extend(rows)
        for r in rows:
            tag_ok = "PASS" if r.ok else "FAIL"
            preview = (r.text or "").replace("\n", " / ")[:160]
            print(
                f"  [{tag_ok}] {r.model_key:<8} latency={r.latency_s:5.1f}s "
                f"got={r.got_numbers} reason={r.reason} preview={preview}"
            )

    bad = [r for r in all_rows if not r.ok]
    by_model: dict[str, dict[str, Any]] = {}
    for mk in selected:
        rows = [r for r in all_rows if r.model_key == mk]
        by_model[mk] = {
            "total": len(rows),
            "passed": len([x for x in rows if x.ok]),
            "failed": len([x for x in rows if not x.ok]),
            "avg_latency_s": round(sum(x.latency_s for x in rows) / max(1, len(rows)), 3),
        }

    report = {
        "tag": tag,
        "base": args.base,
        "selected_models": selected,
        "cases": [asdict(c) for c in cases],
        "summary": {
            "total_checks": len(all_rows),
            "passed": len(all_rows) - len(bad),
            "failed": len(bad),
            "by_model": by_model,
        },
        "failures": [asdict(x) for x in bad],
        "rows": [asdict(x) for x in all_rows],
        "message_trace_jsonl": str(trace_jsonl.resolve()),
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] report={report_json.resolve()}")
    print(f"[INFO] message_trace={trace_jsonl.resolve()}")
    print(
        "[SUMMARY] "
        + json.dumps(
            {
                "total": len(all_rows),
                "passed": len(all_rows) - len(bad),
                "failed": len(bad),
            },
            ensure_ascii=False,
        )
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
