from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List


def norm(text: str) -> str:
    return (text or "").replace("\u200b", "").strip()


def line_key(line: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", norm(line).lower())


class HttpApi:
    def __init__(self, base: str, timeout: float = 80.0) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> Dict[str, Any]:
        url = self.base + path
        req = urllib.request.Request(url=url, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)

    def post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base + path
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            method="POST",
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)


@dataclass
class CaseResult:
    model_key: str
    round_no: int
    ok: bool
    reason: str
    preview: str


def get_models_map(api: HttpApi) -> Dict[str, Dict[str, Any]]:
    data = api.get("/api/models")
    models = data.get("models") or []
    return {str(m.get("key") or "").strip().lower(): m for m in models}


def latest_message_id(api: HttpApi) -> int:
    data = api.get("/api/messages?after=0")
    msgs = data.get("messages") or []
    if not msgs:
        return 0
    return max(int(x.get("id") or 0) for x in msgs)


def ensure_model_authenticated(api: HttpApi, key: str) -> None:
    box = api.post("/api/models/login/check", {"key": key})
    if not box.get("authenticated"):
        raise RuntimeError(f"{key} 未登录或登录态失效，请先在 WebUI 登录并检测。")


def ensure_model_selected(api: HttpApi, key: str, want: bool) -> None:
    key = key.strip().lower()
    models = get_models_map(api)
    m = models.get(key)
    if not m:
        raise RuntimeError(f"未知模型: {key}")
    if not m.get("integrated"):
        raise RuntimeError(f"模型未接入自动化: {key}")

    selected = bool(m.get("selected"))
    if selected == want:
        return

    if want and not m.get("authenticated"):
        ensure_model_authenticated(api, key)

    box = api.post("/api/models/toggle", {"key": key})
    if box.get("need_auth"):
        ensure_model_authenticated(api, key)
        box = api.post("/api/models/toggle", {"key": key})
    if not box.get("ok"):
        raise RuntimeError(f"切换模型失败: {key} -> {box}")


def select_only(api: HttpApi, keys: List[str]) -> None:
    wanted = {k.strip().lower() for k in keys if k.strip()}
    models = get_models_map(api)
    for key, m in models.items():
        if not m.get("integrated"):
            continue
        cur = bool(m.get("selected"))
        if key in wanted:
            if not cur:
                ensure_model_selected(api, key, True)
        else:
            if cur:
                ensure_model_selected(api, key, False)


def wait_model_reply(api: HttpApi, *, after_id: int, key: str, timeout_s: int) -> Dict[str, Any]:
    key = key.strip().lower()
    deadline = time.time() + timeout_s
    last_seen_id = after_id
    best: Dict[str, Any] | None = None
    best_seen_ts = 0.0

    while time.time() < deadline:
        data = api.get(f"/api/messages?after={last_seen_id}")
        msgs = data.get("messages") or []
        for msg in msgs:
            mid = int(msg.get("id") or 0)
            if mid > last_seen_id:
                last_seen_id = mid
            if str(msg.get("role") or "") != "model":
                continue
            if str(msg.get("model_key") or "").strip().lower() != key:
                continue
            best = msg
            best_seen_ts = time.time()

        state = api.get("/api/state")
        status = str(state.get("status") or "")
        if best and status == "idle" and (time.time() - best_seen_ts) >= 1.0:
            return best
        time.sleep(1.0)

    return best or {}


def analyze_reply(model_key: str, text: str) -> tuple[bool, str]:
    t = norm(text)
    if not t:
        return False, "empty"
    if t.startswith("（未能提取到回复"):
        return False, "extract_failed_placeholder"

    low = t.lower()
    exact_bad = {"思考", "思考中", "正在思考", "已完成思考", "pass", "[pass]", "跳过"}
    if norm(low) in exact_bad:
        return False, "thought_or_pass_only"
    for ln in [x.strip().lower() for x in t.splitlines() if x.strip()]:
        if ln in exact_bad:
            return False, "thought_line_leak"

    bad_phrases = [
        "群主刚刚插话：",
        "请优先回应这条插话",
        "这是群聊第",
        "可点名对象：",
        "如果你想加入当前讨论",
        "用户现在需要回应群主的插话",
        "深度思考中",
        "正在思考",
        "跳过思考",
    ]
    if model_key == "doubao":
        bad_phrases.extend(["已完成思考", "整理一下，简洁自然"])

    hit = [x for x in bad_phrases if x in t]
    if hit:
        return False, "prompt_or_thought_leak"

    compact = line_key(t)
    if len(compact) < 16:
        return False, "too_short"

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if model_key == "doubao":
        intro_like = [ln for ln in lines if ln.startswith("我是") and len(line_key(ln)) >= 10]
        if len(intro_like) >= 2:
            return False, "multi_intro_block"
    if len(lines) >= 4:
        keys = [line_key(x) for x in lines if line_key(x)]
        if keys:
            unique = len(set(keys))
            if unique <= max(1, len(keys) // 2):
                return False, "heavy_repetition"

    return True, "ok"


def run_single_rounds(api: HttpApi, model_key: str, rounds: int, timeout_s: int) -> List[CaseResult]:
    out: List[CaseResult] = []
    for i in range(1, rounds + 1):
        marker = f"SELFTEST-{model_key}-{int(time.time())}-{i}"
        prompt = (
            f"{marker}\n"
            "请你只输出正式回复正文，用两段中文：\n"
            "第一段：20-60字自我介绍。\n"
            "第二段：给一个可执行建议，并给出一个追问。\n"
            "不要输出“思考中/已完成思考/PASS/跳过”。"
        )
        start_id = latest_message_id(api)
        api.post("/api/send", {"target": model_key, "text": prompt})
        msg = wait_model_reply(api, after_id=start_id, key=model_key, timeout_s=timeout_s)
        text = norm(str(msg.get("text") or ""))
        ok, reason = analyze_reply(model_key, text)
        preview = text[:180].replace("\n", " / ")
        out.append(CaseResult(model_key=model_key, round_no=i, ok=ok, reason=reason, preview=preview))
    return out


def run_group_smoke(api: HttpApi, keys: List[str], timeout_s: int) -> CaseResult:
    start_id = latest_message_id(api)
    prompt = "SELFTEST-GROUP: 请两位先各给一个观点，再互相反驳一句。"
    api.post("/api/send", {"target": "group", "text": prompt, "rounds": 2})

    deadline = time.time() + timeout_s
    seen: Dict[str, str] = {}
    last_id = start_id
    while time.time() < deadline:
        data = api.get(f"/api/messages?after={last_id}")
        msgs = data.get("messages") or []
        for msg in msgs:
            mid = int(msg.get("id") or 0)
            if mid > last_id:
                last_id = mid
            if str(msg.get("role") or "") != "model":
                continue
            if str(msg.get("visibility") or "") != "public":
                continue
            mk = str(msg.get("model_key") or "").strip().lower()
            if mk in keys and mk not in seen:
                seen[mk] = norm(str(msg.get("text") or ""))
        state = api.get("/api/state")
        if state.get("status") == "idle" and len(seen) >= len(keys):
            break
        time.sleep(1.0)

    missing = [k for k in keys if k not in seen]
    if missing:
        return CaseResult(
            model_key="group",
            round_no=1,
            ok=False,
            reason=f"missing_public_reply:{','.join(missing)}",
            preview="",
        )
    return CaseResult(model_key="group", round_no=1, ok=True, reason="ok", preview="group smoke passed")


def main() -> int:
    ap = argparse.ArgumentParser(description="WebUI capture self-test runner")
    ap.add_argument("--base-url", default="http://127.0.0.1:8765", help="WebUI base URL")
    ap.add_argument("--models", default="doubao,qwen", help="Comma-separated model keys for single-chat tests")
    ap.add_argument("--rounds", type=int, default=2, help="Single-chat rounds per model")
    ap.add_argument("--timeout", type=int, default=220, help="Timeout per round (seconds)")
    ap.add_argument("--group-smoke", action="store_true", help="Run extra group-chat smoke test using first two models")
    args = ap.parse_args()

    api = HttpApi(args.base_url, timeout=85.0)
    try:
        st = api.get("/api/state")
    except (urllib.error.URLError, ConnectionError) as exc:
        print(f"[FAIL] cannot reach WebUI: {exc}")
        return 2

    print(f"[INFO] state: {st}")
    models = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    if not models:
        print("[FAIL] no models specified")
        return 2

    current = get_models_map(api)
    for key in models:
        m = current.get(key)
        if not m:
            raise RuntimeError(f"未知模型: {key}")
        if not m.get("authenticated"):
            ensure_model_authenticated(api, key)

    select_only(api, [])
    all_results: List[CaseResult] = []

    for key in models:
        print(f"[INFO] testing model: {key}")
        select_only(api, [key])
        rs = run_single_rounds(api, key, rounds=max(1, args.rounds), timeout_s=max(60, args.timeout))
        all_results.extend(rs)

    if args.group_smoke and len(models) >= 2:
        print(f"[INFO] group smoke: {models[:2]}")
        select_only(api, models[:2])
        all_results.append(run_group_smoke(api, models[:2], timeout_s=max(300, args.timeout * 2)))

    bad = [r for r in all_results if not r.ok]
    for r in all_results:
        tag = "PASS" if r.ok else "FAIL"
        print(f"[{tag}] {r.model_key}#{r.round_no} {r.reason}")
        if r.preview:
            print(f"       {r.preview}")

    summary = {
        "total": len(all_results),
        "passed": len(all_results) - len(bad),
        "failed": len(bad),
        "failed_items": [f"{x.model_key}#{x.round_no}:{x.reason}" for x in bad],
    }
    print("[SUMMARY] " + json.dumps(summary, ensure_ascii=False))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
