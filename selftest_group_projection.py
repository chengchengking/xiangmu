from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any


def norm(text: str) -> str:
    return (text or "").replace("\u200b", "").strip()


def line_key(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", norm(text).lower())


class Api:
    def __init__(self, base: str, timeout: float = 60.0) -> None:
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


LEAK_HINTS = (
    "这是群聊第",
    "可回应对象：",
    "可点名对象：",
    "请基于这条消息继续群聊",
    "请先回应群主插话",
    "如果你这轮不想发言",
    "只输出 [PASS]",
    "上下文边界",
    "忽略网页里更早的旧对话",
    "用户现在需要",
    "首先，组织语言",
    "整理成自然",
)

HOST_CHATTER_PAT = re.compile(
    r"(?:好(?:的|嘞)?|收到|明白|ok|OK|行(?:吧)?)[，,。!！~～\s]{0,3}(?:群主|主持人|老大|老板)|"
    r"(谁先来|我们这就开始|先来(?:出)?第一个|我先来抛砖引玉)",
    re.I,
)

PASS_PAT = re.compile(r"^\s*(?:\[?\s*pass\s*\]?|跳过|旁听|继续旁听|已完成|已经完成|思考|思考中)\s*$", re.I)


def quality_check(text: str) -> tuple[bool, str]:
    t = norm(text)
    if not t:
        return False, "empty"
    if PASS_PAT.match(t):
        return False, "pass_or_status_only"
    if any(h in t for h in LEAK_HINTS):
        return False, "prompt_leak"
    if HOST_CHATTER_PAT.search(t) and len(line_key(t)) <= 56:
        return False, "host_chatter"

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        keys = [line_key(x) for x in lines if line_key(x)]
        if len(keys) >= 4:
            uniq = len(set(keys))
            if uniq <= max(1, len(keys) // 2):
                return False, "heavy_repetition"
    return True, "ok"


def latest_message_id(api: Api) -> int:
    box = api.get("/api/messages?after=0")
    msgs = box.get("messages") or []
    if not msgs:
        return 0
    return max(int(m.get("id") or 0) for m in msgs)


def wait_idle(api: Api, timeout_s: int) -> None:
    end = time.time() + timeout_s
    while time.time() < end:
        st = api.get("/api/state")
        if str(st.get("status") or "") == "idle":
            return
        time.sleep(0.4)


def ensure_selected(api: Api, keys: list[str]) -> None:
    target = {k.strip().lower() for k in keys if k.strip()}
    box = api.get("/api/models")
    models = {str(m.get("key") or "").strip().lower(): m for m in (box.get("models") or [])}

    for key, m in models.items():
        if not m.get("integrated"):
            continue
        is_selected = bool(m.get("selected"))
        if key in target:
            if not m.get("authenticated"):
                raise RuntimeError(f"{key} 未登录，无法自动测试。")
            if not is_selected:
                res = api.post("/api/models/toggle", {"key": key})
                if not res.get("ok"):
                    raise RuntimeError(f"启用模型失败: {key} -> {res}")
        else:
            if is_selected:
                api.post("/api/models/toggle", {"key": key})


@dataclass
class RoundStat:
    round_no: int
    model_key: str
    latency_s: float
    ok: bool
    reason: str
    preview: str


def run_round(api: Api, keys: list[str], prompt: str, timeout_s: int) -> list[RoundStat]:
    # Ensure previous loop is settled.
    wait_idle(api, timeout_s=15)

    after_id = latest_message_id(api)
    sent_at = time.time()
    api.post("/api/send", {"target": "group", "text": prompt, "rounds": 1})

    seen: dict[str, tuple[dict[str, Any], float]] = {}
    cursor = after_id
    end = time.time() + timeout_s
    while time.time() < end:
        data = api.get(f"/api/messages?after={cursor}")
        msgs = data.get("messages") or []
        for msg in msgs:
            mid = int(msg.get("id") or 0)
            if mid > cursor:
                cursor = mid
            if str(msg.get("role") or "") != "model":
                continue
            if str(msg.get("visibility") or "") != "public":
                continue
            mk = str(msg.get("model_key") or "").strip().lower()
            if mk in keys and mk not in seen:
                seen[mk] = (msg, time.time())
        if len(seen) >= len(keys):
            break
        time.sleep(0.5)

    out: list[RoundStat] = []
    for key in keys:
        if key not in seen:
            out.append(
                RoundStat(
                    round_no=0,
                    model_key=key,
                    latency_s=float(timeout_s),
                    ok=False,
                    reason="missing_public_reply",
                    preview="",
                )
            )
            continue
        msg, recv_at = seen[key]
        text = norm(str(msg.get("text") or ""))
        ok, reason = quality_check(text)
        out.append(
            RoundStat(
                round_no=0,
                model_key=key,
                latency_s=max(0.0, recv_at - sent_at),
                ok=ok,
                reason=reason,
                preview=text.replace("\n", " / ")[:180],
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Group projection latency and extraction self-test")
    ap.add_argument("--base", default="http://127.0.0.1:8765", help="WebUI base URL")
    ap.add_argument("--models", default="doubao,qwen", help="comma separated model keys")
    ap.add_argument("--rounds", type=int, default=12, help="test rounds")
    ap.add_argument("--timeout", type=int, default=90, help="per round timeout seconds")
    args = ap.parse_args()

    api = Api(args.base)
    keys = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    if not keys:
        raise RuntimeError("no model keys")

    ensure_selected(api, keys)
    wait_idle(api, timeout_s=20)

    prompts = [
        "成语接龙测试：直接接一个四字成语，并补一句你对上一个成语的感受。",
        "讨论话题：怎么降低团队会议低效？给一个可执行建议，再追问一句。",
        "你们两位快速辩论：AI 产品先做功能还是先做体验？各说一条理由。",
        "请你先回应用户问题，再给一个补充角度，不要寒暄。",
    ]

    stats: list[RoundStat] = []
    for i in range(1, max(1, args.rounds) + 1):
        prompt = f"[SELFTEST-{i}] {prompts[(i - 1) % len(prompts)]}"
        round_stats = run_round(api, keys, prompt, timeout_s=args.timeout)
        for item in round_stats:
            item.round_no = i
            stats.append(item)
            print(
                f"R{i:02d} {item.model_key:<8} latency={item.latency_s:5.1f}s "
                f"ok={str(item.ok):<5} reason={item.reason:<24} preview={item.preview}"
            )

    by_key: dict[str, list[RoundStat]] = {k: [] for k in keys}
    for s in stats:
        by_key.setdefault(s.model_key, []).append(s)

    print("\n=== SUMMARY ===")
    for key in keys:
        arr = by_key.get(key, [])
        if not arr:
            print(f"{key}: no data")
            continue
        ok_arr = [x for x in arr if x.ok]
        avg_lat = sum(x.latency_s for x in arr) / len(arr)
        p95_src = sorted(x.latency_s for x in arr)
        p95 = p95_src[min(len(p95_src) - 1, int(len(p95_src) * 0.95))]
        print(
            f"{key}: pass={len(ok_arr)}/{len(arr)} "
            f"avg={avg_lat:.1f}s p95={p95:.1f}s fail={len(arr) - len(ok_arr)}"
        )

    bad = [x for x in stats if not x.ok]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
