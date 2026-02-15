"""
一个轻量的本地 Web UI，用来把 ChatGPT + Gemini 的“互怼”对话汇总到一个页面里，
并允许你作为第三位成员插话（群主）。

设计原则：
- 不引入额外依赖（仅使用标准库 + playwright + 现有 ai_duel.py 逻辑）
- UI 通过轮询 /api/messages 获取新消息；通过 /api/send 发送你的消息
- 底层仍然是 Playwright 操控网页端（因此 chatgpt.com / gemini.google.com 仍会打开，但你不需要盯着它们）
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
import traceback
import urllib.parse
import webbrowser
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import ai_duel as core
from playwright.sync_api import sync_playwright

from model_adapters import (
    ChatGPTAdapter,
    DeepSeekAdapter,
    GeminiAdapter,
    GenericWebChatAdapter,
    ModelAdapter,
    ModelMeta,
    default_avatar_svg,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_MESSAGES = 1500
MAX_CONTEXT_MESSAGES = 120
MAX_MODEL_PROMPT_CHARS = 12000
MODEL_REPLY_TIMEOUT_S = 180
GROUP_CONTINUOUS_MAX_ROUNDS = 60
GROUP_DEFAULT_ROUNDS = -1  # -1 means continuous rounds until manual stop/safety cap
FOCUS_RECOVERY_ROUNDS = 2

MODEL_MENTION_ALIASES: dict[str, tuple[str, ...]] = {
    "chatgpt": ("chatgpt", "gpt", "chat-gpt", "chat gpt"),
    "gemini": ("gemini", "google gemini"),
    "deepseek": ("deepseek", "deep seek", "deep-seek"),
    "doubao": ("doubao", "豆包"),
    "qwen": ("qwen", "通义", "千问"),
    "kimi": ("kimi",),
}

# “对话规则”会被作为用户消息的一部分发送到网页端模型（无法真正注入 system prompt）。
# 目标：更短、更像人说话；允许联网获取事实，但“结论/洞见/权衡”必须独立推理而非“搜索答案式”复述。
DEFAULT_RULES = """【对话规则】
- 语气像正常人聊天：自然、有温度，但不要废话。
- 禁止“编号式模板”（不要 1/2/3/4 逐条报菜名）。可以分 2-4 个短段落，或极少量要点（<=3）。
- 允许联网：仅在需要“最新事实/具体数据/可核验信息”时联网；联网只用于补充事实，不要把网页复述当结论。
- 争论时要抓重点：先用 1 句复述对方核心主张（不超过 25 字），再用 2-4 句给出最关键的反驳/补充。
- 必须给 1 个反例/边界条件（1 句即可），避免空泛。
- 最后给 1 个追问，推动更深一步讨论。
- 总长度尽量 <= 320 字。
- 末尾追加一段【转发摘要】（<= 120 字），只写关键观点，供另一模型接续（不要写元信息/不要说“我将转发”）。
""".strip()

# 每轮轻量提醒：避免模型“跑偏”回长文/检索式回答
RULES_REMINDER = "【提醒】像人说话但要短；别用 1/2/3 模板；可联网补事实但结论要自己推理；抓住对方核心点反驳；末尾带【转发摘要】。"

# 转发给另一模型的内容长度上限（过长会导致输入卡顿，也会让对话越来越发散）
FORWARD_MAX_CHARS = 900

# 模型槽位（UI 左侧 1..10）
# 说明：
# - enabled=False 表示“UI 可见但暂未接入自动化”；会以灰色不可点击展示。
# - 未来要接入更多模型：为每个模型补齐 Playwright 适配器（发送/等待/提取）后，把 enabled 改为 True。
MODEL_SLOTS: list[dict[str, Any]] = [
    {"slot": 1, "key": "chatgpt", "name": "ChatGPT", "enabled": True},
    {"slot": 2, "key": "gemini", "name": "Gemini", "enabled": True},
    {"slot": 3, "key": "deepseek", "name": "DeepSeek", "enabled": False},
    {"slot": 4, "key": "doubao", "name": "豆包", "enabled": False},
    {"slot": 5, "key": "qwen", "name": "通义千问", "enabled": False},
    {"slot": 6, "key": "kimi", "name": "Kimi", "enabled": False},
    {"slot": 7, "key": "zhipu", "name": "智谱", "enabled": False},
    {"slot": 8, "key": "claude", "name": "Claude", "enabled": False},
    {"slot": 9, "key": "glmm", "name": "GLM", "enabled": False},
    {"slot": 10, "key": "other", "name": "Other", "enabled": False},
]

# 默认启用：和当前代码一致（ChatGPT + Gemini）
# 需求：禁止默认 GPT/Gemini 自动启用，初始 selected 为空
DEFAULT_SELECTED_KEYS: list[str] = []


def _mk_meta(
    slot: int,
    key: str,
    name: str,
    url: str,
    color: str,
    integrated: bool,
    login_help: str,
    avatar_label: str,
) -> ModelMeta:
    return ModelMeta(
        slot=slot,
        key=key,
        name=name,
        url=url,
        color=color,
        integrated=integrated,
        avatar_url=default_avatar_svg(color, avatar_label),
        login_help=login_help,
    )


# 需求：左侧固定 1..10 槽位；初始 selected 为空（禁止默认启用）。
# 至少落地 1 个第三方模型：DeepSeek（slot=3）。
MODEL_METAS: list[ModelMeta] = [
    _mk_meta(
        1,
        "chatgpt",
        "ChatGPT",
        "https://chatgpt.com/",
        "#16a34a",
        True,
        "登录说明（ChatGPT）\n1) 点击【打开登录窗口】会打开 chatgpt.com 官方网页。\n2) 请手动登录并关闭欢迎弹窗。\n3) 回来点击【我已登录，重新检测】。\n登录态会保存在 user_data/chatgpt。",
        "G",
    ),
    _mk_meta(
        2,
        "gemini",
        "Gemini",
        "https://gemini.google.com/app",
        "#f59e0b",
        True,
        "登录说明（Gemini）\n1) 点击【打开登录窗口】会打开 gemini.google.com 官方网页。\n2) 手动完成 Google 登录并关闭弹窗。\n3) 回来点击【我已登录，重新检测】。\n登录态会保存在 user_data/gemini。",
        "2",
    ),
    _mk_meta(
        3,
        "deepseek",
        "DeepSeek",
        "https://chat.deepseek.com/",
        "#38bdf8",
        True,
        "登录说明（DeepSeek）\n1) 点击【打开登录窗口】会打开 chat.deepseek.com 官方网页。\n2) 手动登录并关闭弹窗。\n3) 回来点击【我已登录，重新检测】。\n登录态会保存在 user_data/deepseek。",
        "D",
    ),
    _mk_meta(
        4,
        "doubao",
        "豆包",
        "https://www.doubao.com/chat/",
        "#a855f7",
        True,
        "登录说明（豆包）\n1) 点击【打开登录窗口】打开 doubao.com。\n2) 手动登录并关闭弹窗。\n3) 点【我已登录，重新检测】。\n注：当前为通用适配器（启发式 selector），站点 UI 变更可能需调整。",
        "B",
    ),
    _mk_meta(
        5,
        "qwen",
        "Qwen",
        "https://chat.qwen.ai/",
        "#22c55e",
        True,
        "登录说明（Qwen）\n1) 点击【打开登录窗口】打开 chat.qwen.ai。\n2) 手动登录并关闭弹窗。\n3) 点【我已登录，重新检测】。\n注：当前为通用适配器（启发式 selector），站点 UI 变更可能需调整。",
        "Q",
    ),
    _mk_meta(
        6,
        "kimi",
        "Kimi",
        "https://kimi.moonshot.cn/",
        "#0ea5e9",
        True,
        "登录说明（Kimi）\n1) 点击【打开登录窗口】打开 Kimi 官方网页。\n2) 手动登录并关闭弹窗。\n3) 点【我已登录，重新检测】。\nTODO：如输入框识别不稳，请为 Kimi 写专用适配器。",
        "K",
    ),
    _mk_meta(7, "zhipu", "智谱（占位）", "https://example.com/", "#64748b", False, "未接入：请补齐 URL + selector 适配器。", "7"),
    _mk_meta(8, "claude", "Claude（占位）", "https://example.com/", "#64748b", False, "未接入：请补齐 URL + selector 适配器。", "8"),
    _mk_meta(9, "slot9", "槽位9（占位）", "https://example.com/", "#64748b", False, "未接入：请补齐 URL + selector 适配器。", "9"),
    _mk_meta(10, "slot10", "槽位10（占位）", "https://example.com/", "#64748b", False, "未接入：请补齐 URL + selector 适配器。", "10"),
]


def build_adapter(meta: ModelMeta) -> ModelAdapter:
    if meta.key == "chatgpt":
        return ChatGPTAdapter(meta)
    if meta.key == "gemini":
        return GeminiAdapter(meta)
    if meta.key == "deepseek":
        return DeepSeekAdapter(meta)
    return GenericWebChatAdapter(meta)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _open_webui_window(url: str) -> None:
    """
    尽量以“独立窗口”的方式打开 UI（Chrome/Edge 的 --app 模式），避免淹没在一堆浏览器标签页里。
    找不到浏览器可执行文件时，回退到系统默认浏览器。
    """

    if os.name == "nt":
        pf = os.environ.get("PROGRAMFILES") or ""
        pfx86 = os.environ.get("PROGRAMFILES(X86)") or ""
        local = os.environ.get("LOCALAPPDATA") or ""

        candidates = [
            # Chrome
            os.path.join(pf, "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(pfx86, "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(local, "Google", "Chrome", "Application", "chrome.exe"),
            # Edge
            os.path.join(pf, "Microsoft", "Edge", "Application", "msedge.exe"),
            os.path.join(pfx86, "Microsoft", "Edge", "Application", "msedge.exe"),
            os.path.join(local, "Microsoft", "Edge", "Application", "msedge.exe"),
        ]

        for exe in candidates:
            try:
                if exe and Path(exe).exists():
                    subprocess.Popen(  # noqa: S603
                        [
                            exe,
                            f"--app={url}",
                            "--new-window",
                            "--window-size=1440,900",
                            "--window-position=60,40",
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return
            except Exception:
                continue

    try:
        webbrowser.open_new_tab(url)
    except Exception:
        pass

@dataclass
class UiMessage:
    id: int
    ts: str
    role: str  # user | model | system
    speaker: str
    text: str
    visibility: str  # public | shadow
    model_key: Optional[str] = None  # 私聊线程归属；或 model 回复来源


@dataclass
class ModelRuntime:
    slot: int
    key: str
    name: str
    url: str
    color: str
    integrated: bool
    avatar_url: str
    login_help: str
    selected: bool = False
    authenticated: bool = False


class SharedState:
    def __init__(self) -> None:
        # 用于保护 messages/models/status 的共享状态；RLock 允许同线程内嵌套调用（尽量仍避免）。
        self._lock = threading.RLock()
        self._next_id = 1
        self._messages: list[UiMessage] = []
        self._status: str = "idle"
        self._stop: bool = False
        self._round_stop_requested: bool = False
        self._rules: str = DEFAULT_RULES

        self._pending_enable: set[str] = set()
        self._models: dict[str, ModelRuntime] = {}
        for meta in MODEL_METAS:
            self._models[meta.key] = ModelRuntime(
                slot=meta.slot,
                key=meta.key,
                name=meta.name,
                url=meta.url,
                color=meta.color,
                integrated=meta.integrated,
                avatar_url=meta.avatar_url,
                login_help=meta.login_help,
                selected=False,  # 需求：初始 0 模型启用
                authenticated=False,
            )

        # 后端动作队列（Playwright 只能在同一线程执行）
        self.inbox: "queue.Queue[dict[str, Any]]" = queue.Queue()

    # --- messages ------------------------------------------------------

    def add_message(
        self,
        role: str,
        speaker: str,
        text: str,
        *,
        visibility: str,
        model_key: Optional[str] = None,
    ) -> int:
        t = core.normalize_text(text)
        if not t:
            return 0
        with self._lock:
            mid = self._next_id
            self._next_id += 1
            self._messages.append(
                UiMessage(
                    id=mid,
                    ts=_now_iso(),
                    role=role,
                    speaker=speaker,
                    text=t,
                    visibility=visibility,
                    model_key=model_key,
                )
            )
            if len(self._messages) > MAX_MESSAGES:
                self._messages = self._messages[-MAX_MESSAGES:]
            return mid

    def add_system(self, text: str) -> int:
        return self.add_message("system", "系统", text, visibility="public")

    def get_messages_after(self, after_id: int) -> dict[str, Any]:
        with self._lock:
            msgs = [asdict(m) for m in self._messages if m.id > after_id]
        return {"ok": True, "messages": msgs}

    def get_all_messages(self) -> list[UiMessage]:
        with self._lock:
            return list(self._messages)

    # --- models --------------------------------------------------------

    def get_models(self) -> dict[str, Any]:
        with self._lock:
            items = [asdict(m) for m in self._models.values()]
        items.sort(key=lambda x: x["slot"])
        return {"ok": True, "models": items}

    def get_model(self, key: str) -> Optional[ModelRuntime]:
        key = (key or "").strip().lower()
        with self._lock:
            return self._models.get(key)

    def is_selected(self, key: str) -> bool:
        m = self.get_model(key)
        return bool(m and m.selected)

    def selected_keys(self) -> list[str]:
        with self._lock:
            pairs = [(k, m.slot) for k, m in self._models.items() if m.selected]
        pairs.sort(key=lambda x: x[1])
        return [k for k, _ in pairs]

    def set_authenticated(self, key: str, value: bool) -> None:
        key = (key or "").strip().lower()
        with self._lock:
            m = self._models.get(key)
            if not m:
                return
            m.authenticated = bool(value)

    def toggle_selected(self, key: str) -> dict[str, Any]:
        key = (key or "").strip().lower()
        with self._lock:
            m = self._models.get(key)
            if not m:
                return {"ok": False, "error": "unknown_model"}
            if not m.integrated:
                return {"ok": False, "error": "not_integrated"}

            # turn off
            if m.selected:
                m.selected = False
                self._pending_enable.discard(key)
                self._append_system_locked(f"{m.name} 已退出群聊")
                items = [asdict(mm) for mm in self._models.values()]
                items.sort(key=lambda x: x["slot"])
                return {"ok": True, "need_auth": False, "models": items}

            # turn on
            if not m.authenticated:
                self._pending_enable.add(key)
                items = [asdict(mm) for mm in self._models.values()]
                items.sort(key=lambda x: x["slot"])
                return {"ok": False, "need_auth": True, "model": asdict(m), "models": items}

            m.selected = True
            self._append_system_locked(f"{m.name} 已加入群聊")
            items = [asdict(mm) for mm in self._models.values()]
            items.sort(key=lambda x: x["slot"])
            return {"ok": True, "need_auth": False, "models": items}

    def mark_pending_enable_done(self, key: str) -> bool:
        key = (key or "").strip().lower()
        with self._lock:
            if key not in self._pending_enable:
                return False
            m = self._models.get(key)
            if not m or not m.integrated or not m.authenticated:
                return False
            self._pending_enable.discard(key)
            if not m.selected:
                m.selected = True
                self._append_system_locked(f"{m.name} 已加入群聊")
            return True

    # --- status / stop -------------------------------------------------

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return {"ok": True, "status": self._status, "stop": self._stop}

    def request_stop(self) -> None:
        with self._lock:
            self._stop = True

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop

    def request_round_stop(self) -> None:
        with self._lock:
            self._round_stop_requested = True

    def clear_round_stop(self) -> None:
        with self._lock:
            self._round_stop_requested = False

    def should_round_stop(self) -> bool:
        with self._lock:
            return self._round_stop_requested

    # --- internal (lock held) ------------------------------------------

    def _append_system_locked(self, text: str) -> None:
        t = core.normalize_text(text)
        if not t:
            return
        mid = self._next_id
        self._next_id += 1
        self._messages.append(
            UiMessage(
                id=mid,
                ts=_now_iso(),
                role="system",
                speaker="系统",
                text=t,
                visibility="public",
                model_key=None,
            )
        )
        if len(self._messages) > MAX_MESSAGES:
            self._messages = self._messages[-MAX_MESSAGES:]


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AI 群聊控制台</title>
    <style>
      :root {
        --bg: #d9ecfb;
        --panel: #84b9df;
        --panel2: #9bc992;
        --text: #16324a;
        --muted: #35566d;
        --accent: #22c55e;
        --warn: #f59e0b;
        --danger: #ef4444;
        --border: rgba(13, 43, 64, 0.28);
      }
      html, body { height: 100%; }
      body {
        margin: 0;
        background:
          radial-gradient(1200px 700px at 15% 8%, rgba(255,255,255,0.42), transparent 58%),
          radial-gradient(1000px 620px at 92% 90%, rgba(120, 180, 220, 0.28), transparent 62%),
          linear-gradient(180deg, #dff1ff 0%, #cae5fb 100%);
        color: var(--text);
        font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans", "Helvetica Neue", Arial;
      }
      .app {
        display: grid;
        grid-template-columns: 96px 1fr;
        height: calc(100% - 26px);
        margin: 13px;
        border-radius: 28px;
        overflow: hidden;
        border: 2px solid rgba(17, 98, 152, 0.58);
        box-shadow: 0 28px 70px rgba(13,43,64,0.25);
        background: linear-gradient(180deg, #89bce0, #73abd8);
      }
      .modelbar {
        border-right: 1px solid var(--border);
        background: linear-gradient(180deg, #8fc0e3, #77afd9);
        padding: 12px 10px;
        display: flex;
        flex-direction: column;
        gap: 12px;
        align-items: center;
      }
      .modelList { display: flex; flex-direction: column; gap: 10px; align-items: center; }
      .modelBtn {
        width: 64px;
        height: 64px;
        padding: 0;
        border-radius: 18px;
        display: grid;
        place-items: center;
        position: relative;
        border: 2px solid rgba(13,43,64,0.62);
        background: rgba(246, 251, 255, 0.88);
        cursor: pointer;
        transition: transform 120ms ease, border-color 120ms ease, filter 120ms ease, opacity 120ms ease;
        user-select: none;
      }
      .modelBtn:hover { transform: translateY(-1px); border-color: rgba(13,43,64,0.82); }
      .modelBtn:active { transform: translateY(0px) scale(0.98); }
      .modelBtn.off { filter: grayscale(0.88); opacity: 0.74; }
      .modelBtn.disabled { filter: grayscale(1); opacity: 0.40; cursor: not-allowed; }
      .modelBtn.on { border-color: rgba(255,255,255,0.72); box-shadow: 0 14px 28px rgba(13,43,64,0.22); filter: none; opacity: 1; }
      .modelBtn.active { box-shadow: 0 0 0 4px rgba(34,197,94,0.16), 0 16px 32px rgba(13,43,64,0.28); border-color: rgba(255,255,255,0.88); }
      .modelNum {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        font-weight: 900;
        font-size: 18px;
        color: rgba(255,255,255,0.95);
        border: 1px solid rgba(0,0,0,0.18);
        box-shadow: 0 14px 28px rgba(0,0,0,0.22);
        overflow: hidden;
        position: relative;
      }
      .modelNum img { width: 100%; height: 100%; object-fit: cover; }
      .modelSlotNum {
        position: absolute;
        right: 5px;
        bottom: 5px;
        width: 18px;
        height: 18px;
        border-radius: 7px;
        display: grid;
        place-items: center;
        font-size: 11px;
        font-weight: 900;
        background: rgba(0,0,0,0.28);
        border: 1px solid rgba(255,255,255,0.16);
      }
      .authDot {
        position: absolute;
        left: 6px;
        bottom: 6px;
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: var(--accent);
        border: 1px solid rgba(0,0,0,0.22);
        box-shadow: 0 10px 20px rgba(34,197,94,0.25);
        display: none;
      }
      .modelBtn.authed .authDot { display: block; }
      .lockIcon {
        position: absolute;
        right: 8px;
        top: 8px;
        width: 16px;
        height: 12px;
        border-radius: 4px;
        border: 2px solid rgba(255,255,255,0.60);
        opacity: 0.85;
        display: none;
      }
      .lockIcon:before {
        content: "";
        position: absolute;
        left: 50%;
        top: -9px;
        width: 10px;
        height: 9px;
        transform: translateX(-50%);
        border-radius: 10px 10px 0 0;
        border: 2px solid rgba(255,255,255,0.60);
        border-bottom: none;
      }
      .modelBtn.disabled .lockIcon { display: block; }
      .nudgeBtn {
        position: absolute;
        left: 6px;
        top: 6px;
        width: 22px;
        height: 22px;
        border-radius: 9px;
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(0,0,0,0.22);
        color: rgba(255,255,255,0.92);
        display: none;
        place-items: center;
        font-size: 12px;
        cursor: pointer;
        user-select: none;
      }
      .modelBtn.on .nudgeBtn { display: grid; }
      .nudgeBtn:hover { background: rgba(0,0,0,0.28); }
      .badge {
        position: absolute;
        top: -8px;
        right: -8px;
        min-width: 18px;
        height: 18px;
        padding: 0 6px;
        border-radius: 999px;
        background: rgba(239,68,68,0.90);
        color: rgba(255,255,255,0.98);
        font-weight: 800;
        font-size: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(0,0,0,0.18);
      }
      .badge.hidden { display: none; }
      .modelStart {
        width: 64px;
        padding: 10px 0;
        border-radius: 18px;
        border: 2px solid rgba(13,43,64,0.65);
        background: linear-gradient(180deg, #6fc1f0, #4da4d8);
        color: #f4fbff;
        font-weight: 800;
      }
      .content { display: grid; grid-template-columns: 1fr; height: 100%; min-width: 0; }
      .sidebar {
        display: none;
        border-right: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(17,24,38,0.95), rgba(15,23,42,0.85));
        padding: 16px;
        /* hidden (we keep markup for simplicity, but the UI is single-panel) */
        flex-direction: column;
        gap: 12px;
        min-width: 0;
      }
      .brand { display: flex; align-items: baseline; gap: 10px; }
      .brand h1 { margin: 0; font-size: 18px; letter-spacing: 0.4px; }
      .pill {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 6px 10px; border: 1px solid var(--border);
        border-radius: 999px; font-size: 12px; color: var(--muted);
        background: rgba(241, 249, 255, 0.82);
      }
      .dot { width: 8px; height: 8px; border-radius: 999px; background: var(--warn); }
      .dot.ok { background: var(--accent); }
      .dot.bad { background: var(--danger); }
      .row { display: flex; gap: 10px; }
      button, select, textarea {
        border: 1px solid var(--border);
        background: rgba(244, 251, 255, 0.90);
        color: var(--text);
        border-radius: 10px;
        font-size: 13px;
      }
      button { padding: 10px 12px; cursor: pointer; }
      button:hover { border-color: rgba(148,163,184,0.35); }
      button.primary { background: rgba(34,197,94,0.22); border-color: rgba(34,197,94,0.42); }
      button.danger { background: rgba(239,68,68,0.18); border-color: rgba(239,68,68,0.40); }
      select {
        padding: 10px 40px 10px 12px;
        flex: 1;
        appearance: none;
        -webkit-appearance: none;
        border-radius: 14px;
        background-image:
          linear-gradient(45deg, transparent 50%, rgba(229,231,235,0.65) 50%),
          linear-gradient(135deg, rgba(229,231,235,0.65) 50%, transparent 50%);
        background-position:
          calc(100% - 18px) 18px,
          calc(100% - 12px) 18px;
        background-size: 6px 6px, 6px 6px;
        background-repeat: no-repeat;
      }
      textarea {
        width: 100%; min-height: 120px; resize: vertical;
        padding: 10px 12px; outline: none;
      }
      .hint { font-size: 12px; color: var(--muted); line-height: 1.5; }
      .main { display: flex; flex-direction: column; height: 100%; min-width: 0; }
      .topbar {
        padding: 14px 18px;
        border-bottom: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(232, 245, 255, 0.92), rgba(218, 239, 255, 0.88));
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .status { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--muted); }
      .chatWrap { position: relative; flex: 1; min-height: 0; }
      .chat {
        position: absolute;
        inset: 0;
        padding: 18px 18px 26px;
        overflow: auto;
        scroll-behavior: smooth;
        border-radius: 24px;
        border: 1px solid rgba(15, 23, 42, 0.22);
        box-shadow: 0 22px 54px rgba(0,0,0,0.22);
        background: linear-gradient(180deg, rgba(204, 236, 207, 0.98), rgba(191, 230, 195, 0.98));
      }
      .chat::-webkit-scrollbar { width: 12px; }
      .chat::-webkit-scrollbar-track { background: rgba(2,6,23,0.14); border-radius: 999px; }
      .chat::-webkit-scrollbar-thumb {
        background: rgba(15,23,42,0.35);
        border-radius: 999px;
        border: 3px solid rgba(2,6,23,0.14);
      }
      .chat::-webkit-scrollbar-thumb:hover { background: rgba(15,23,42,0.48); }

      .jump {
        display: none;
        position: absolute;
        right: 18px;
        bottom: 18px;
        z-index: 5;
        border-radius: 999px;
        padding: 8px 10px;
        font-size: 12px;
        background: rgba(17, 98, 152, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.28);
        box-shadow: 0 16px 40px rgba(0,0,0,0.25);
      }
      .jump.hidden { display: none; }

      .msg-row { display: flex; align-items: flex-end; gap: 10px; margin: 10px 0; }
      .msg-row.you { justify-content: flex-end; }
      .msg-row.system { justify-content: center; }

      .avatar {
        width: 34px;
        height: 34px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 13px;
        color: rgba(255,255,255,0.95);
        box-shadow: 0 16px 40px rgba(0,0,0,0.25);
        flex: 0 0 auto;
        user-select: none;
      }
      .avatar.you { background: linear-gradient(180deg, rgba(96,165,250,0.95), rgba(37,99,235,0.90)); }
      .avatar.chatgpt { background: linear-gradient(180deg, rgba(34,197,94,0.95), rgba(16,185,129,0.88)); }
      .avatar.gemini { background: linear-gradient(180deg, rgba(245,158,11,0.95), rgba(251,191,36,0.86)); }
      .avatar.system { background: linear-gradient(180deg, rgba(167,139,250,0.95), rgba(139,92,246,0.88)); }

      .bubble {
        max-width: 780px;
        position: relative;
        border: 1px solid rgba(15, 23, 42, 0.18);
        background: rgba(255,255,255,0.74);
        color: rgba(15, 23, 42, 0.92);
        border-radius: 18px;
        padding: 12px 14px;
        box-shadow: 0 16px 34px rgba(0,0,0,0.22);
      }
      .msg-row.system .bubble { background: rgba(255,255,255,0.70); text-align: center; }
      .msg-row.you .bubble {
        background: linear-gradient(180deg, rgba(22,163,74,0.95), rgba(15,122,55,0.92));
        border-color: rgba(0,0,0,0.18);
        color: rgba(255,255,255,0.95);
      }

      /* bubble tails */
      .msg-row.you .bubble::before {
        content: "";
        position: absolute;
        right: -10px;
        top: 18px;
        width: 0;
        height: 0;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
        border-left: 12px solid rgba(22,163,74,0.95);
        filter: drop-shadow(2px 2px 2px rgba(0,0,0,0.12));
      }
      .msg-row.model .bubble::before {
        content: "";
        position: absolute;
        left: -10px;
        top: 18px;
        width: 0;
        height: 0;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
        border-right: 12px solid rgba(255,255,255,0.74);
        filter: drop-shadow(-2px 2px 2px rgba(0,0,0,0.10));
      }

      .bubble-meta { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
      .name { font-weight: 750; font-size: 12px; letter-spacing: 0.2px; }
      .name.you { color: rgba(255,255,255,0.88); }
      .name.system { color: rgba(15,23,42,0.70); }
      .time { font-size: 11px; color: var(--muted); white-space: nowrap; }
      .bubble-text { margin-top: 6px; white-space: pre-wrap; line-height: 1.55; font-size: 14px; }
      .sysTime { margin-top: 6px; font-size: 11px; color: var(--muted); }

      .msg-row.continued .avatar { visibility: hidden; }
      .msg-row.continued .name { display: none; }

      .composer {
        border-top: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(121, 181, 222, 0.92), rgba(99, 165, 210, 0.90));
        padding: 12px 18px 14px;
        display: grid;
        grid-template-columns: 240px 1fr 160px;
        gap: 12px;
        align-items: end;
      }
      .targetStack {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }
      .targetStack select { width: 100%; }
      .composer textarea {
        width: 100%;
        min-height: 58px;
        max-height: 180px;
        resize: none;
        padding: 12px 12px;
        border-radius: 14px;
        border: 2px solid rgba(13,43,64,0.56);
        background: linear-gradient(180deg, rgba(88, 171, 220, 0.95), rgba(74, 157, 208, 0.94));
        color: rgba(255,255,255,0.95);
        font-size: 14px;
        line-height: 1.5;
      }
      .composer textarea::placeholder { color: rgba(239, 248, 255, 0.92); }
      .composerActions { display: flex; flex-direction: column; gap: 10px; }
      .composerHint {
        grid-column: 1 / -1;
        display: flex;
        justify-content: space-between;
        gap: 12px;
        font-size: 12px;
        color: var(--muted);
      }
      .kbd {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 8px;
        border: 1px solid rgba(148,163,184,0.28);
        background: rgba(239, 248, 255, 0.85);
        color: rgba(23,42,58,0.95);
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        font-size: 11px;
      }
      code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }

      /* auth modal */
      .hidden { display: none !important; }
      .modalMask {
        position: fixed;
        inset: 0;
        background: rgba(13,43,64,0.30);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 18px;
        z-index: 50;
      }
      .modal {
        width: min(760px, 96vw);
        border-radius: 22px;
        border: 2px solid rgba(13,43,64,0.42);
        background: rgba(245, 252, 255, 0.98);
        box-shadow: 0 30px 80px rgba(13,43,64,0.24);
        overflow: hidden;
      }
      .modalHead {
        padding: 14px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        border-bottom: 1px solid rgba(148,163,184,0.16);
        background: rgba(219, 239, 255, 0.86);
      }
      .modalTitle { font-weight: 900; letter-spacing: 0.3px; color: #12324a; }
      .modalClose {
        width: 34px;
        height: 34px;
        border-radius: 12px;
        border: 1px solid rgba(13,43,64,0.28);
        background: rgba(255,255,255,0.92);
        color: #12324a;
        cursor: pointer;
      }
      .modalBody { padding: 14px 16px 16px; }
      .modalHelp {
        white-space: pre-wrap;
        line-height: 1.45;
        color: rgba(23,42,58,0.90);
        border-radius: 16px;
        border: 1px solid rgba(13,43,64,0.18);
        background: rgba(234, 246, 255, 0.88);
        padding: 12px 12px;
      }
      .modalBtns { display: flex; gap: 12px; margin-top: 12px; flex-wrap: wrap; }
      .modalBtn {
        height: 44px;
        padding: 0 16px;
        border-radius: 16px;
        border: 1px solid rgba(0,0,0,0.18);
        font-weight: 900;
        cursor: pointer;
        color: rgba(255,255,255,0.95);
        box-shadow: 0 12px 24px rgba(13,43,64,0.22);
      }
      .modalOpen { background: rgba(34,197,94,0.20); border-color: rgba(34,197,94,0.35); }
      .modalCheck { background: rgba(245,158,11,0.18); border-color: rgba(245,158,11,0.35); }
    </style>
  </head>
  <body>
    <div class="app">
      <aside class="modelbar">
        <div id="modelList" class="modelList"></div>
        <button id="startSessionBtn" class="modelStart" title="停止自动轮聊（不关闭服务）">停止轮聊</button>
      </aside>

      <div class="content">
      <aside class="sidebar">
        <div class="brand">
          <h1>AI 群聊控制台</h1>
          <span class="pill"><span id="dot" class="dot"></span><span id="conn">connecting</span></span>
        </div>

        <div class="pill" id="statusPill">status: <span id="statusText">starting</span></div>
        <div class="pill" id="unseenPill">未读：ChatGPT 0 | Gemini 0</div>

        <div class="row">
          <button id="pauseBtn">暂停</button>
          <button id="stopBtn" class="danger">停止</button>
        </div>

        <div class="row">
          <select id="target">
            <option value="next" selected>互怼模式（A↔B 自动转发）</option>
            <option value="chatgpt">前台 ChatGPT（Gemini 后台同步）</option>
            <option value="gemini">前台 Gemini（ChatGPT 后台同步）</option>
            <option value="both">广播两边（各回复一次，随后暂停）</option>
          </select>
        </div>

        <details id="rulesBox">
          <summary class="pill">对话规则（两边共享，可改）</summary>
          <textarea id="rulesInput" placeholder="在这里写规则（会作为用户消息的一部分发送给模型）..."></textarea>
          <div class="row">
            <button id="saveRulesBtn">应用规则</button>
          </div>
          <div class="hint">
            <div>- 规则会在下一轮发送给两边（自动重新注入）。</div>
            <div>- 过长会被截断，建议保持简短。</div>
          </div>
        </details>

        <div class="hint">
          <div>说明：</div>
          <div>- 页面只汇总文本，不会嵌入 chatgpt.com / gemini.google.com（跨域限制）。</div>
          <div>- 底层仍会打开一个 Playwright 浏览器（你可以最小化它）。</div>
          <div>- 如果出现验证码/风控/弹窗，仍需你在原网页手动处理。</div>
          <div>- 两个模型平等：选择“前台 ChatGPT/Gemini”仅影响你当前看到哪一边；另一边仍会后台同步并生成回复，计入“未读”。</div>
        </div>
      </aside>

      <main class="main">
         <div class="topbar">
           <div class="status">
             <span>消息流</span>
             <span class="pill" id="countPill">0</span>
           </div>
           <div class="hint">刷新间隔 800ms，长对话建议定期清屏</div>
         </div>
        <div class="chatWrap">
          <div id="chat" class="chat"></div>
          <button id="jumpBtn" class="jump hidden" title="回到最新">↓ 新消息</button>
        </div>
        <div class="composer">
          <div class="targetStack">
            <select id="sendTarget"></select>
            <select id="groupRounds" title="群聊自动轮聊轮数">
              <option value="-1" selected>轮聊：持续（直到点“停止轮聊”）</option>
              <option value="5">轮聊：5 轮</option>
              <option value="3">轮聊：3 轮</option>
              <option value="1">轮聊：1 轮（每模型一次）</option>
            </select>
          </div>
          <textarea id="input" placeholder="请输入你的文本（群主发言）..."></textarea>
          <div class="composerActions">
            <button id="sendBtn" class="primary" disabled>发送</button>
            <button id="clearBtn">清空</button>
          </div>
          <div class="composerHint">
            <div id="hintLeft" style="color: var(--warn); font-weight: 800;">请先选择发送目标</div>
            <div>快捷键：<span class="kbd">Enter</span> 发送，<span class="kbd">Shift+Enter</span> 换行</div>
          </div>
        </div>
      </main>
      </div>
    </div>

    <!-- Auth modal -->
    <div id="authMask" class="modalMask hidden">
      <div class="modal">
        <div class="modalHead">
          <div id="authTitle" class="modalTitle">登录</div>
          <button id="authClose" class="modalClose">×</button>
        </div>
        <div class="modalBody">
          <div id="authHelp" class="modalHelp"></div>
          <div class="modalBtns">
            <button id="authOpen" class="modalBtn modalOpen">打开登录窗口</button>
            <button id="authCheck" class="modalBtn modalCheck">我已登录，重新检测</button>
          </div>
        </div>
      </div>
    </div>

    <script>
      // === WebUI v2 (multi-model group chat) ===
      (() => {
        const $ = (id) => document.getElementById(id);
        const esc = (s) => (s || '')
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;');

        const chat = $('chat');
        const modelList = $('modelList');
        const stopTopBtn = $('startSessionBtn');
        const sendTarget = $('sendTarget');
        const groupRounds = $('groupRounds');
        const input = $('input');
        const sendBtn = $('sendBtn');
        const clearBtn = $('clearBtn');
        const hintLeft = $('hintLeft');
        const countPill = $('countPill');

        const authMask = $('authMask');
        const authTitle = $('authTitle');
        const authHelp = $('authHelp');
        const authClose = $('authClose');
        const authOpen = $('authOpen');
        const authCheck = $('authCheck');

        const st = {
          models: [],
          allMessages: [],
          lastId: 0,
          viewTarget: '',
          authKey: null,
        };

        const apiGet = async (url) => {
          try {
            const r = await fetch(url, { cache: 'no-store' });
            return await r.json();
          } catch (e) {
            console.error(e);
            return null;
          }
        };

        const apiPost = async (url, body) => {
          try {
            const r = await fetch(url, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(body || {}),
            });
            return await r.json();
          } catch (e) {
            console.error(e);
            alert('请求失败: ' + e);
            return null;
          }
        };

        const openAuth = (key) => {
          st.authKey = key;
          const m = st.models.find(x => x.key === key);
          authTitle.textContent = m ? ('登录：' + m.name) : '登录';
          authHelp.textContent = m && m.login_help ? m.login_help : '请在新窗口手动登录，然后点击重新检测。';
          authMask.classList.remove('hidden');
        };

        const closeAuth = () => {
          st.authKey = null;
          authMask.classList.add('hidden');
        };

        const updateComposerEnabled = () => {
          const selectedCount = st.models.filter(m => m.integrated && m.selected).length;
          const hasTarget = !!st.viewTarget && st.viewTarget !== '__sep__';
          const canSend = hasTarget && !(st.viewTarget === 'group' && selectedCount === 0);
          sendBtn.disabled = !canSend;

          if (!hasTarget) {
            hintLeft.textContent = '请先选择发送目标';
            hintLeft.style.color = 'var(--warn)';
          } else if (st.viewTarget === 'group' && selectedCount === 0) {
            hintLeft.textContent = '群聊目标已选，但还没有已启用模型（先点左侧模型并完成登录）';
            hintLeft.style.color = 'var(--danger)';
          } else if (st.viewTarget === 'group') {
            const rv = parseInt((groupRounds && groupRounds.value) ? groupRounds.value : '-1', 10);
            const roundTip = rv < 0 ? '持续轮聊' : ('自动 ' + rv + ' 轮');
            hintLeft.textContent = '群聊：public · ' + roundTip;
            hintLeft.style.color = 'rgba(23,42,58,0.86)';
          } else {
            const m = st.models.find(x => x.key === st.viewTarget);
            hintLeft.textContent = '单聊：shadow（' + (m ? m.name : st.viewTarget) + '）';
            hintLeft.style.color = 'rgba(23,42,58,0.86)';
          }
        };

        const rebuildTargets = () => {
          const prev = sendTarget.value;
          sendTarget.innerHTML = '';

          const opt0 = document.createElement('option');
          opt0.value = '';
          opt0.textContent = '请选择发送目标';
          sendTarget.appendChild(opt0);

          const optG = document.createElement('option');
          optG.value = 'group';
          optG.textContent = '群聊（发给所有已启用模型）';
          sendTarget.appendChild(optG);

          const selected = st.models.filter(m => m.integrated && m.selected);
          if (selected.length) {
            const sep = document.createElement('option');
            sep.value = '__sep__';
            sep.disabled = true;
            sep.textContent = '──────── 单聊（仅已启用） ────────';
            sendTarget.appendChild(sep);
          }
          for (const m of selected) {
            const opt = document.createElement('option');
            opt.value = m.key;
            opt.textContent = '单聊：' + m.name;
            sendTarget.appendChild(opt);
          }

          const values = new Set(Array.from(sendTarget.options).map(o => o.value));
          if (values.has(prev)) {
            sendTarget.value = prev;
            st.viewTarget = prev;
          } else {
            sendTarget.value = '';
            st.viewTarget = '';
          }
          updateComposerEnabled();
        };

        const renderModels = () => {
          modelList.innerHTML = '';
          for (const m of st.models) {
            const btn = document.createElement('button');
            btn.className = 'modelBtn' + (m.integrated ? (m.selected ? ' on' : ' off') : ' disabled');
            if (m.authenticated) btn.classList.add('authed');
            if (st.viewTarget === m.key) btn.classList.add('active');

            const num = document.createElement('div');
            num.className = 'modelNum';
            num.style.background = 'linear-gradient(180deg,' + m.color + ', rgba(2,6,23,0.72))';

            const img = document.createElement('img');
            img.src = m.avatar_url;
            img.alt = m.name;
            num.appendChild(img);

            const badge = document.createElement('div');
            badge.className = 'modelSlotNum';
            badge.textContent = String(m.slot);
            num.appendChild(badge);

            const authDot = document.createElement('div');
            authDot.className = 'authDot';

            const lock = document.createElement('div');
            lock.className = 'lockIcon';

            const nudge = document.createElement('div');
            nudge.className = 'nudgeBtn';
            nudge.textContent = '💬';
            nudge.title = '让TA发言';
            nudge.addEventListener('click', async (ev) => {
              ev.stopPropagation();
              await apiPost('/api/models/nudge', { key: m.key });
            });

            btn.appendChild(num);
            btn.appendChild(authDot);
            btn.appendChild(lock);
            btn.appendChild(nudge);

            btn.title = m.integrated
              ? (m.name + '\\n点击：加入/退出群聊\\n绿点：已登录\\n💬：让TA发言')
              : (m.name + '\\n未接入（锁定）');

            btn.addEventListener('click', async () => {
              if (!m.integrated) return;
              // 关闭模型：直接切换
              if (m.selected) {
                await apiPost('/api/models/toggle', { key: m.key });
                await pollModels();
                rebuildTargets();
                return;
              }

              // 启用模型：先静默检测登录态（已登录则不弹窗）
              let authed = !!m.authenticated;
              if (!authed) {
                const chk = await apiPost('/api/models/login/check', { key: m.key });
                await pollModels();
                rebuildTargets();
                authed = !!(chk && chk.ok && chk.authenticated);
              }

              if (authed) {
                await apiPost('/api/models/toggle', { key: m.key });
                await pollModels();
                rebuildTargets();
                return;
              }

              // 未登录：沿用原流程（设置 pending_enable，并弹出登录提示）
              const res = await apiPost('/api/models/toggle', { key: m.key });
              await pollModels();
              rebuildTargets();
              if (res && res.need_auth) openAuth(m.key);
            });

            modelList.appendChild(btn);
          }
        };

        const shouldShowMsg = (msg) => {
          if (!msg) return false;
          if (msg.visibility === 'public') return true;
          // 单聊视图：显示当前目标模型的 shadow 消息（群聊视图仍只显示 public）
          if (msg.visibility === 'shadow' && st.viewTarget && st.viewTarget !== 'group') {
            return String(msg.model_key || '').toLowerCase() === String(st.viewTarget).toLowerCase();
          }
          return false;
        };

        const appendMsg = (msg) => {
          if (!shouldShowMsg(msg)) return;

          const row = document.createElement('div');
          let cls = 'msg-row ';
          if (msg.role === 'system') cls += 'system';
          else if (msg.role === 'user') cls += 'you';
          else cls += 'model';
          row.className = cls;

          if (msg.role !== 'system') {
            const av = document.createElement('div');
            av.className = 'avatar ' + (msg.role === 'user' ? 'you' : 'system');
            if (msg.role === 'user') {
              av.className = 'avatar you';
              av.textContent = '用户';
            } else {
              const m = st.models.find(x => x.key === msg.model_key);
              av.textContent = m ? String(m.slot) : 'AI';
              av.style.background = m ? ('linear-gradient(180deg,' + m.color + ', rgba(2,6,23,0.72))') : '';
            }
            row.appendChild(av);
          }

          const bub = document.createElement('div');
          bub.className = 'bubble';
          if (msg.role === 'user') bub.classList.add('user');
          if (msg.role === 'system') bub.classList.add('system');

          if (msg.role === 'model') {
            const m = st.models.find(x => x.key === msg.model_key);
            if (m) bub.style.borderColor = 'rgba(15,23,42,0.18)';
          }

          const meta = document.createElement('div');
          meta.className = 'bubble-meta';
          const name = document.createElement('div');
          name.className = 'name ' + (msg.role === 'user' ? 'you' : (msg.role === 'system' ? 'system' : ''));
          name.textContent = msg.speaker || '';
          const time = document.createElement('div');
          time.className = 'time';
          time.textContent = msg.ts || '';
          meta.appendChild(name);
          meta.appendChild(time);

          const text = document.createElement('div');
          text.className = 'bubble-text';
          text.innerHTML = esc(msg.text || '');

          bub.appendChild(meta);
          bub.appendChild(text);
          row.appendChild(bub);
          chat.appendChild(row);
        };

        const redrawMessages = () => {
          chat.innerHTML = '';
          for (const m of st.allMessages) {
            appendMsg(m);
          }
          maybeScroll();
        };

        const maybeScroll = () => {
          const el = chat;
          const nearBottom = (el.scrollHeight - el.scrollTop - el.clientHeight) < 140;
          if (nearBottom) el.scrollTop = el.scrollHeight;
        };

        const pollMessages = async () => {
          const res = await apiGet('/api/messages?after=' + st.lastId);
          if (!res || !res.ok) return;
          const list = res.messages || [];
          if (!list.length) return;
          for (const m of list) {
            st.lastId = Math.max(st.lastId, m.id || 0);
            st.allMessages.push(m);
            appendMsg(m);
          }
          countPill.textContent = String(st.lastId);
          maybeScroll();
        };

        const pollModels = async () => {
          const res = await apiGet('/api/models');
          if (!res || !res.ok) return;
          st.models = res.models || [];
          renderModels();
        };

        const pollState = async () => {
          const res = await apiGet('/api/state');
          if (!res || !res.ok) return;
          // Could render status in UI if needed.
        };

        const sendNow = async () => {
          const v = sendTarget.value;
          if (!v || v === '__sep__') {
            st.viewTarget = '';
            updateComposerEnabled();
            return;
          }
          const text = input.value || '';
          if (!text.trim()) return;
          const payload = { target: v, text };
          if (v === 'group') {
            const rv = parseInt((groupRounds && groupRounds.value) ? groupRounds.value : '-1', 10);
            payload.rounds = Number.isFinite(rv) ? rv : -1;
          }
          const res = await apiPost('/api/send', payload);
          if (!res || !res.ok) {
            const err = (res && res.error) ? String(res.error) : 'unknown';
            if (err === 'no_selected_models') {
              hintLeft.textContent = '发送失败：还没有已启用模型（先点左侧模型并完成登录）';
            } else if (err === 'target_not_selected') {
              hintLeft.textContent = '发送失败：单聊目标未启用，请先在左侧启用该模型';
            } else if (err === 'empty_text') {
              hintLeft.textContent = '发送失败：消息不能为空';
            } else {
              hintLeft.textContent = '发送失败：' + err;
            }
            hintLeft.style.color = 'var(--danger)';
            return;
          }
          input.value = '';
          updateComposerEnabled();
        };

        // events
        stopTopBtn && stopTopBtn.addEventListener('click', async () => {
          await apiPost('/api/debate/stop', {});
          hintLeft.textContent = '已请求停止自动轮聊';
          hintLeft.style.color = 'var(--warn)';
        });
        clearBtn && clearBtn.addEventListener('click', () => {
          input.value = '';
          input.focus();
        });
        sendBtn && sendBtn.addEventListener('click', sendNow);
        input && input.addEventListener('keydown', (ev) => {
          if (ev.key === 'Enter' && !ev.shiftKey) {
            ev.preventDefault();
            if (!sendBtn.disabled) sendNow();
          }
        });
        sendTarget && sendTarget.addEventListener('change', () => {
          st.viewTarget = sendTarget.value;
          updateComposerEnabled();
          redrawMessages();
        });
        groupRounds && groupRounds.addEventListener('change', updateComposerEnabled);

        authClose && authClose.addEventListener('click', closeAuth);
        authMask && authMask.addEventListener('click', (e) => { if (e.target === authMask) closeAuth(); });
        authOpen && authOpen.addEventListener('click', async () => {
          if (!st.authKey) return;
          await apiPost('/api/models/login/open', { key: st.authKey });
        });
        authCheck && authCheck.addEventListener('click', async () => {
          if (!st.authKey) return;
          const res = await apiPost('/api/models/login/check', { key: st.authKey });
          await pollModels();
          rebuildTargets();
          if (res && res.ok && res.authenticated) closeAuth();
          else alert('仍未检测到登录成功。请确认已登录并关闭欢迎弹窗后重试。');
        });

        const boot = async () => {
          await pollModels();
          rebuildTargets();
          updateComposerEnabled();
          renderModels();
          await pollMessages();
          setInterval(pollState, 900);
          setInterval(async () => { await pollModels(); rebuildTargets(); }, 1200);
          setInterval(pollMessages, 800);
        };

        boot();
      })();

      // === Legacy WebUI disabled ===
      if (false) {
      let lastId = 0;
      let paused = false;
      let total = 0;
      const allMessages = [];
      const unseen = { chatgpt: 0, gemini: 0 };
      let newBelow = 0;
      let sessionStarted = false;
      let startRequested = false;
      let modelSlots = [];
      let selectedKeys = new Set();

      const chat = document.getElementById('chat');
      const jumpBtn = document.getElementById('jumpBtn');
      const modelList = document.getElementById('modelList');
      const startSessionBtn = document.getElementById('startSessionBtn');
      const conn = document.getElementById('conn');
      const dot = document.getElementById('dot');
      const statusText = document.getElementById('statusText');
      const unseenPill = document.getElementById('unseenPill');
      const pauseBtn = document.getElementById('pauseBtn');
      const stopBtn = document.getElementById('stopBtn');
      const sendBtn = document.getElementById('sendBtn');
      const clearBtn = document.getElementById('clearBtn');
      const rulesInput = document.getElementById('rulesInput');
      const saveRulesBtn = document.getElementById('saveRulesBtn');
      const input = document.getElementById('input');
      const target = document.getElementById('target');
      const countPill = document.getElementById('countPill');

      function avatarText(cls) {
        if (cls === 'you') return '你';
        if (cls === 'chatgpt') return 'C';
        if (cls === 'gemini') return 'G';
        if (cls === 'system') return '!';
        return '?';
      }

      function speakerClass(s) {
        const x = (s || '').toLowerCase();
        if (x.includes('chatgpt')) return 'chatgpt';
        if (x.includes('gemini')) return 'gemini';
        if (x === 'you' || x === 'user' || x.includes('用户')) return 'you';
        if (x.includes('system') || x.includes('系统')) return 'system';
        return '';
      }

      function modelNumClass(key) {
        const k = (key || '').toLowerCase();
        if (k === 'chatgpt') return 'chatgpt';
        if (k === 'gemini') return 'gemini';
        return 'other';
      }

      function setUiEnabled(enabled) {
        input.disabled = !enabled;
        sendBtn.disabled = !enabled;
        target.disabled = !enabled;
        pauseBtn.disabled = !enabled;
        clearBtn.disabled = !enabled;
        if (!enabled) {
          input.placeholder = '先在左侧选择模型并点击「启动」...';
        } else {
          input.placeholder = '在这里插话（群主发言）...';
        }
      }

      function updateModelBadges() {
        if (!modelList) return;
        const badges = modelList.querySelectorAll('.badge');
        for (const b of badges) {
          const key = b.getAttribute('data-key') || '';
          let n = 0;
          if (key === 'chatgpt') n = unseen.chatgpt || 0;
          if (key === 'gemini') n = unseen.gemini || 0;
          if (n > 0) {
            b.textContent = String(n);
            b.classList.remove('hidden');
          } else {
            b.classList.add('hidden');
          }
        }
      }

      function applyTargetForKey(key) {
        const k = (key || '').toLowerCase();
        if (k === 'chatgpt') target.value = 'chatgpt';
        else if (k === 'gemini') target.value = 'gemini';
        else target.value = 'next';
        // 切换视图时清零对应未读
        const f = currentFilter();
        if (f.chatgpt) unseen.chatgpt = 0;
        if (f.gemini) unseen.gemini = 0;
        updateUnseenPill();
        updateModelBadges();
        rerender();
      }

      function renderModels() {
        if (!modelList) return;
        modelList.innerHTML = '';

        for (const slot of (modelSlots || [])) {
          const key = String(slot.key || '').toLowerCase();
          const name = String(slot.name || key || 'model');
          const slotNo = String(slot.slot || '');
          const enabled = !!slot.enabled;
          const selected = selectedKeys.has(key);

          const btn = document.createElement('button');
          btn.className = 'modelBtn';
          btn.setAttribute('type', 'button');
          btn.setAttribute('data-key', key);
          btn.title = enabled ? `${slotNo}. ${name}` : `${slotNo}. ${name}（待接入）`;

          if (!enabled) btn.classList.add('disabled');
          else if (selected) btn.classList.add('on');
          else btn.classList.add('off');

          const inner = document.createElement('div');
          inner.className = 'modelNum ' + modelNumClass(key);
          inner.textContent = slotNo;

          const badge = document.createElement('div');
          badge.className = 'badge hidden';
          badge.setAttribute('data-key', key);

          btn.appendChild(inner);
          btn.appendChild(badge);

          btn.addEventListener('click', async () => {
            if (!enabled) return;
            if (!sessionStarted && !startRequested) {
              // 选择阶段：点击表示“启用/禁用”
              if (selectedKeys.has(key)) selectedKeys.delete(key);
              else selectedKeys.add(key);
              try {
                const resp = await apiPost('/api/models/select', { keys: Array.from(selectedKeys) });
                if (resp && resp.selected_keys) selectedKeys = new Set(resp.selected_keys.map(x => String(x).toLowerCase()));
                if (resp && resp.slots) modelSlots = resp.slots;
                renderModels();
                updateModelBadges();
              } catch (e) {
                alert('更新模型选择失败：' + e + netHint(e));
              }
              return;
            }

            // 启动后：点击用于切换“查看焦点”（不改变启用集合）
            if (sessionStarted) applyTargetForKey(key);
          });

          modelList.appendChild(btn);
        }

        updateModelBadges();
      }

      async function loadModels() {
        try {
          const data = await apiGet('/api/models');
          sessionStarted = !!(data && data.session_started);
          startRequested = !!(data && data.start_requested);
          const keys = (data && data.selected_keys) ? data.selected_keys : [];
          selectedKeys = new Set((keys || []).map(x => String(x).toLowerCase()));
          modelSlots = (data && data.slots) ? data.slots : [];
          renderModels();
          if (startSessionBtn) {
            if (sessionStarted) {
              startSessionBtn.style.display = 'none';
            } else {
              startSessionBtn.style.display = 'inline-block';
              startSessionBtn.disabled = !!startRequested;
              startSessionBtn.textContent = startRequested ? '启动中' : '启动';
            }
          }
          setUiEnabled(sessionStarted);
        } catch (e) {
          // ignore
        }
      }

      function isNearBottom() {
        const threshold = 140;
        const remain = chat.scrollHeight - (chat.scrollTop + chat.clientHeight);
        return remain < threshold;
      }

      function updateJump() {
        if (!jumpBtn) return;
        if (newBelow > 0 && !isNearBottom()) {
          jumpBtn.classList.remove('hidden');
          jumpBtn.textContent = `↓ ${newBelow} 条新消息`;
        } else {
          jumpBtn.classList.add('hidden');
          newBelow = 0;
        }
      }

      function currentFilter() {
        const t = (target.value || 'next').toLowerCase();
        if (t === 'chatgpt') return { chatgpt: true, gemini: false };
        if (t === 'gemini') return { chatgpt: false, gemini: true };
        // next/both: show all
        return { chatgpt: true, gemini: true };
      }

      function matchesFilter(m) {
        const f = currentFilter();
        if (m.speaker === 'System' || m.speaker === 'You') return true;
        if (m.speaker === 'ChatGPT') return !!f.chatgpt;
        if (m.speaker === 'Gemini') return !!f.gemini;
        return true;
      }

      function appendMessage(m) {
        const cls = speakerClass(m.speaker);
        const row = document.createElement('div');
        row.className = 'msg-row ' + (cls || '');
        row.dataset.speaker = m.speaker || '';

        const prev = chat.lastElementChild;
        if (prev && prev.dataset && prev.dataset.speaker === row.dataset.speaker && cls !== 'system') {
          row.classList.add('continued');
        }

        if (cls === 'system') {
          row.className = 'msg-row system';
          const bubble = document.createElement('div');
          bubble.className = 'bubble system';
          const text = document.createElement('div');
          text.className = 'bubble-text';
          text.textContent = m.text;
          const time = document.createElement('div');
          time.className = 'sysTime';
          time.textContent = m.ts;
          bubble.appendChild(text);
          bubble.appendChild(time);
          row.appendChild(bubble);
          chat.appendChild(row);
          return;
        }

        const avatar = document.createElement('div');
        avatar.className = 'avatar ' + (cls || '');
        avatar.textContent = avatarText(cls);

        const bubble = document.createElement('div');
        bubble.className = 'bubble';

        const meta = document.createElement('div');
        meta.className = 'bubble-meta';

        const name = document.createElement('div');
        name.className = 'name ' + (cls || '');
        name.textContent = m.speaker;

        const time = document.createElement('div');
        time.className = 'time';
        time.textContent = m.ts;

        meta.appendChild(name);
        meta.appendChild(time);

        const text = document.createElement('div');
        text.className = 'bubble-text';
        text.textContent = m.text;

        bubble.appendChild(meta);
        bubble.appendChild(text);

        if (cls === 'you') {
          row.classList.add('you');
          row.appendChild(bubble);
          row.appendChild(avatar);
        } else {
          row.appendChild(avatar);
          row.appendChild(bubble);
        }

        chat.appendChild(row);
      }

      function updateUnseenPill() {
        unseenPill.textContent = `未读：ChatGPT ${unseen.chatgpt} | Gemini ${unseen.gemini}`;
      }

      function rerender() {
        chat.innerHTML = '';
        newBelow = 0;
        updateJump();
        for (const m of allMessages) {
          if (matchesFilter(m)) appendMessage(m);
        }
        scrollToBottom();
      }

      function scrollToBottom(smooth=false) {
        chat.scrollTo({ top: chat.scrollHeight, behavior: smooth ? 'smooth' : 'auto' });
        newBelow = 0;
        updateJump();
      }

      async function apiGet(url) {
        const r = await fetch(url, { cache: 'no-store' });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return await r.json();
      }

      async function apiPost(url, payload) {
        const r = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload || {}),
        });
        if (!r.ok) {
          let msg = 'HTTP ' + r.status;
          try {
            const j = await r.json();
            if (j && j.error) msg = msg + ': ' + String(j.error);
          } catch (e) {
            // ignore
          }
          throw new Error(msg);
        }
        return await r.json();
      }

      function netHint(e) {
        const s = String(e || '');
        if (
          s.includes('Failed to fetch') ||
          s.includes('NetworkError') ||
          s.includes('ERR_CONNECTION') ||
          s.includes('ECONN')
        ) {
          return '\\n\\n提示：后端未运行或已退出。请重新运行 start_webui.bat；如果仍失败，查看 .tmp\\\\webui.err.log。';
        }
        return '';
      }

      async function loadRules() {
        try {
          const data = await apiGet('/api/rules');
          if (data && typeof data.rules === 'string') {
            rulesInput.value = data.rules;
          }
        } catch (e) {
          // ignore
        }
      }

      async function poll() {
        try {
          const state = await apiGet('/api/state');
          statusText.textContent = state.status || '';
          paused = !!state.paused;
          pauseBtn.textContent = paused ? '继续' : '暂停';
          if (typeof state.session_started !== 'undefined' || typeof state.start_requested !== 'undefined') {
            const started = !!state.session_started;
            const requested = !!state.start_requested;
            if (started !== sessionStarted || requested !== startRequested) {
              sessionStarted = started;
              startRequested = requested;
              // 会话状态发生变化时，刷新模型栏与控件状态
              loadModels();
            }
          }

          const msgs = await apiGet('/api/messages?after=' + lastId);
          if (Array.isArray(msgs) && msgs.length) {
            dot.className = 'dot ok';
            conn.textContent = 'ok';
            const stick = isNearBottom();
            let appended = 0;
            for (const m of msgs) {
              allMessages.push(m);
              if (matchesFilter(m)) {
                appendMessage(m);
                appended += 1;
              } else if (m.speaker === 'ChatGPT') {
                unseen.chatgpt += 1;
              } else if (m.speaker === 'Gemini') {
                unseen.gemini += 1;
              }
              lastId = Math.max(lastId, m.id || lastId);
              total++;
            }
            countPill.textContent = String(total);
            updateUnseenPill();
            updateModelBadges();
            if (appended > 0) {
              if (stick) {
                scrollToBottom();
              } else {
                newBelow += appended;
                updateJump();
              }
            }
          } else {
            dot.className = 'dot ok';
            conn.textContent = 'ok';
          }
        } catch (e) {
          dot.className = 'dot bad';
          conn.textContent = 'error';
          statusText.textContent = '后端断开：请重新运行 start_webui.bat（必要时查看 .tmp\\\\webui.err.log）';
          setUiEnabled(false);
        } finally {
          setTimeout(poll, 800);
        }
      }

      async function send() {
        const text = (input.value || '').trim();
        if (!text) return;
        const to = target.value || 'next';
        input.value = '';
        try {
          await apiPost('/api/send', { text, to });
        } catch (e) {
          alert('发送失败：' + e + netHint(e));
        }
      }

      pauseBtn.addEventListener('click', async () => {
        try {
          await apiPost('/api/pause', { paused: !paused });
        } catch (e) {
          alert('操作失败：' + e + netHint(e));
        }
      });

      stopBtn.addEventListener('click', async () => {
        if (!confirm('确定要停止脚本吗？')) return;
        try {
          await apiPost('/api/stop', {});
        } catch (e) {
          alert('停止失败：' + e + netHint(e));
        }
      });

      if (startSessionBtn) {
        startSessionBtn.addEventListener('click', async () => {
          if (sessionStarted || startRequested) return;
          startSessionBtn.disabled = true;
          const oldText = startSessionBtn.textContent;
          startSessionBtn.textContent = '启动中';
          try {
            await apiPost('/api/session/start', {});
           await loadModels();
         } catch (e) {
            alert('启动失败：' + e + netHint(e));
            startSessionBtn.disabled = false;
            startSessionBtn.textContent = oldText;
          }
        });
      }

      saveRulesBtn.addEventListener('click', async () => {
        try {
          await apiPost('/api/rules', { rules: rulesInput.value || '' });
          alert('规则已应用（下一轮生效）');
        } catch (e) {
          alert('应用失败：' + e + netHint(e));
        }
      });

      sendBtn.addEventListener('click', send);
      target.addEventListener('change', () => {
        const f = currentFilter();
        if (f.chatgpt) unseen.chatgpt = 0;
        if (f.gemini) unseen.gemini = 0;
        updateUnseenPill();
        updateModelBadges();
        rerender();
      });
      clearBtn.addEventListener('click', () => {
        chat.innerHTML = '';
        total = 0;
        allMessages.length = 0;
        unseen.chatgpt = 0;
        unseen.gemini = 0;
        newBelow = 0;
        countPill.textContent = '0';
        updateUnseenPill();
        updateModelBadges();
        updateJump();
      });

      input.addEventListener('keydown', (ev) => {
        // Chat-like:
        // - Enter: send
        // - Shift+Enter: newline
        if (ev.key === 'Enter' && !ev.shiftKey) {
          ev.preventDefault();
          send();
        }
      });

      chat.addEventListener('scroll', () => {
        if (isNearBottom()) {
          newBelow = 0;
          updateJump();
        }
      });

      if (jumpBtn) {
        jumpBtn.addEventListener('click', () => scrollToBottom(true));
      }

      // 默认：未启动前禁止输入；模型选择 + 启动后再放开
      setUiEnabled(false);
      updateUnseenPill();
      loadRules();
      loadModels();
      poll();
      }
    </script>
  </body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    state: SharedState  # injected
    worker: Any  # injected

    def _send_json(self, obj: Any, status: int = 200) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, html: str) -> None:
        raw = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            data = json.loads(body)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
            return

        if parsed.path == "/api/state":
            self._send_json(self.state.get_state())
            return

        if parsed.path == "/api/models":
            self._send_json(self.state.get_models())
            return

        if parsed.path == "/api/messages":
            q = urllib.parse.parse_qs(parsed.query)
            after_s = (q.get("after") or ["0"])[0]
            try:
                after_id = int(after_s)
            except Exception:
                after_id = 0
            self._send_json(self.state.get_messages_after(after_id))
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        data = self._read_json_body()

        if parsed.path == "/api/models/toggle":
            key = str(data.get("key") or "")
            self._send_json(self.state.toggle_selected(key))
            return

        if parsed.path == "/api/models/login/open":
            key = str(data.get("key") or "")
            ev = threading.Event()
            box: dict[str, Any] = {}
            self.state.inbox.put({"kind": "login_open", "key": key, "_ev": ev, "_reply": box})
            ev.wait(timeout=25)
            self._send_json(box or {"ok": True})
            return

        if parsed.path == "/api/models/login/check":
            key = str(data.get("key") or "")
            ev = threading.Event()
            box: dict[str, Any] = {}
            self.state.inbox.put({"kind": "login_check", "key": key, "_ev": ev, "_reply": box})
            ev.wait(timeout=25)
            self._send_json(box or {"ok": True, "authenticated": False})
            return

        if parsed.path == "/api/models/nudge":
            key = str(data.get("key") or "")
            self.state.inbox.put({"kind": "nudge", "key": key})
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/send":
            text = core.normalize_text(str(data.get("text") or ""))
            target = str(data.get("target") or "").strip().lower()
            if not text:
                self._send_json({"ok": False, "error": "empty text"}, status=400)
                return
            if not target:
                self._send_json({"ok": False, "error": "missing target"}, status=400)
                return
            rounds_raw = data.get("rounds", GROUP_DEFAULT_ROUNDS)
            self.state.inbox.put({"kind": "send", "target": target, "text": text, "rounds": rounds_raw})
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/debate/stop":
            self.state.request_round_stop()
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/stop":
            self.state.request_stop()
            self.state.inbox.put({"kind": "stop"})
            self._send_json({"ok": True})
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # 静默 httpserver 日志，避免刷屏
        return


def start_server(state: SharedState, host: str, port: int) -> ThreadingHTTPServer:
    _Handler.state = state
    _Handler.worker = None
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, name="webui", daemon=True)
    t.start()
    return httpd


def _split_forward_summary(text: str) -> tuple[str, str]:
    t = core.normalize_text(text)
    if not t:
        return "", ""
    m = re.search(r"【转发摘要】", t)
    if not m:
        return t, ""
    main = t[: m.start()].strip()
    tail = t[m.end() :].strip()
    summary = core.normalize_text(tail)
    if len(summary) > 200:
        summary = summary[:200].rstrip() + "…"
    return main, summary


def _clip_text(s: str, max_chars: int) -> str:
    s = core.normalize_text(s)
    if len(s) <= max_chars:
        return s
    head = s[: int(max_chars * 0.7)].rstrip()
    tail = s[-int(max_chars * 0.2) :].lstrip()
    return f"{head}…（略）…{tail}".strip()


def _format_msg_for_context(msg: UiMessage) -> str:
    # 优先使用模型的【转发摘要】压缩上下文，避免“已截断”。
    if msg.role == "model":
        main, summary = _split_forward_summary(msg.text)
        body = summary or _clip_text(main or msg.text, 240)
    else:
        body = _clip_text(msg.text, 260)
    prefix = "系统" if msg.role == "system" else msg.speaker
    return f"{prefix}: {body}".strip()


def build_model_prompt(history: list[UiMessage], instruction: str) -> str:
    inst = core.normalize_text(instruction) or "请基于当前对话提出观点/反驳/补充。"
    header = DEFAULT_RULES + "\n\n" + RULES_REMINDER + "\n\n【对话上下文】\n"
    footer = "\n\n【本轮群主消息】\n" + inst.strip() + "\n"

    used = len(header) + len(footer)
    budget = MAX_MODEL_PROMPT_CHARS
    lines: list[str] = []

    for msg in reversed(history[-MAX_CONTEXT_MESSAGES:]):
        line = _format_msg_for_context(msg)
        if not line:
            continue
        if used + len(line) + 1 > budget:
            break
        lines.append(line)
        used += len(line) + 1

    lines.reverse()
    ctx = "\n".join(lines) if lines else "（暂无历史）"
    return header + ctx + footer


class Worker:
    """Playwright 单线程执行器：所有网页登录/发送/提取都在这里串行化，避免线程安全问题。"""

    def __init__(self, state: SharedState) -> None:
        self.state = state
        self._thread = threading.Thread(target=self._run, name="pw-worker", daemon=True)
        self._pw = None
        self._sp = None
        self._adapters: dict[str, ModelAdapter] = {}

    def start(self) -> None:
        self._thread.start()

    def _ensure_playwright(self) -> None:
        if self._sp is not None and self._pw is not None:
            return
        self.state.set_status("starting_playwright")
        self._sp = sync_playwright().start()
        self._pw = self._sp
        self.state.set_status("idle")

    def _get_adapter(self, key: str) -> Optional[ModelAdapter]:
        key = (key or "").strip().lower()
        if not key:
            return None
        if key in self._adapters:
            return self._adapters[key]
        meta = next((m for m in MODEL_METAS if m.key == key), None)
        if not meta or not meta.integrated:
            return None
        ad = build_adapter(meta)
        self._adapters[key] = ad
        return ad

    @staticmethod
    def _safe_reply(action: dict[str, Any], payload: dict[str, Any]) -> None:
        ev = action.get("_ev")
        box = action.get("_reply")
        if isinstance(box, dict):
            box.clear()
            box.update(payload)
        if isinstance(ev, threading.Event):
            ev.set()

    def _run(self) -> None:
        while not self.state.should_stop():
            try:
                action = self.state.inbox.get(timeout=0.4)
            except queue.Empty:
                continue

            kind = str(action.get("kind") or "")
            try:
                if kind == "login_open":
                    self._handle_login_open(action)
                elif kind == "login_check":
                    self._handle_login_check(action)
                elif kind == "send":
                    self._handle_send(action)
                elif kind == "nudge":
                    self._handle_nudge(action)
                elif kind == "stop":
                    self.state.request_stop()
                else:
                    pass
            except Exception as exc:
                self.state.add_system(f"后台执行异常: {exc}")
                traceback.print_exc()
                self._safe_reply(action, {"ok": False, "error": str(exc)})

        # cleanup
        try:
            self.state.set_status("stopping")
            for ad in list(self._adapters.values()):
                ad.close()
            self._adapters.clear()
        finally:
            try:
                if self._sp is not None:
                    self._sp.stop()
            except Exception:
                pass
            self._sp = None
            self._pw = None
            self.state.set_status("stopped")

    def _handle_login_open(self, action: dict[str, Any]) -> None:
        key = str(action.get("key") or "").strip().lower()
        ad = self._get_adapter(key)
        if ad is None:
            self._safe_reply(action, {"ok": False, "error": "unknown_or_not_integrated"})
            return
        self._ensure_playwright()
        assert self._pw is not None

        self.state.set_status(f"opening_login:{key}")
        page = ad.ensure_page(self._pw)
        try:
            page.goto(ad.meta.url, wait_until="domcontentloaded")
        except Exception:
            pass
        m = self.state.get_model(key)
        if m and m.key == "deepseek":
            self._ensure_deepseek_dialog(page)
        ad.bring_to_front()
        self.state.set_status("idle")
        self._safe_reply(action, {"ok": True})

    def _handle_login_check(self, action: dict[str, Any]) -> None:
        key = str(action.get("key") or "").strip().lower()
        ad = self._get_adapter(key)
        if ad is None:
            self._safe_reply(action, {"ok": False, "error": "unknown_or_not_integrated"})
            return
        self._ensure_playwright()
        assert self._pw is not None

        self.state.set_status(f"checking_login:{key}")
        ad.ensure_page(self._pw)
        ok = ad.is_authenticated_now()
        self.state.set_authenticated(key, ok)
        if ok:
            self.state.mark_pending_enable_done(key)
        self.state.set_status("idle")
        self._safe_reply(action, {"ok": True, "authenticated": bool(ok)})

    def _ensure_deepseek_dialog(self, page: Any) -> None:
        """DeepSeek occasionally lands on list-only view; open a chat panel for stability."""
        try:
            if page.locator("div.ds-message").count() > 0:
                return
        except Exception:
            return

        clicked = False
        # Prefer opening the latest existing thread (keeps context visible).
        for sel in ("div._3098d02", "div[class*='_3098d02']"):
            try:
                loc = page.locator(sel)
                cnt = loc.count()
                for i in range(cnt):
                    node = loc.nth(i)
                    txt = core.normalize_text(node.inner_text())
                    if not txt:
                        continue
                    if txt in {"今天", "7 天内", "30 天内", "2026-01", "2025-12", "2025-11"}:
                        continue
                    node.click(timeout=2500)
                    clicked = True
                    break
                if clicked:
                    break
            except Exception:
                continue

        # Fallback: open a new chat.
        if not clicked:
            candidates = [
                page.get_by_role("button", name=re.compile(r"开启新对话|新对话|new chat", re.I)),
                page.get_by_text(re.compile(r"开启新对话|新对话|new chat", re.I)),
            ]
            for loc in candidates:
                node = core.pick_visible(loc, prefer_last=False)
                if node is None:
                    continue
                try:
                    node.click(timeout=3000)
                    clicked = True
                    break
                except Exception:
                    continue

        if clicked:
            time.sleep(0.8)

    def _ensure_chat_surface(self, ad: ModelAdapter, m: ModelRuntime) -> Any:
        assert self._pw is not None
        page = ad.ensure_page(self._pw)

        # Avoid forcing home navigation on every turn; it can hide thread panel.
        if ad.find_input() is None:
            try:
                page.goto(ad.meta.url, wait_until="domcontentloaded")
            except Exception:
                pass
            time.sleep(0.5)

        if m.key == "deepseek":
            self._ensure_deepseek_dialog(page)

        if ad.find_input() is None:
            # Generic recovery for popups/overlay.
            try:
                page.keyboard.press("Escape")
            except Exception:
                pass
            try:
                page.reload(wait_until="domcontentloaded")
            except Exception:
                pass
            time.sleep(0.6)

        if ad.find_input() is None:
            raise RuntimeError(f"{m.name} 对话输入框不可用（可能被弹窗遮挡或未进入会话）")
        return page

    def _run_model_turn(self, key: str, instruction: str, *, visibility: str) -> tuple[bool, str]:
        m = self.state.get_model(key)
        if not m or not m.integrated:
            self.state.add_system(f"目标模型不可用: {key}")
            return False, ""
        if not m.authenticated:
            self.state.add_system(f"{m.name} 未登录：请先点模型 -> 打开登录窗口 -> 重新检测")
            return False, ""

        ad = self._get_adapter(key)
        if ad is None:
            self.state.add_system(f"适配器缺失: {m.name}")
            return False, ""

        try:
            self.state.set_status(f"sending:{key}")
            self._ensure_chat_surface(ad, m)

            history = self.state.get_all_messages()
            prompt = build_model_prompt(history, instruction)

            before = ad.snapshot_conversation()
            ad.send_user_text(prompt)
            reply = core.normalize_text(ad.wait_reply_and_extract(before, timeout_s=MODEL_REPLY_TIMEOUT_S))
            if not reply:
                reply = "（未能提取到回复，可能仍在生成中或页面结构变化）"

            self.state.add_message(
                "model",
                m.name,
                reply,
                visibility=visibility,
                model_key=key,
            )
            return True, reply
        except Exception as exc:
            self.state.add_system(f"{m.name} 本轮失败：{exc}")
            return False, ""

    def _extract_target_keys_from_text(self, text: str, candidates: list[str]) -> list[str]:
        raw = core.normalize_text(text).lower()
        if not raw:
            return []
        compact = raw.replace(" ", "")
        found: list[str] = []
        for key in candidates:
            m = self.state.get_model(key)
            if not m:
                continue

            aliases: set[str] = set(MODEL_MENTION_ALIASES.get(key, ()))
            aliases.add(key.lower())
            name = core.normalize_text(m.name).lower()
            if name:
                aliases.add(name)
                aliases.add(re.sub(r"[（(].*?[)）]", "", name).strip())

            matched = False
            for alias in aliases:
                a = core.normalize_text(alias).lower()
                if not a:
                    continue
                cands = {a, a.replace(" ", "")}
                for tok in cands:
                    if not tok:
                        continue
                    # Encourage explicit mention with @; also allow plain-name reference.
                    if f"@{tok}" in compact or tok in compact:
                        matched = True
                        break
                if matched:
                    break
            if matched and key not in found:
                found.append(key)
        return found

    def _latest_public_model_message(self, *, exclude_key: Optional[str] = None) -> Optional[UiMessage]:
        msgs = self.state.get_all_messages()
        for msg in reversed(msgs):
            if msg.role != "model":
                continue
            if msg.visibility != "public":
                continue
            if exclude_key and msg.model_key == exclude_key:
                continue
            return msg
        return None

    def _handle_send(self, action: dict[str, Any]) -> None:
        target = str(action.get("target") or "").strip().lower()
        text = core.normalize_text(str(action.get("text") or ""))
        if not text:
            self._safe_reply(action, {"ok": False, "error": "empty_text"})
            return

        if target == "group":
            raw_rounds = action.get("rounds", GROUP_DEFAULT_ROUNDS)
            try:
                rounds = int(raw_rounds)
            except Exception:
                rounds = GROUP_DEFAULT_ROUNDS
            if rounds == 0:
                rounds = 1
            continuous = rounds < 0
            max_rounds = GROUP_CONTINUOUS_MAX_ROUNDS if continuous else max(1, rounds)

            keys = self.state.selected_keys()
            if not keys:
                self.state.add_system("没有已启用模型：请先在左侧启用至少 1 个模型。")
                self._safe_reply(action, {"ok": False, "error": "no_selected_models"})
                return
            visibility = "public"
            thread_key: Optional[str] = None
            self.state.clear_round_stop()
        else:
            if not self.state.is_selected(target):
                self.state.add_system("单聊目标未启用：请先在左侧启用该模型。")
                self._safe_reply(action, {"ok": False, "error": "target_not_selected"})
                return
            keys = [target]
            visibility = "shadow"
            thread_key = target

        self.state.add_message("user", "用户", text, visibility=visibility, model_key=thread_key)

        self._ensure_playwright()
        assert self._pw is not None

        if target != "group":
            self._run_model_turn(keys[0], text, visibility=visibility)
            self.state.set_status("idle")
            self._safe_reply(action, {"ok": True})
            return

        focus_keys: Optional[set[str]] = None
        focus_idle_rounds = 0
        round_no = 0
        while True:
            if self.state.should_round_stop():
                self.state.add_system("已停止自动轮聊。")
                break

            current_keys = self.state.selected_keys()
            if not current_keys:
                self.state.add_system("轮聊结束：当前没有已启用模型。")
                break
            if round_no >= 1 and len(current_keys) < 2:
                self.state.add_system("轮聊结束：当前仅 1 个模型，无法继续轮流发言。")
                break

            if focus_keys:
                focus_keys = {k for k in focus_keys if k in current_keys}
                if len(focus_keys) < 2:
                    focus_keys = None

            if focus_keys:
                talk_keys = [k for k in current_keys if k in focus_keys]
            else:
                talk_keys = list(current_keys)

            round_no += 1
            any_success = False
            mention_switched = False
            for k in talk_keys:
                if self.state.should_round_stop():
                    break

                if round_no == 1:
                    turn_instruction = (
                        f"{text}\n"
                        "补充：如果你认为应让某个模型加入当前讨论，请在结尾显式写 @模型名。"
                    )
                else:
                    last_msg = self._latest_public_model_message(exclude_key=k)
                    if last_msg is None:
                        turn_instruction = (
                            "请继续群聊，给出补充/反驳，并抛出一个追问。"
                            "如果你希望某个模型加入，请显式写 @模型名。"
                        )
                    else:
                        compact = core.normalize_text(last_msg.text)
                        if len(compact) > 280:
                            compact = compact[:280] + "…"
                        scope_line = ""
                        if focus_keys:
                            observers = [x for x in current_keys if x not in focus_keys]
                            if observers:
                                names = []
                                for ok in observers:
                                    om = self.state.get_model(ok)
                                    names.append(om.name if om else ok)
                                scope_line = "旁听模型：" + "、".join(names) + "。如需其加入请显式 @点名。\n"
                        turn_instruction = (
                            f"这是群聊第 {round_no} 轮。\n"
                            f"上一位发言（{last_msg.speaker}）：{compact}\n"
                            f"{scope_line}"
                            "请你基于当前对话提出观点/反驳/补充，并追加一个追问。"
                        )

                ok, reply_text = self._run_model_turn(k, turn_instruction, visibility="public")
                any_success = any_success or ok
                if ok and reply_text:
                    targets = self._extract_target_keys_from_text(
                        reply_text,
                        [x for x in current_keys if x != k],
                    )
                    if targets:
                        new_focus = set([k, *targets])
                        if focus_keys != new_focus:
                            ordered = [x for x in current_keys if x in new_focus]
                            names = []
                            for kk in ordered:
                                mm = self.state.get_model(kk)
                                names.append(mm.name if mm else kk)
                            self.state.add_system("焦点讨论切换：" + " ↔ ".join(names))
                        focus_keys = new_focus
                        focus_idle_rounds = 0
                        mention_switched = True
                core.jitter("模型轮转间隔")

            if self.state.should_round_stop():
                self.state.add_system("已停止自动轮聊。")
                break
            if not any_success:
                self.state.add_system("轮聊结束：本轮没有模型成功回复。")
                break
            if focus_keys and not mention_switched:
                focus_idle_rounds += 1
                if focus_idle_rounds >= FOCUS_RECOVERY_ROUNDS:
                    focus_keys = None
                    focus_idle_rounds = 0
                    self.state.add_system("焦点讨论结束，恢复全员轮聊。")

            if round_no >= max_rounds:
                if continuous:
                    self.state.add_system(f"已达到连续轮聊安全上限（{GROUP_CONTINUOUS_MAX_ROUNDS} 轮），已自动停止。")
                break

        self.state.set_status("idle")
        self._safe_reply(action, {"ok": True})

    def _handle_nudge(self, action: dict[str, Any]) -> None:
        key = str(action.get("key") or "").strip().lower()
        if not self.state.is_selected(key):
            self._safe_reply(action, {"ok": False, "error": "not_selected"})
            return
        m = self.state.get_model(key)
        if not m or not m.integrated:
            self._safe_reply(action, {"ok": False, "error": "unknown_or_not_integrated"})
            return
        if not m.authenticated:
            self._safe_reply(action, {"ok": False, "error": "not_authenticated"})
            return
        ad = self._get_adapter(key)
        if ad is None:
            self._safe_reply(action, {"ok": False, "error": "adapter_missing"})
            return

        self._ensure_playwright()
        assert self._pw is not None

        self.state.add_system(f"已请求 {m.name} 发言")
        self.state.set_status(f"nudge:{key}")

        self._ensure_chat_surface(ad, m)

        history = self.state.get_all_messages()
        instruction = "请基于当前对话提出观点/反驳/补充，抓重点，像人聊天，尽量短。"
        prompt = build_model_prompt(history, instruction)

        before = ad.snapshot_conversation()
        ad.send_user_text(prompt)
        reply = core.normalize_text(ad.wait_reply_and_extract(before, timeout_s=MODEL_REPLY_TIMEOUT_S)) or "（未能提取到回复）"
        self.state.add_message("model", m.name, reply, visibility="public", model_key=key)
        self.state.set_status("idle")
        self._safe_reply(action, {"ok": True})


def run_webui_app(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, *, no_open: bool = False) -> None:
    core.disable_windows_console_quickedit()

    state = SharedState()
    worker = Worker(state)
    worker.start()

    _Handler.state = state
    _Handler.worker = worker

    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.daemon_threads = True

    url = f"http://{host}:{port}/"
    state.set_status(f"webui ready: {url}")
    state.add_system(f"Web UI 已启动：{url}")
    state.add_system("请在左侧选择要启用的模型（未登录会弹出登录提示）。")
    state.add_system("底部请选择发送目标：请选择 / 群聊 / 单聊（仅已启用模型）。")

    if not no_open:
        _open_webui_window(url)

    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        state.request_stop()
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass


def _format_forward(speaker: str, text: str) -> str:
    # 给另一位模型看的消息，带上来源，增强“群聊”一致性
    return f"【{speaker}】\n{text}".strip()


def _format_user(text: str) -> str:
    # 给模型看的“群主插话”，明确这是用户说的
    return f"【用户】\n{text}".strip()

def _pick_forward_payload(full_reply: str) -> str:
    """
    互怼转发时不直接把“整段长回复”扔给对方（会越来越长，且很难抓重点）。
    优先从回复里提取【转发摘要】；没有则做一个保守截取。
    """
    text = core.normalize_text(full_reply)
    if not text:
        return ""

    # 取最后一个【转发摘要】（模型有时会重复输出）
    matches = list(re.finditer(r"【转发摘要】\s*[:：]?\s*", text))
    if matches:
        start = matches[-1].end()
        tail = text[start:].strip()
        # 遇到下一个“【xxx】”段落就截断，避免把正文又带进去
        parts = re.split(r"\n\s*【[^】]{1,12}】", tail, maxsplit=1)
        summary = (parts[0] if parts else tail).strip()
        if summary:
            if len(summary) > FORWARD_MAX_CHARS:
                return summary[:FORWARD_MAX_CHARS] + "\n...(省略)"
            return summary

    # fallback：抓前几行要点/结论
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    compact = "\n".join(lines[:10]).strip()
    if len(compact) > FORWARD_MAX_CHARS:
        return compact[:FORWARD_MAX_CHARS] + "\n...(省略)"
    return compact


def _strip_forward_summary(full_reply: str) -> str:
    """
    为了让 UI 里看起来更像“正常聊天”，把末尾的【转发摘要】从展示文本中移除。
    该摘要仍会用于模型间转发（见 _pick_forward_payload）。
    """
    text = core.normalize_text(full_reply)
    if not text:
        return ""
    matches = list(re.finditer(r"【转发摘要】\s*[:：]?\s*", text))
    if not matches:
        return text
    body = text[: matches[-1].start()].rstrip()
    return body or text


def _compose_shadow_sync_message(source_site: str, user_text: str, source_reply: str) -> str:
    """
    把“当前主对话模型”的这一轮（用户 + 主模型回复）同步给另一个模型，让它在后台生成自己的回复。

    重点：
    - 用户当前不一定会立刻看到这个模型的回复（UI 会隐藏/延后显示），因此提示模型不要假设用户已读。
    - 同时要求回复尽量简洁、要点化，减少后续“上下文补齐/截断”的痛点。
    """
    user_text = core.normalize_text(user_text)
    source_reply = core.normalize_text(source_reply)

    # 这段提示词是“用户消息”的一部分（网页端无法注入 system prompt），尽量短且可控。
    sys_hint = "\n".join(
        [
            "【系统提示（旁听同步）】",
            "- 你现在处于“隐藏回复”模式：你的回复不会立刻展示给用户，用户稍后才会查看。",
            "- 用户可能尚未阅读你之前的回复：不要用“如我上面所说”等依赖已读的指代；如需引用，请简要重述关键点。",
            "- 像正常人聊天：自然、有温度，但要短；不要用 1/2/3 模板。",
            "- 允许联网：仅在需要最新事实/数据时联网；只补充事实，不要复述网页当结论；不要贴 URL。",
            "- 抓重点：先用 1 句复述对方核心主张，再用 2-4 句回应（反驳/补充）。",
            "- 末尾加【转发摘要】<=120字，只写关键观点。",
            "- 不要在回复中提及“隐藏/旁听/未读”等元信息，直接正常回答即可。",
        ]
    ).strip()

    body = "\n\n".join(
        [
            sys_hint,
            f"下面是用户与 {source_site} 的最新对话：",
            _format_user(user_text),
            _format_forward(source_site, source_reply),
            "请你现在给用户的回复：",
        ]
    ).strip()
    return body


def run_webui_duel(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    core.disable_windows_console_quickedit()

    state = SharedState()
    httpd = start_server(state, host=host, port=port)
    url = f"http://{host}:{port}/"
    state.set_status(f"webui ready: {url}")
    state.add_message("System", f"Web UI 已启动：{url}")

    _open_webui_window(url)

    # 先启动 UI，再等待你在 UI 左侧选择模型并点击“启动”，然后才打开对应网页
    state.add_message("System", "请先在最左侧选择要使用的模型槽位，然后点击「启动」。")
    state.add_message("System", "为避免风控与不必要的加载，脚本会在「启动」之后才打开对应网页。")
    state.set_status("waiting session start (select models + click Start)")
    while not state.should_stop():
        if state.wait_for_session_start(timeout_s=1):
            break
    if state.should_stop():
        state.set_status("stopped")
        try:
            httpd.shutdown()
        except Exception:
            pass
        return

    info = state.get_models()
    selected = list(info.get("selected_keys") or [])
    state.add_message("System", f"已选择模型：{', '.join(selected) if selected else '（空）'}")

    user_data_dir = str(Path("./user_data").resolve())
    Path(user_data_dir).mkdir(parents=True, exist_ok=True)

    with core.sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=False,
            viewport={"width": 1920, "height": 1080},
            args=["--disable-blink-features=AutomationControlled"],
        )
        context.set_default_timeout(15000)

        page_chatgpt = context.pages[0] if context.pages else context.new_page()
        page_gemini = context.new_page()

        state.set_status("opening pages")
        core.log("打开 ChatGPT 页面...")
        page_chatgpt.goto(core.CHATGPT_URL, wait_until="domcontentloaded")
        core.log("打开 Gemini 页面...")
        page_gemini.goto(core.GEMINI_URL, wait_until="domcontentloaded")

        # 登录检查（需要用户手动）
        state.set_status("waiting login (manual)")
        page_chatgpt.bring_to_front()
        core.wait_for_login(page_chatgpt, "ChatGPT")
        state.add_message("System", "ChatGPT 登录检查通过（检测到输入框）")

        page_gemini.bring_to_front()
        core.wait_for_login(page_gemini, "Gemini")
        state.add_message("System", "Gemini 登录检查通过（检测到输入框）")

        # 到这里为止：两边网页都已打开且通过登录检查，可以正式开始群聊（放开 WebUI 输入框）。
        state.mark_session_started()

        next_site = "ChatGPT"
        # 不再自动发送“你好，请开始你的论述。”：
        # 第一条消息由群主（你）在 WebUI 输入框里发出。
        pending_text: Optional[str] = None
        pending_upto_mid = 0  # pending_text 对应的群聊消息 id（用于已同步游标）
        pending_user_plain = ""  # 仅用于“单聊时后台同步给另一模型”的提示词拼接
        shadow_sync_to: Optional[str] = None  # 单聊模式下：主模型回复后，把这一轮同步给哪一边
        auto_duel = True  # 仅当目标为“发给下一位”时，才持续把回复转发给另一边
        # 记录每个模型“已经看过”的群聊消息 id，用于切换目标时自动补上下文
        seen_upto: dict[str, int] = {"ChatGPT": 0, "Gemini": 0}
        state.add_message("System", "请在 WebUI 里输入群主第一句话开始对话。")

        # 规则注入：记录每个站点最后一次注入的 rules version，避免每条都重复长规则
        rules_version_sent: dict[str, int] = {"ChatGPT": 0, "Gemini": 0}

        def _with_rules(site: str, msg: str) -> str:
            info = state.get_rules()
            rules = str(info.get("rules") or "").strip()
            version = int(info.get("version") or 0)
            body = (msg or "").strip()
            if not body:
                return ""
            if not rules:
                return body
            if version > rules_version_sent.get(site, 0):
                rules_version_sent[site] = version
                return f"{rules}\n\n{body}".strip()
            return f"{RULES_REMINDER}\n\n{body}".strip()

        # 后台同步队列：当你“只跟 A 聊”时，把 (用户+主模型回复) 的这一轮同步给 B，让 B 也持续跟上上下文。
        # B 的回复会写入消息流，但前端会根据 target 下拉框进行过滤（即：你不选 B 时看不到）。
        shadow_queue: dict[str, list[tuple[str, int]]] = {"ChatGPT": [], "Gemini": []}  # (text, upto_mid)
        shadow_inflight: dict[str, Optional[dict[str, Any]]] = {"ChatGPT": None, "Gemini": None}

        def pump_shadow_once() -> bool:
            """
            空闲时做一次“后台同步”推进：
            - 优先尝试收割已完成的后台回复
            - 然后再发送一条排队中的同步消息（一次只处理一个动作，避免长时间占用主循环）
            """
            # 1) Try collect (non-blocking checks first; only read stable text when likely done)
            for site in ("ChatGPT", "Gemini"):
                infl = shadow_inflight.get(site)
                if not infl:
                    continue

                page = page_chatgpt if site == "ChatGPT" else page_gemini

                try:
                    if site == "ChatGPT":
                        cont = core.find_chatgpt_continue_button(page)
                        if core.safe_is_enabled(cont):
                            core.warn("后台同步：检测到 ChatGPT Continue generating，自动点击继续")
                            core.jitter("后台同步点击 Continue generating 前")
                            cont.click(timeout=8000)
                            core.jitter("后台同步点击 Continue generating 后")
                            return True

                        stop_visible = core.find_chatgpt_stop_button(page) is not None
                        if stop_visible:
                            continue

                        prev_count = int(infl.get("prev_count") or 0)
                        prev_last = str(infl.get("prev_last") or "")
                        cur_count = core.count_chatgpt_assistant_messages(page)
                        last_text = core.normalize_text(core.extract_chatgpt_last_reply(page))
                        has_new = (cur_count > prev_count) or (last_text and last_text != prev_last)
                        if not has_new:
                            continue

                        reply = core.read_stable_text(
                            lambda: core.extract_chatgpt_last_reply(page), "ChatGPT", rounds=6
                        )
                        display = _strip_forward_summary(reply)
                        state.add_message("ChatGPT", display or "（ChatGPT 回复为空或提取失败）")
                        shadow_inflight[site] = None
                        return True

                    else:
                        stop_visible = core.find_gemini_stop_button(page) is not None
                        if stop_visible:
                            continue

                        prev_count = int(infl.get("prev_count") or 0)
                        cur_count = core.count_gemini_responses(page)
                        if cur_count <= prev_count:
                            continue

                        reply = core.read_stable_text(lambda: core.extract_gemini_last_reply(page), "Gemini", rounds=6)
                        display = _strip_forward_summary(reply)
                        state.add_message("Gemini", display or "（Gemini 回复为空或提取失败）")
                        shadow_inflight[site] = None
                        return True
                except Exception:
                    # 后台同步失败不应阻塞主流程；保留 inflight，后续再试
                    continue

            # 2) Try send one queued shadow sync message
            for site in ("ChatGPT", "Gemini"):
                if shadow_inflight.get(site) is not None:
                    continue
                q = shadow_queue.get(site) or []
                if not q:
                    continue

                text, upto_mid = q.pop(0)
                page = page_chatgpt if site == "ChatGPT" else page_gemini
                try:
                    if site == "ChatGPT":
                        prev_count = core.count_chatgpt_assistant_messages(page)
                        prev_last = core.normalize_text(core.extract_chatgpt_last_reply(page))
                        core.send_message(page, "ChatGPT", text)
                        shadow_inflight[site] = {"prev_count": prev_count, "prev_last": prev_last, "sent_at": time.time()}
                    else:
                        prev_count = core.count_gemini_responses(page)
                        core.send_message(page, "Gemini", text)
                        shadow_inflight[site] = {"prev_count": prev_count, "sent_at": time.time()}

                    # 同步消息已成功发出，则认为该模型已“看到”截至 upto_mid 的群聊内容
                    seen_upto[site] = max(seen_upto.get(site, 0), int(upto_mid or 0))
                    return True
                except Exception:
                    # 发不出去就塞回队首，避免丢失
                    q.insert(0, (text, upto_mid))
                    shadow_queue[site] = q
                    continue

            return False

        while not state.should_stop():
            if state.is_paused():
                state.set_status("paused")
                time.sleep(0.25)
                continue

            # 优先处理 UI 的用户插话（不会中断正在生成的页面，因为我们只在“回合边界”处理）
            drained = 0
            while drained < 3:
                try:
                    item = state.inbox.get_nowait()
                except queue.Empty:
                    break

                user_text = str(item.get("text") or "")
                to = str(item.get("to") or "next").strip().lower()
                user_mid = state.add_message("You", user_text)
                pending_upto_mid = user_mid
                pending_user_plain = user_text
                shadow_sync_to = None

                if to == "chatgpt":
                    next_site = "ChatGPT"
                    auto_duel = False
                    pending_text = _format_user(user_text)
                    shadow_sync_to = "Gemini"
                elif to == "gemini":
                    next_site = "Gemini"
                    auto_duel = False
                    pending_text = _format_user(user_text)
                    shadow_sync_to = "ChatGPT"
                elif to == "both":
                    # “广播”语义：把用户消息发给两边，让两边各自回复一次，然后停在这里等待下一条用户消息。
                    # 这样更像“群聊”，也能显著减少自动互怼带来的高频请求（更稳、更不容易触发风控）。
                    auto_duel = False
                    broadcast_text = _format_user(user_text)

                    try:
                        state.set_status("broadcast: sending user -> ChatGPT + Gemini")
                        prev_a = core.count_chatgpt_assistant_messages(page_chatgpt)
                        prev_b = core.count_gemini_responses(page_gemini)
                        core.send_message(page_chatgpt, "ChatGPT", _with_rules("ChatGPT", broadcast_text))
                        core.send_message(page_gemini, "Gemini", _with_rules("Gemini", broadcast_text))
                        # 两边都已看到截至 user_mid 的群聊消息
                        seen_upto["ChatGPT"] = max(seen_upto["ChatGPT"], user_mid)
                        seen_upto["Gemini"] = max(seen_upto["Gemini"], user_mid)

                        state.set_status("broadcast: waiting ChatGPT")
                        core.wait_chatgpt_generation_done(page_chatgpt, prev_a)
                        reply_a = core.read_stable_text(
                            lambda: core.extract_chatgpt_last_reply(page_chatgpt), "ChatGPT"
                        )
                        display_a = _strip_forward_summary(reply_a)
                        mid_a = state.add_message("ChatGPT", display_a or "（ChatGPT 回复为空或提取失败）")
                        seen_upto["ChatGPT"] = max(seen_upto["ChatGPT"], mid_a)

                        state.set_status("broadcast: waiting Gemini")
                        core.wait_gemini_generation_done(page_gemini, prev_b)
                        reply_b = core.read_stable_text(lambda: core.extract_gemini_last_reply(page_gemini), "Gemini")
                        display_b = _strip_forward_summary(reply_b)
                        mid_b = state.add_message("Gemini", display_b or "（Gemini 回复为空或提取失败）")
                        seen_upto["Gemini"] = max(seen_upto["Gemini"], mid_b)
                    except Exception as exc:
                        tb = traceback.format_exc(limit=10)
                        state.add_message("System", f"广播异常：{exc}\n{tb}")
                        state.set_status("error (check web pages)")
                    finally:
                        pending_text = None
                        state.set_status("waiting user input")

                    drained += 1
                    continue
                else:
                    # 默认：发给下一位（维持互怼的“单线程”节奏）
                    auto_duel = True
                    pending_text = _format_user(user_text)

                drained += 1

            if pending_text is None:
                # 空闲时推进一次后台同步，尽量让另一边“持续跟上”上下文，但不打断你的主对话节奏
                if pump_shadow_once():
                    continue
                state.set_status("waiting user input")
                time.sleep(0.25)
                continue

            # 执行一轮：把 pending_text 发送给 next_site，等待生成完成，提取回复，然后转发给另一位
            try:
                if next_site == "ChatGPT":
                    state.set_status("ChatGPT generating")
                    prev = core.count_chatgpt_assistant_messages(page_chatgpt)
                    # 如果 ChatGPT 还在处理“后台同步消息”，先等它完成再发新消息，避免输入框/按钮状态异常。
                    infl = shadow_inflight.get("ChatGPT")
                    if infl is not None:
                        core.warn("ChatGPT 仍在后台生成上一条同步消息，先等待完成以保证上下文一致...")
                        prev0 = int(infl.get("prev_count") or 0)
                        core.wait_chatgpt_generation_done(page_chatgpt, prev0)
                        extra = core.read_stable_text(
                            lambda: core.extract_chatgpt_last_reply(page_chatgpt), "ChatGPT", rounds=6
                        )
                        display_extra = _strip_forward_summary(extra)
                        state.add_message("ChatGPT", display_extra or "（ChatGPT 回复为空或提取失败）")
                        shadow_inflight["ChatGPT"] = None
                    core.send_message(page_chatgpt, "ChatGPT", _with_rules("ChatGPT", pending_text))
                    seen_upto["ChatGPT"] = max(seen_upto["ChatGPT"], int(pending_upto_mid or 0))
                    core.wait_chatgpt_generation_done(page_chatgpt, prev)
                    reply = core.read_stable_text(lambda: core.extract_chatgpt_last_reply(page_chatgpt), "ChatGPT")
                    if not reply:
                        reply = "（ChatGPT 回复为空或提取失败）"
                    display = _strip_forward_summary(reply)
                    reply_mid = state.add_message("ChatGPT", display or "（ChatGPT 回复为空或提取失败）")
                    seen_upto["ChatGPT"] = max(seen_upto["ChatGPT"], reply_mid)
                    if auto_duel:
                        pending_text = _format_forward("ChatGPT", _pick_forward_payload(reply))
                        pending_upto_mid = reply_mid
                        next_site = "Gemini"
                        # Gemini 下一轮会收到这条转发
                    else:
                        # 单聊模式：把这一轮同步给另一边，但不展示它的回复（前端会按 target 过滤）
                        if shadow_sync_to:
                            shadow_queue[shadow_sync_to].append(
                                (
                                    _compose_shadow_sync_message(
                                        "ChatGPT", pending_user_plain, _pick_forward_payload(reply) or reply
                                    ),
                                    reply_mid,
                                )
                            )
                        shadow_sync_to = None
                        pending_text = None
                        state.set_status("waiting user input")
                else:
                    state.set_status("Gemini generating")
                    prev = core.count_gemini_responses(page_gemini)
                    infl = shadow_inflight.get("Gemini")
                    if infl is not None:
                        core.warn("Gemini 仍在后台生成上一条同步消息，先等待完成以保证上下文一致...")
                        prev0 = int(infl.get("prev_count") or 0)
                        core.wait_gemini_generation_done(page_gemini, prev0)
                        extra = core.read_stable_text(
                            lambda: core.extract_gemini_last_reply(page_gemini), "Gemini", rounds=6
                        )
                        display_extra = _strip_forward_summary(extra)
                        state.add_message("Gemini", display_extra or "（Gemini 回复为空或提取失败）")
                        shadow_inflight["Gemini"] = None
                    core.send_message(page_gemini, "Gemini", _with_rules("Gemini", pending_text))
                    seen_upto["Gemini"] = max(seen_upto["Gemini"], int(pending_upto_mid or 0))
                    core.wait_gemini_generation_done(page_gemini, prev)
                    reply = core.read_stable_text(lambda: core.extract_gemini_last_reply(page_gemini), "Gemini")
                    if not reply:
                        reply = "（Gemini 回复为空或提取失败）"
                    display = _strip_forward_summary(reply)
                    reply_mid = state.add_message("Gemini", display or "（Gemini 回复为空或提取失败）")
                    seen_upto["Gemini"] = max(seen_upto["Gemini"], reply_mid)
                    if auto_duel:
                        pending_text = _format_forward("Gemini", _pick_forward_payload(reply))
                        pending_upto_mid = reply_mid
                        next_site = "ChatGPT"
                    else:
                        if shadow_sync_to:
                            shadow_queue[shadow_sync_to].append(
                                (
                                    _compose_shadow_sync_message(
                                        "Gemini", pending_user_plain, _pick_forward_payload(reply) or reply
                                    ),
                                    reply_mid,
                                )
                            )
                        shadow_sync_to = None
                        pending_text = None
                        state.set_status("waiting user input")
            except Exception as exc:
                # 任何异常都写到 UI，方便你看到，并给你机会手动处理网页上的风控/弹窗
                tb = traceback.format_exc(limit=10)
                state.add_message("System", f"异常：{exc}\n{tb}")
                state.set_status("error (check web pages)")
                time.sleep(2)

    try:
        httpd.shutdown()
    except Exception:
        pass


def main() -> None:
    try:
        parser = argparse.ArgumentParser(description="ChatGPT + Gemini 互怼的本地 Web UI（含群主插话）")
        parser.add_argument("--host", default=DEFAULT_HOST, help=f"监听地址（默认 {DEFAULT_HOST}）")
        parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"监听端口（默认 {DEFAULT_PORT}）")
        parser.add_argument("--no-open", action="store_true", help="不要自动打开 WebUI 窗口（给 start_webui.bat 用）")
        args = parser.parse_args()
        run_webui_app(host=args.host, port=args.port, no_open=bool(getattr(args, "no_open", False)))
    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")


if __name__ == "__main__":
    main()

