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
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import ai_duel as core


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_MESSAGES = 1500

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
DEFAULT_SELECTED_KEYS = ["chatgpt", "gemini"]


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
    speaker: str
    text: str


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_id = 1
        self._messages: list[UiMessage] = []
        self._status: str = "starting"
        self._paused: bool = False
        self._stop: bool = False
        self._rules: str = DEFAULT_RULES
        self._rules_version: int = 1
        # session_started: Playwright 已成功启动并通过登录检查，可以开始群聊。
        self._session_started: bool = False
        # start_requested: 前端点了「启动」，后台正在启动 Playwright（或准备启动）。
        self._start_requested: bool = False
        self._selected_keys: list[str] = list(DEFAULT_SELECTED_KEYS)
        self._start_event = threading.Event()
        self.inbox: "queue.Queue[dict[str, Any]]" = queue.Queue()

    def add_message(self, speaker: str, text: str) -> int:
        text = core.normalize_text(text)
        if not text:
            return 0
        with self._lock:
            mid = self._next_id
            self._next_id += 1
            self._messages.append(UiMessage(id=mid, ts=_now_iso(), speaker=speaker, text=text))
            # 控制内存：只保留最近 MAX_MESSAGES 条
            if len(self._messages) > MAX_MESSAGES:
                self._messages = self._messages[-MAX_MESSAGES:]
            return mid

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "paused": self._paused,
                "stop": self._stop,
                "session_started": self._session_started,
                "start_requested": self._start_requested,
                "selected_keys": list(self._selected_keys),
            }

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = paused

    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def request_stop(self) -> None:
        with self._lock:
            self._stop = True
        self._start_event.set()

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop

    def get_models(self) -> dict[str, Any]:
        with self._lock:
            slots = []
            selected = set(self._selected_keys)
            for s in MODEL_SLOTS:
                item = dict(s)
                item["selected"] = bool(item.get("key") in selected)
                item["session_started"] = self._session_started
                item["start_requested"] = self._start_requested
                slots.append(item)
            return {
                "session_started": self._session_started,
                "start_requested": self._start_requested,
                "selected_keys": list(self._selected_keys),
                "slots": slots,
            }

    def set_selected_keys(self, keys: list[str]) -> dict[str, Any]:
        # 只能在 session 启动前修改
        cleaned: list[str] = []
        allowed = {str(s.get("key") or "") for s in MODEL_SLOTS if s.get("enabled")}
        for k in keys:
            k = str(k).strip().lower()
            if not k:
                continue
            if k in allowed and k not in cleaned:
                cleaned.append(k)
        if not cleaned:
            cleaned = list(DEFAULT_SELECTED_KEYS)
        with self._lock:
            if not self._session_started and not self._start_requested:
                self._selected_keys = cleaned
        return self.get_models()

    def request_session_start(self) -> dict[str, Any]:
        # 仅“请求启动”，不直接置 session_started=True。
        with self._lock:
            if self._session_started:
                return self.get_models()
            self._start_requested = True
        self._start_event.set()
        return self.get_models()

    def mark_session_started(self) -> None:
        with self._lock:
            self._session_started = True
            self._start_requested = False

    def reset_session(self) -> None:
        with self._lock:
            self._session_started = False
            self._start_requested = False
        self._start_event.clear()

    def wait_for_session_start(self, timeout_s: int = 3600) -> bool:
        return self._start_event.wait(timeout=timeout_s)

    def get_messages_after(self, after_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"id": m.id, "ts": m.ts, "speaker": m.speaker, "text": m.text}
                for m in self._messages
                if m.id > after_id
            ]

    def get_rules(self) -> dict[str, Any]:
        with self._lock:
            return {"rules": self._rules, "version": self._rules_version}

    def set_rules(self, rules: str) -> None:
        rules = (rules or "").strip()
        # 防止 UI 一不小心塞入超长文本把输入框撑爆
        if len(rules) > 6000:
            rules = rules[:6000] + "\n...(已截断)"
        with self._lock:
            self._rules = rules
            self._rules_version += 1

    def last_id(self) -> int:
        with self._lock:
            return self._next_id - 1

    def get_messages_range(self, after_id: int, upto_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"id": m.id, "ts": m.ts, "speaker": m.speaker, "text": m.text}
                for m in self._messages
                if m.id > after_id and m.id <= upto_id
            ]


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AI 群聊控制台</title>
    <style>
      :root {
        --bg: #0b0f14;
        --panel: #111826;
        --panel2: #0f172a;
        --text: #e5e7eb;
        --muted: #94a3b8;
        --accent: #22c55e;
        --warn: #f59e0b;
        --danger: #ef4444;
        --border: rgba(148, 163, 184, 0.20);
      }
      html, body { height: 100%; }
      body {
        margin: 0;
        background: radial-gradient(1200px 700px at 20% 10%, rgba(34, 197, 94, 0.10), transparent 55%),
                    radial-gradient(900px 600px at 90% 30%, rgba(59, 130, 246, 0.10), transparent 60%),
                    var(--bg);
        color: var(--text);
        font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans", "Helvetica Neue", Arial;
      }
      .app { display: grid; grid-template-columns: 96px 1fr; height: 100%; }
      .modelbar {
        border-right: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(17,24,38,0.95), rgba(15,23,42,0.85));
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
        border: 1px solid rgba(148,163,184,0.30);
        background: rgba(17,24,38,0.55);
        cursor: pointer;
        transition: transform 120ms ease, border-color 120ms ease, filter 120ms ease, opacity 120ms ease;
        user-select: none;
      }
      .modelBtn:hover { transform: translateY(-1px); border-color: rgba(148,163,184,0.45); }
      .modelBtn:active { transform: translateY(0px) scale(0.98); }
      .modelBtn.off { filter: grayscale(1); opacity: 0.48; }
      .modelBtn.disabled { filter: grayscale(1); opacity: 0.30; cursor: not-allowed; }
      .modelBtn.on { border-color: rgba(34,197,94,0.40); box-shadow: 0 18px 48px rgba(0,0,0,0.25); }
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
      }
      .modelNum.chatgpt { background: linear-gradient(180deg, rgba(34,197,94,0.95), rgba(16,185,129,0.86)); }
      .modelNum.gemini { background: linear-gradient(180deg, rgba(245,158,11,0.95), rgba(251,191,36,0.86)); }
      .modelNum.other { background: linear-gradient(180deg, rgba(96,165,250,0.95), rgba(37,99,235,0.86)); }
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
        border: 1px solid rgba(34,197,94,0.40);
        background: rgba(34,197,94,0.16);
        font-weight: 800;
      }
      .content { display: grid; grid-template-columns: 360px 1fr; height: 100%; min-width: 0; }
      .sidebar {
        border-right: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(17,24,38,0.95), rgba(15,23,42,0.85));
        padding: 16px;
        display: flex;
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
        background: rgba(15,23,42,0.7);
      }
      .dot { width: 8px; height: 8px; border-radius: 999px; background: var(--warn); }
      .dot.ok { background: var(--accent); }
      .dot.bad { background: var(--danger); }
      .row { display: flex; gap: 10px; }
      button, select, textarea {
        border: 1px solid var(--border);
        background: rgba(17,24,38,0.65);
        color: var(--text);
        border-radius: 10px;
        font-size: 13px;
      }
      button { padding: 10px 12px; cursor: pointer; }
      button:hover { border-color: rgba(148,163,184,0.35); }
      button.primary { background: rgba(34,197,94,0.15); border-color: rgba(34,197,94,0.35); }
      button.danger { background: rgba(239,68,68,0.12); border-color: rgba(239,68,68,0.35); }
      select { padding: 10px 12px; flex: 1; }
      textarea {
        width: 100%; min-height: 120px; resize: vertical;
        padding: 10px 12px; outline: none;
      }
      .hint { font-size: 12px; color: var(--muted); line-height: 1.5; }
      .main { display: flex; flex-direction: column; height: 100%; min-width: 0; }
      .topbar {
        padding: 14px 18px;
        border-bottom: 1px solid var(--border);
        background: rgba(15,23,42,0.55);
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
        background:
          radial-gradient(900px 520px at 12% 12%, rgba(34, 197, 94, 0.10), transparent 62%),
          radial-gradient(900px 520px at 88% 24%, rgba(59, 130, 246, 0.10), transparent 65%),
          repeating-linear-gradient(0deg, rgba(148, 163, 184, 0.05) 0 1px, transparent 1px 22px),
          repeating-linear-gradient(90deg, rgba(148, 163, 184, 0.04) 0 1px, transparent 1px 22px);
      }

      .jump {
        position: absolute;
        right: 18px;
        bottom: 18px;
        z-index: 5;
        border-radius: 999px;
        padding: 8px 10px;
        font-size: 12px;
        background: rgba(15,23,42,0.88);
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
        border: 1px solid var(--border);
        background: rgba(17,24,38,0.62);
        border-radius: 16px;
        padding: 10px 12px;
        box-shadow: 0 18px 48px rgba(0,0,0,0.25);
      }
      .msg-row.you .bubble { background: rgba(96,165,250,0.16); border-color: rgba(96,165,250,0.28); }
      .msg-row.chatgpt .bubble { border-color: rgba(34,197,94,0.28); }
      .msg-row.gemini .bubble { border-color: rgba(245,158,11,0.28); }
      .msg-row.system .bubble { background: rgba(148,163,184,0.10); border-color: rgba(148,163,184,0.24); text-align: center; }

      .bubble-meta { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
      .name { font-weight: 750; font-size: 12px; letter-spacing: 0.2px; }
      .name.you { color: #93c5fd; }
      .name.chatgpt { color: #34d399; }
      .name.gemini { color: #fbbf24; }
      .name.system { color: #c4b5fd; }
      .time { font-size: 11px; color: var(--muted); white-space: nowrap; }
      .bubble-text { margin-top: 6px; white-space: pre-wrap; line-height: 1.55; font-size: 14px; }
      .sysTime { margin-top: 6px; font-size: 11px; color: var(--muted); }

      .msg-row.continued .avatar { visibility: hidden; }
      .msg-row.continued .name { display: none; }

      .composer {
        border-top: 1px solid var(--border);
        background: rgba(15,23,42,0.60);
        padding: 12px 18px 14px;
        display: grid;
        grid-template-columns: 1fr 140px;
        gap: 12px;
        align-items: end;
      }
      .composer textarea {
        width: 100%;
        min-height: 58px;
        max-height: 180px;
        resize: none;
        padding: 12px 12px;
        border-radius: 14px;
        font-size: 14px;
        line-height: 1.5;
      }
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
        background: rgba(17,24,38,0.55);
        color: rgba(229,231,235,0.95);
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        font-size: 11px;
      }
      code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    </style>
  </head>
  <body>
    <div class="app">
      <aside class="modelbar">
        <div id="modelList" class="modelList"></div>
        <button id="startSessionBtn" class="modelStart">启动</button>
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
          <textarea id="input" placeholder="在这里插话（群主发言）..."></textarea>
          <div class="composerActions">
            <button id="sendBtn" class="primary">发送</button>
            <button id="clearBtn">清屏</button>
          </div>
          <div class="composerHint">
            <div>快捷键：<span class="kbd">Enter</span> 发送，<span class="kbd">Shift+Enter</span> 换行</div>
            <div>上滑看历史时，不会强制拉回底部</div>
          </div>
        </div>
      </main>
      </div>
    </div>

    <script>
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
                alert('更新模型选择失败：' + e);
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
          alert('发送失败：' + e);
        }
      }

      pauseBtn.addEventListener('click', async () => {
        try {
          await apiPost('/api/pause', { paused: !paused });
        } catch (e) {
          alert('操作失败：' + e);
        }
      });

      stopBtn.addEventListener('click', async () => {
        if (!confirm('确定要停止脚本吗？')) return;
        try {
          await apiPost('/api/stop', {});
        } catch (e) {
          alert('停止失败：' + e);
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
            alert('启动失败：' + e);
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
          alert('应用失败：' + e);
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
    </script>
  </body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    state: SharedState  # injected

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

        if parsed.path == "/api/rules":
            self._send_json(self.state.get_rules())
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        data = self._read_json_body()

        if parsed.path == "/api/models/select":
            keys = data.get("keys")
            if not isinstance(keys, list):
                keys = []
            payload = self.state.set_selected_keys([str(x) for x in keys])
            self._send_json({"ok": True, **payload})
            return

        if parsed.path == "/api/session/start":
            info = self.state.get_models()
            selected = set(info.get("selected_keys") or [])
            # 目前仅接入 ChatGPT + Gemini；先强制选择 1/2，避免“只选了未接入模型”导致启动后不可用。
            if not {"chatgpt", "gemini"}.issubset(selected):
                self._send_json(
                    {"ok": False, "error": "当前版本请先选择 1=ChatGPT 与 2=Gemini 后再启动"},
                    status=400,
                )
                return
            payload = self.state.request_session_start()
            self._send_json({"ok": True, **payload})
            return

        if parsed.path == "/api/send":
            if not self.state.get_state().get("session_started"):
                self._send_json({"ok": False, "error": "session not started"}, status=409)
                return
            text = core.normalize_text(str(data.get("text") or ""))
            to = str(data.get("to") or "next").strip().lower()
            if not text:
                self._send_json({"ok": False, "error": "empty text"}, status=400)
                return
            if to not in ("next", "chatgpt", "gemini", "both"):
                to = "next"
            self.state.inbox.put({"text": text, "to": to})
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/pause":
            paused = bool(data.get("paused"))
            self.state.set_paused(paused)
            self._send_json({"ok": True, "paused": paused})
            return

        if parsed.path == "/api/stop":
            self.state.request_stop()
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/rules":
            rules = str(data.get("rules") or "")
            self.state.set_rules(rules)
            self._send_json({"ok": True, **self.state.get_rules()})
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # 静默 httpserver 日志，避免刷屏
        return


def start_server(state: SharedState, host: str, port: int) -> ThreadingHTTPServer:
    _Handler.state = state
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, name="webui", daemon=True)
    t.start()
    return httpd


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
        args = parser.parse_args()
        run_webui_duel(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")


if __name__ == "__main__":
    main()
