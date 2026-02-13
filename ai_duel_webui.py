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
import queue
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
MAX_CONTEXT_CHARS = 6000
MAX_CONTEXT_ITEMS = 24  # 仅取最近 N 条记录，避免一次性塞太多上下文


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


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
            return {"status": self._status, "paused": self._paused, "stop": self._stop}

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = paused

    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def request_stop(self) -> None:
        with self._lock:
            self._stop = True

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop

    def get_messages_after(self, after_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"id": m.id, "ts": m.ts, "speaker": m.speaker, "text": m.text}
                for m in self._messages
                if m.id > after_id
            ]

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
      .app { display: grid; grid-template-columns: 360px 1fr; height: 100%; }
      .sidebar {
        border-right: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(17,24,38,0.95), rgba(15,23,42,0.85));
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 12px;
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
      .main { display: flex; flex-direction: column; height: 100%; }
      .topbar {
        padding: 14px 18px;
        border-bottom: 1px solid var(--border);
        background: rgba(15,23,42,0.55);
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .status { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--muted); }
      .chat {
        flex: 1;
        padding: 18px;
        overflow: auto;
      }
      .msg {
        border: 1px solid var(--border);
        background: rgba(17,24,38,0.40);
        border-radius: 14px;
        padding: 12px 14px;
        margin: 0 0 12px 0;
        max-width: 980px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
      }
      .meta { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
      .who { font-weight: 600; font-size: 13px; }
      .ts { font-size: 12px; color: var(--muted); }
      .text { margin-top: 8px; white-space: pre-wrap; line-height: 1.55; font-size: 13px; }
      .who.you { color: #60a5fa; }
      .who.chatgpt { color: #22c55e; }
      .who.gemini { color: #f59e0b; }
      .who.system { color: #a78bfa; }
      .footer {
        border-top: 1px solid var(--border);
        background: rgba(15,23,42,0.55);
        padding: 10px 18px;
        font-size: 12px;
        color: var(--muted);
      }
      code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    </style>
  </head>
  <body>
    <div class="app">
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

        <textarea id="input" placeholder="在这里插话（群主发言）..."></textarea>
        <div class="row">
          <button id="sendBtn" class="primary">发送</button>
          <button id="clearBtn">清屏</button>
        </div>

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
        <div id="chat" class="chat"></div>
        <div class="footer">
          快捷键：<code>Ctrl+Enter</code> 发送
        </div>
      </main>
    </div>

    <script>
      let lastId = 0;
      let paused = false;
      let total = 0;
      const allMessages = [];
      const unseen = { chatgpt: 0, gemini: 0 };

      const chat = document.getElementById('chat');
      const conn = document.getElementById('conn');
      const dot = document.getElementById('dot');
      const statusText = document.getElementById('statusText');
      const unseenPill = document.getElementById('unseenPill');
      const pauseBtn = document.getElementById('pauseBtn');
      const stopBtn = document.getElementById('stopBtn');
      const sendBtn = document.getElementById('sendBtn');
      const clearBtn = document.getElementById('clearBtn');
      const input = document.getElementById('input');
      const target = document.getElementById('target');
      const countPill = document.getElementById('countPill');

      function speakerClass(s) {
        const x = (s || '').toLowerCase();
        if (x.includes('chatgpt')) return 'chatgpt';
        if (x.includes('gemini')) return 'gemini';
        if (x === 'you' || x === 'user' || x.includes('用户')) return 'you';
        if (x.includes('system') || x.includes('系统')) return 'system';
        return '';
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
        const div = document.createElement('div');
        div.className = 'msg';

        const meta = document.createElement('div');
        meta.className = 'meta';

        const who = document.createElement('div');
        who.className = 'who ' + speakerClass(m.speaker);
        who.textContent = m.speaker;

        const ts = document.createElement('div');
        ts.className = 'ts';
        ts.textContent = m.ts;

        meta.appendChild(who);
        meta.appendChild(ts);

        const text = document.createElement('div');
        text.className = 'text';
        text.textContent = m.text;

        div.appendChild(meta);
        div.appendChild(text);
        chat.appendChild(div);
      }

      function updateUnseenPill() {
        unseenPill.textContent = `未读：ChatGPT ${unseen.chatgpt} | Gemini ${unseen.gemini}`;
      }

      function rerender() {
        chat.innerHTML = '';
        for (const m of allMessages) {
          if (matchesFilter(m)) appendMessage(m);
        }
        scrollToBottom();
      }

      function scrollToBottom() {
        chat.scrollTop = chat.scrollHeight;
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
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return await r.json();
      }

      async function poll() {
        try {
          const state = await apiGet('/api/state');
          statusText.textContent = state.status || '';
          paused = !!state.paused;
          pauseBtn.textContent = paused ? '继续' : '暂停';

          const msgs = await apiGet('/api/messages?after=' + lastId);
          if (Array.isArray(msgs) && msgs.length) {
            dot.className = 'dot ok';
            conn.textContent = 'ok';
            for (const m of msgs) {
              allMessages.push(m);
              if (matchesFilter(m)) {
                appendMessage(m);
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
            scrollToBottom();
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

      sendBtn.addEventListener('click', send);
      target.addEventListener('change', () => {
        const f = currentFilter();
        if (f.chatgpt) unseen.chatgpt = 0;
        if (f.gemini) unseen.gemini = 0;
        updateUnseenPill();
        rerender();
      });
      clearBtn.addEventListener('click', () => {
        chat.innerHTML = '';
        lastId = 0;
        total = 0;
        allMessages.length = 0;
        unseen.chatgpt = 0;
        unseen.gemini = 0;
        countPill.textContent = '0';
        updateUnseenPill();
      });

      input.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter' && ev.ctrlKey) {
          ev.preventDefault();
          send();
        }
      });

      updateUnseenPill();
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

        if parsed.path == "/api/send":
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


def _other(site: str) -> str:
    return "Gemini" if site == "ChatGPT" else "ChatGPT"


def _build_context_bundle(
    state: SharedState,
    target_site: str,
    after_id: int,
    upto_id: int,
) -> tuple[str, int]:
    """
    当你长时间只跟 A 聊，突然切到 B 时，为了让 B “知道之前聊了什么”，需要把 A 的对话摘要/记录补给 B。

    注意：网页端没有“系统提示词”注入能力，所以这里采用“把上下文作为用户消息的一部分”方式。
    """
    other = _other(target_site)
    msgs = state.get_messages_range(after_id, upto_id)
    msgs = [m for m in msgs if m.get("speaker") in ("You", other)]
    if not msgs:
        return "", after_id

    # 只取最后 N 条，避免上下文过长
    if len(msgs) > MAX_CONTEXT_ITEMS:
        msgs = msgs[-MAX_CONTEXT_ITEMS:]

    lines: list[str] = []
    for m in msgs:
        speaker = str(m.get("speaker") or "")
        text = core.normalize_text(str(m.get("text") or ""))
        if not text:
            continue
        # 控制每条长度，避免单条超长（尤其是模型长回复）
        if len(text) > 1200:
            text = text[:1200] + "\n...(已截断)"
        prefix = "用户" if speaker == "You" else other
        lines.append(f"- {prefix}: {text}")

    bundle = "\n".join(lines).strip()
    if not bundle:
        return "", after_id

    # 全局长度控制
    if len(bundle) > MAX_CONTEXT_CHARS:
        bundle = bundle[-MAX_CONTEXT_CHARS:]

    new_cursor = max(int(m.get("id") or 0) for m in msgs)
    header = f"下面是我与 {other} 的最近对话记录（供你补齐上下文）：\n"
    return header + bundle, new_cursor


def _compose_user_message_with_context(context: str, user_text: str) -> str:
    base = _format_user(user_text)
    if not context:
        return base
    return f"{context}\n\n我现在要说：\n{base}".strip()

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
            "- 回复请尽量精炼：先 1 句话结论，再列 3-6 个要点。",
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

    try:
        webbrowser.open_new_tab(url)
    except Exception:
        pass

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

        seed_message = "你好，请开始你的论述。"
        seed_mid = state.add_message("You", seed_message)

        next_site = "ChatGPT"
        pending_text: Optional[str] = _format_user(seed_message)
        pending_upto_mid = seed_mid  # pending_text 对应的群聊消息 id（用于已同步游标）
        pending_user_plain = seed_message  # 仅用于“单聊时后台同步给另一模型”的提示词拼接
        shadow_sync_to: Optional[str] = None  # 单聊模式下：主模型回复后，把这一轮同步给哪一边
        auto_duel = True  # 仅当目标为“发给下一位”时，才持续把回复转发给另一边
        # 记录每个模型“已经看过”的群聊消息 id，用于切换目标时自动补上下文
        seen_upto: dict[str, int] = {"ChatGPT": 0, "Gemini": 0}
        # 种子消息会先发给 ChatGPT，因此先把它标记为 ChatGPT 已见
        seen_upto["ChatGPT"] = seed_mid

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

                        reply = core.read_stable_text(lambda: core.extract_chatgpt_last_reply(page), "ChatGPT", rounds=6)
                        state.add_message("ChatGPT", reply or "（ChatGPT 回复为空或提取失败）")
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
                        state.add_message("Gemini", reply or "（Gemini 回复为空或提取失败）")
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
                        core.send_message(page_chatgpt, "ChatGPT", broadcast_text)
                        core.send_message(page_gemini, "Gemini", broadcast_text)
                        # 两边都已看到截至 user_mid 的群聊消息
                        seen_upto["ChatGPT"] = max(seen_upto["ChatGPT"], user_mid)
                        seen_upto["Gemini"] = max(seen_upto["Gemini"], user_mid)

                        state.set_status("broadcast: waiting ChatGPT")
                        core.wait_chatgpt_generation_done(page_chatgpt, prev_a)
                        reply_a = core.read_stable_text(lambda: core.extract_chatgpt_last_reply(page_chatgpt), "ChatGPT")
                        mid_a = state.add_message("ChatGPT", reply_a or "（ChatGPT 回复为空或提取失败）")
                        seen_upto["ChatGPT"] = max(seen_upto["ChatGPT"], mid_a)

                        state.set_status("broadcast: waiting Gemini")
                        core.wait_gemini_generation_done(page_gemini, prev_b)
                        reply_b = core.read_stable_text(lambda: core.extract_gemini_last_reply(page_gemini), "Gemini")
                        mid_b = state.add_message("Gemini", reply_b or "（Gemini 回复为空或提取失败）")
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
                        state.add_message("ChatGPT", extra or "（ChatGPT 回复为空或提取失败）")
                        shadow_inflight["ChatGPT"] = None
                    core.send_message(page_chatgpt, "ChatGPT", pending_text)
                    seen_upto["ChatGPT"] = max(seen_upto["ChatGPT"], int(pending_upto_mid or 0))
                    core.wait_chatgpt_generation_done(page_chatgpt, prev)
                    reply = core.read_stable_text(lambda: core.extract_chatgpt_last_reply(page_chatgpt), "ChatGPT")
                    if not reply:
                        reply = "（ChatGPT 回复为空或提取失败）"
                    reply_mid = state.add_message("ChatGPT", reply)
                    seen_upto["ChatGPT"] = max(seen_upto["ChatGPT"], reply_mid)
                    if auto_duel:
                        pending_text = _format_forward("ChatGPT", reply)
                        pending_upto_mid = reply_mid
                        next_site = "Gemini"
                        # Gemini 下一轮会收到这条转发
                    else:
                        # 单聊模式：把这一轮同步给另一边，但不展示它的回复（前端会按 target 过滤）
                        if shadow_sync_to:
                            shadow_queue[shadow_sync_to].append(
                                (_compose_shadow_sync_message("ChatGPT", pending_user_plain, reply), reply_mid)
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
                        extra = core.read_stable_text(lambda: core.extract_gemini_last_reply(page_gemini), "Gemini", rounds=6)
                        state.add_message("Gemini", extra or "（Gemini 回复为空或提取失败）")
                        shadow_inflight["Gemini"] = None
                    core.send_message(page_gemini, "Gemini", pending_text)
                    seen_upto["Gemini"] = max(seen_upto["Gemini"], int(pending_upto_mid or 0))
                    core.wait_gemini_generation_done(page_gemini, prev)
                    reply = core.read_stable_text(lambda: core.extract_gemini_last_reply(page_gemini), "Gemini")
                    if not reply:
                        reply = "（Gemini 回复为空或提取失败）"
                    reply_mid = state.add_message("Gemini", reply)
                    seen_upto["Gemini"] = max(seen_upto["Gemini"], reply_mid)
                    if auto_duel:
                        pending_text = _format_forward("Gemini", reply)
                        pending_upto_mid = reply_mid
                        next_site = "ChatGPT"
                    else:
                        if shadow_sync_to:
                            shadow_queue[shadow_sync_to].append(
                                (_compose_shadow_sync_message("Gemini", pending_user_plain, reply), reply_mid)
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
