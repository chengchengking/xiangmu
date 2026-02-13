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

        <div class="row">
          <button id="pauseBtn">暂停</button>
          <button id="stopBtn" class="danger">停止</button>
        </div>

        <div class="row">
          <select id="target">
            <option value="next" selected>发给下一位（推荐）</option>
            <option value="chatgpt">只发给 ChatGPT</option>
            <option value="gemini">只发给 Gemini</option>
            <option value="both">广播给两边（每边回复一次）</option>
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

      const chat = document.getElementById('chat');
      const conn = document.getElementById('conn');
      const dot = document.getElementById('dot');
      const statusText = document.getElementById('statusText');
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
              appendMessage(m);
              lastId = Math.max(lastId, m.id || lastId);
              total++;
            }
            countPill.textContent = String(total);
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
      clearBtn.addEventListener('click', () => {
        chat.innerHTML = '';
        lastId = 0;
        total = 0;
        countPill.textContent = '0';
      });

      input.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter' && ev.ctrlKey) {
          ev.preventDefault();
          send();
        }
      });

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
        state.add_message("You", seed_message)

        next_site = "ChatGPT"
        pending_text: Optional[str] = _format_user(seed_message)
        auto_duel = True  # 仅当目标为“发给下一位”时，才持续把回复转发给另一边

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
                state.add_message("You", user_text)

                if to == "chatgpt":
                    next_site = "ChatGPT"
                    auto_duel = False
                elif to == "gemini":
                    next_site = "Gemini"
                    auto_duel = False
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

                        state.set_status("broadcast: waiting ChatGPT")
                        core.wait_chatgpt_generation_done(page_chatgpt, prev_a)
                        reply_a = core.read_stable_text(lambda: core.extract_chatgpt_last_reply(page_chatgpt), "ChatGPT")
                        state.add_message("ChatGPT", reply_a or "（ChatGPT 回复为空或提取失败）")

                        state.set_status("broadcast: waiting Gemini")
                        core.wait_gemini_generation_done(page_gemini, prev_b)
                        reply_b = core.read_stable_text(lambda: core.extract_gemini_last_reply(page_gemini), "Gemini")
                        state.add_message("Gemini", reply_b or "（Gemini 回复为空或提取失败）")
                    except Exception as exc:
                        tb = traceback.format_exc(limit=10)
                        state.add_message("System", f"广播异常：{exc}\n{tb}")
                        state.set_status("error (check web pages)")
                    finally:
                        pending_text = None
                        state.set_status("waiting user input")

                    drained += 1
                    continue

                # 默认：发给下一位（维持互怼的“单线程”节奏）
                auto_duel = True
                pending_text = _format_user(user_text)
                drained += 1

            if pending_text is None:
                state.set_status("waiting user input")
                time.sleep(0.25)
                continue

            # 执行一轮：把 pending_text 发送给 next_site，等待生成完成，提取回复，然后转发给另一位
            try:
                if next_site == "ChatGPT":
                    state.set_status("ChatGPT generating")
                    prev = core.count_chatgpt_assistant_messages(page_chatgpt)
                    core.send_message(page_chatgpt, "ChatGPT", pending_text)
                    core.wait_chatgpt_generation_done(page_chatgpt, prev)
                    reply = core.read_stable_text(lambda: core.extract_chatgpt_last_reply(page_chatgpt), "ChatGPT")
                    if not reply:
                        reply = "（ChatGPT 回复为空或提取失败）"
                    state.add_message("ChatGPT", reply)
                    if auto_duel:
                        pending_text = _format_forward("ChatGPT", reply)
                        next_site = "Gemini"
                    else:
                        pending_text = None
                        state.set_status("waiting user input")
                else:
                    state.set_status("Gemini generating")
                    prev = core.count_gemini_responses(page_gemini)
                    core.send_message(page_gemini, "Gemini", pending_text)
                    core.wait_gemini_generation_done(page_gemini, prev)
                    reply = core.read_stable_text(lambda: core.extract_gemini_last_reply(page_gemini), "Gemini")
                    if not reply:
                        reply = "（Gemini 回复为空或提取失败）"
                    state.add_message("Gemini", reply)
                    if auto_duel:
                        pending_text = _format_forward("Gemini", reply)
                        next_site = "ChatGPT"
                    else:
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
