"""Model adapter layer for driving official web chat UIs via Playwright.

Goals
- Each model uses its own persistent profile directory: `user_data/<model_key>` (login persistence).
- Avoid brittle hash CSS classes; prefer role/placeholder/aria-* selectors.
- Keep the adapter interface small so adding models is mostly: URL + input/snapshot heuristics.

Note
- This project intentionally drives the *web UI* (not API calls). Please follow each site\'s TOS.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from playwright.sync_api import BrowserContext, Locator, Page, Playwright

import ai_duel as core


@dataclass(frozen=True)
class ModelMeta:
    slot: int
    key: str
    name: str
    url: str
    color: str
    integrated: bool
    avatar_url: str
    login_help: str

    def user_data_dir(self) -> Path:
        return Path("user_data") / self.key


def _data_svg(svg: str) -> str:
    # Keep it ASCII-only; percent-encode only what is necessary.
    import urllib.parse

    raw = svg.strip().replace("\n", "")
    return "data:image/svg+xml," + urllib.parse.quote(raw, safe=",:;()@/+-_=.*!~' ")


def default_avatar_svg(color: str, label: str) -> str:
    """Simple glossy circle avatar as a data-uri SVG."""
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
      <defs>
        <radialGradient id="g" cx="30%" cy="25%" r="75%">
          <stop offset="0" stop-color="#ffffff" stop-opacity="0.9"/>
          <stop offset="0.25" stop-color="{color}" stop-opacity="0.95"/>
          <stop offset="1" stop-color="#101418" stop-opacity="1"/>
        </radialGradient>
      </defs>
      <circle cx="48" cy="48" r="44" fill="url(#g)" stroke="rgba(255,255,255,0.18)" stroke-width="2"/>
      <circle cx="34" cy="30" r="10" fill="rgba(255,255,255,0.18)"/>
      <text x="50%" y="56%" text-anchor="middle" font-family="ui-sans-serif,Segoe UI,Arial" font-size="34"
            fill="rgba(255,255,255,0.88)" font-weight="700">{label}</text>
    </svg>
    """
    return _data_svg(svg)


class ModelAdapter:
    """Minimal adapter contract.

    Notes
    - All methods are expected to run on the same thread as Playwright.
    - For unknown/3rd-party UIs, we use "conversation snapshot diff" as a fallback extraction strategy.
    """

    def __init__(self, meta: ModelMeta) -> None:
        self.meta = meta
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    # --- lifecycle ---------------------------------------------------------

    def ensure_page(self, p: Playwright) -> Page:
        """Ensure the official web chat page is opened (persistent profile)."""
        if self.page is not None:
            return self.page

        prof = self.meta.user_data_dir()
        prof.mkdir(parents=True, exist_ok=True)

        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(prof.resolve()),
            headless=False,
            viewport={"width": 1920, "height": 1080},
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx.set_default_timeout(15000)

        self.context = ctx
        self.page = ctx.pages[0] if ctx.pages else ctx.new_page()

        self.goto_home()
        return self.page

    def goto_home(self) -> None:
        if self.page is None:
            return
        try:
            self.page.goto(self.meta.url, wait_until="domcontentloaded")
        except Exception:
            # Ignore transient navigation errors.
            pass

    def bring_to_front(self) -> None:
        if self.page is None:
            return
        try:
            self.page.bring_to_front()
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self.context is not None:
                self.context.close()
        except Exception:
            pass
        self.context = None
        self.page = None

    # --- auth --------------------------------------------------------------

    def find_input(self) -> Optional[Locator]:
        raise NotImplementedError

    def is_authenticated_now(self) -> bool:
        """Heuristic: if the chat input is present, consider it authenticated."""
        try:
            box = self.find_input()
            return box is not None
        except Exception:
            return False

    # --- messaging ---------------------------------------------------------

    def snapshot_conversation(self) -> str:
        """A coarse snapshot of the current conversation text for diff-based extraction."""
        if self.page is None:
            return ""
        for sel in ("main", "body"):
            try:
                loc = self.page.locator(sel)
                node = core.pick_visible(loc, prefer_last=False)
                if node is None:
                    continue
                return core.normalize_text(node.inner_text())
            except Exception:
                continue
        return ""

    def send_user_text(self, text: str) -> None:
        """Default implementation: fill a textbox then click send / press Enter."""
        if self.page is None:
            raise RuntimeError("page not ready")

        content = core.normalize_text(text)
        if not content:
            raise ValueError("empty text")

        box = self.find_input()
        if box is None:
            raise RuntimeError(f"{self.meta.name} 未找到输入框（可能未登录/弹窗遮挡/页面未加载）")

        core.clear_and_fill_input(self.page, box, content, self.meta.name)

        send_btn = self._find_send_button_generic()
        core.jitter(f"{self.meta.name} 发送前")
        if core.safe_is_enabled(send_btn):
            try:
                send_btn.click(timeout=8000)
            except Exception:
                self.page.keyboard.press("Enter")
        else:
            self.page.keyboard.press("Enter")
        core.jitter(f"{self.meta.name} 发送后")

    def _find_send_button_generic(self) -> Optional[Locator]:
        if self.page is None:
            return None

        candidates = [
            self.page.get_by_role("button", name=re.compile(r"send|发送|提交|发布|↩|➤|➡", re.I)),
            self.page.locator("button[aria-label*='Send' i]"),
            self.page.locator("button[aria-label*='发送']"),
            self.page.locator("form button[type='submit']"),
        ]
        for loc in candidates:
            node = core.pick_visible(loc, prefer_last=True)
            if node is not None:
                return node
        return None

    def wait_reply_and_extract(self, before_snapshot: str, timeout_s: int = 300) -> str:
        """Default: wait until snapshot changes and stabilizes, then diff."""
        if self.page is None:
            return ""

        begin = time.time()
        # Wait for any change.
        while time.time() - begin < timeout_s:
            cur = self.snapshot_conversation()
            if cur and cur != before_snapshot:
                break
            time.sleep(0.6)

        stable = core.read_stable_text(lambda: self.snapshot_conversation(), self.meta.name, rounds=10)
        reply = self._diff_reply(before_snapshot, stable)
        return reply

    @staticmethod
    def _diff_reply(before: str, after: str) -> str:
        before = core.normalize_text(before)
        after = core.normalize_text(after)
        if not after:
            return ""
        if before and after.startswith(before) and len(after) > len(before):
            tail = after[len(before) :].strip()
            return tail[-4000:].strip() if tail else ""

        # Fallback: take the last chunk.
        lines = [x.strip() for x in after.splitlines() if x.strip()]
        if not lines:
            return after[-1200:].strip()
        tail = "\n".join(lines[-40:]).strip()
        return tail[-4000:].strip()


class ChatGPTAdapter(ModelAdapter):
    def find_input(self) -> Optional[Locator]:
        if self.page is None:
            return None
        return core.find_chatgpt_input(self.page)

    def send_user_text(self, text: str) -> None:
        if self.page is None:
            raise RuntimeError("page not ready")
        core.send_message(self.page, "ChatGPT", text)

    def wait_reply_and_extract(self, before_snapshot: str, timeout_s: int = 600) -> str:
        if self.page is None:
            return ""
        prev_count = core.count_chatgpt_assistant_messages(self.page)
        core.wait_chatgpt_generation_done(self.page, prev_count, timeout_s=timeout_s)
        return core.read_stable_text(lambda: core.extract_chatgpt_last_reply(self.page), "ChatGPT", rounds=12)


class GeminiAdapter(ModelAdapter):
    def find_input(self) -> Optional[Locator]:
        if self.page is None:
            return None
        return core.find_gemini_input(self.page)

    def send_user_text(self, text: str) -> None:
        if self.page is None:
            raise RuntimeError("page not ready")
        core.send_message(self.page, "Gemini", text)

    def wait_reply_and_extract(self, before_snapshot: str, timeout_s: int = 600) -> str:
        if self.page is None:
            return ""
        prev_count = core.count_gemini_responses(self.page)
        core.wait_gemini_generation_done(self.page, prev_count, timeout_s=timeout_s)
        return core.read_stable_text(lambda: core.extract_gemini_last_reply(self.page), "Gemini", rounds=12)


class DeepSeekAdapter(ModelAdapter):
    """DeepSeek official web chat: https://chat.deepseek.com/"""

    def find_input(self) -> Optional[Locator]:
        if self.page is None:
            return None

        candidates = [
            self.page.locator("textarea"),
            self.page.get_by_placeholder(re.compile(r"send|message|输入|发送|提问|Ask", re.I)),
            self.page.get_by_role("textbox"),
            self.page.locator("rich-textarea [contenteditable='true']"),
            self.page.locator("[contenteditable='true']"),
        ]
        for loc in candidates:
            node = core.pick_visible(loc, prefer_last=True)
            if node is not None:
                return node
        return None

    def snapshot_conversation(self) -> str:
        if self.page is None:
            return ""

        # DeepSeek often has no <main>; capture from message nodes directly.
        for sel in ("div.ds-message", "div.ds-markdown", "div[class*='message']"):
            try:
                loc = self.page.locator(sel)
                cnt = loc.count()
                if cnt <= 0:
                    continue
                parts = []
                start = max(0, cnt - 24)
                for i in range(start, cnt):
                    txt = core.normalize_text(loc.nth(i).inner_text())
                    if txt:
                        parts.append(txt)
                merged = core.normalize_text("\n\n".join(parts))
                if merged:
                    return merged
            except Exception:
                continue

        # Fallback: body may be reported as not visible; read text directly.
        try:
            return core.normalize_text(self.page.locator("body").first.inner_text())
        except Exception:
            pass
        return ""

    def _count_assistant_messages(self) -> int:
        if self.page is None:
            return 0

        selectors = [
            "div.ds-message:not(.d29f3d7d)",
            "div.ds-message:not([class*='d29f3d7d'])",
            "div[class*='ds-message']:not(.d29f3d7d)",
        ]
        counts = []
        for sel in selectors:
            try:
                counts.append(self.page.locator(sel).count())
            except Exception:
                continue
        return max(counts) if counts else 0

    @staticmethod
    def _looks_like_thought_text(text: str) -> bool:
        t = core.normalize_text(text)
        if not t:
            return False
        # Final answer in this project is expected to include explicit forwarding summary marker.
        if "【转发摘要】" in t:
            return False

        if t.startswith("已思考（") or t.startswith("已思考("):
            return True

        head = t[:420]
        hints = [
            "嗯，用户",
            "用户继续",
            "我们收到用户消息",
            "用户消息",
            "作为deepseek",
            "作为deeps",
            "回应思路",
            "注意规则",
            "最后要加【转发摘要】",
            "因此，回应可以这样",
            "我需要",
            "我要",
            "先抓住",
            "反驳方向",
            "最后追问",
            "转发摘要要",
            "思路",
            "接下来",
            "这个总结",
        ]
        hit = sum(1 for h in hints if h in head)
        if hit >= 2:
            return True

        # Many thought blocks start with planning language and are long but have no summary marker.
        if len(t) >= 120 and ("回应" in head or "总结" in head or "需要" in head) and "用户" in head:
            return True
        return False

    @staticmethod
    def _sanitize_reply_text(text: str) -> str:
        t = core.normalize_text(text)
        if not t:
            return ""

        # Remove citation-only artifact lines such as "-", "8", "- 9".
        cleaned: list[str] = []
        for ln in t.splitlines():
            s = ln.strip()
            if not s:
                cleaned.append("")
                continue
            if re.fullmatch(r"[-–—]\s*", s):
                continue
            if re.fullmatch(r"[-–—]?\s*\d+\s*", s):
                continue
            cleaned.append(s)

        # Compress repeated blanks while preserving paragraph breaks.
        out: list[str] = []
        blank = False
        for ln in cleaned:
            if not ln:
                if not blank:
                    out.append("")
                blank = True
            else:
                out.append(ln)
                blank = False
        return core.normalize_text("\n".join(out))

    def _last_assistant_message(self) -> Optional[Locator]:
        if self.page is None:
            return None

        selectors = [
            "div.ds-message:not(.d29f3d7d)",
            "div.ds-message:not([class*='d29f3d7d'])",
            "div[class*='ds-message']:not(.d29f3d7d)",
        ]
        for sel in selectors:
            try:
                loc = self.page.locator(sel)
                cnt = loc.count()
                if cnt > 0:
                    return loc.nth(cnt - 1)
            except Exception:
                continue
        return None

    def _extract_last_assistant_reply(self) -> str:
        last_msg = self._last_assistant_message()
        if last_msg is None:
            return ""

        try:
            raw = last_msg.evaluate(
                """
                (el) => {
                  const clone = el.cloneNode(true);

                  // Remove obvious "thinking" and footer areas.
                  const dropSelectors = [
                    '.ds-think-content',
                    '[class*="think-content"]',
                    '._74c0879',
                    '.dbe8cf4a',
                    '.f93f59e4',
                    '.ffdab56b',
                    '.c2b72bb8',
                  ];
                  for (const sel of dropSelectors) {
                    for (const n of Array.from(clone.querySelectorAll(sel))) n.remove();
                  }

                  const toText = (n) => ((n && n.innerText) ? n.innerText.trim() : '');
                  const isThoughtLike = (s) => {
                    const t = (s || '').trim();
                    return !t || t.startsWith('已思考（') || t.startsWith('已思考(') || t.includes('回应思路') || t.includes('我们收到用户消息');
                  };

                  // Prefer top-level markdown blocks (typically final answer block is here).
                  const top = Array.from(clone.querySelectorAll(':scope > div.ds-markdown, :scope > div[class*="markdown"]'))
                    .map(n => toText(n))
                    .filter(t => t && !isThoughtLike(t));
                  if (top.length) {
                    const withSummary = top.filter(t => t.includes('【转发摘要】'));
                    if (withSummary.length) return withSummary[withSummary.length - 1];
                    top.sort((a, b) => b.length - a.length);
                    return top[0];
                  }

                  // Fallback: any non-think markdown block.
                  const anyMd = Array.from(clone.querySelectorAll('div.ds-markdown, div[class*="markdown"]'))
                    .filter(n => !n.closest('.ds-think-content,[class*="think-content"]'))
                    .map(n => toText(n))
                    .filter(t => t && !isThoughtLike(t));
                  if (anyMd.length) {
                    const withSummary = anyMd.filter(t => t.includes('【转发摘要】'));
                    if (withSummary.length) return withSummary[withSummary.length - 1];
                    anyMd.sort((a, b) => b.length - a.length);
                    return anyMd[0];
                  }

                  // Last fallback: cleaned message text.
                  return toText(clone);
                }
                """
            )
            txt = self._sanitize_reply_text(raw)
            if txt and not self._looks_like_thought_text(txt):
                return txt
        except Exception:
            pass
        return ""

    def wait_reply_and_extract(self, before_snapshot: str, timeout_s: int = 600) -> str:
        if self.page is None:
            return ""

        before_count = self._count_assistant_messages()
        before_last = core.normalize_text(self._extract_last_assistant_reply())

        begin = time.time()
        while time.time() - begin < timeout_s:
            cur_count = self._count_assistant_messages()
            cur_last = core.normalize_text(self._extract_last_assistant_reply())

            # Wait for a new *final* answer text, not intermediate thought content.
            if (cur_count > before_count or (cur_last and cur_last != before_last)) and cur_last:
                stable = core.read_stable_text(lambda: self._extract_last_assistant_reply(), "DeepSeek", rounds=6)
                stable = core.normalize_text(stable)
                if stable and stable != before_last and not self._looks_like_thought_text(stable):
                    return stable
            time.sleep(0.8)

        stable = core.read_stable_text(lambda: self._extract_last_assistant_reply(), "DeepSeek", rounds=10)
        stable = core.normalize_text(stable)
        if stable and stable != before_last and not self._looks_like_thought_text(stable):
            return stable

        # Give one extra short window for the final answer block to appear after thought block.
        grace_begin = time.time()
        while time.time() - grace_begin < 12:
            cur = core.normalize_text(self._extract_last_assistant_reply())
            if cur and cur != before_last and not self._looks_like_thought_text(cur):
                return cur
            time.sleep(0.8)

        # Avoid diff fallback here: it often captures thought/citation fragments like "-" or "1".
        return ""


class GenericWebChatAdapter(ModelAdapter):
    """Template adapter for other web chats (Doubao/Qwen/Kimi/etc)."""

    def find_input(self) -> Optional[Locator]:
        if self.page is None:
            return None

        candidates = [
            self.page.get_by_placeholder(re.compile(r"send|message|输入|发送|提问|聊天|Ask", re.I)),
            self.page.locator("rich-textarea [contenteditable='true']"),
            self.page.get_by_role("textbox"),
            self.page.locator("textarea"),
            self.page.locator("[contenteditable='true']"),
        ]
        for loc in candidates:
            node = core.pick_visible(loc, prefer_last=True)
            if node is not None:
                return node
        return None
