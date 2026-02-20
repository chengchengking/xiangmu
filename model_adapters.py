"""Model adapter layer for driving official web chat UIs via Playwright.

Goals
- Each model uses its own persistent profile directory: `user_data/<model_key>` (login persistence).
- Avoid brittle hash CSS classes; prefer role/placeholder/aria-* selectors.
- Keep the adapter interface small so adding models is mostly: URL + input/snapshot heuristics.

Note
- This project intentionally drives the *web UI* (not API calls). Please follow each site\'s TOS.
"""

from __future__ import annotations

import json
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

        try:
            core.clear_and_fill_input(self.page, box, content, self.meta.name)
        except Exception:
            # One retry for transient stale/overlay issues on dynamic web UIs.
            time.sleep(0.4)
            box2 = self.find_input()
            if box2 is None:
                raise
            core.clear_and_fill_input(self.page, box2, content, self.meta.name)

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
        if before and after == before:
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
    _GEMINI_STRICT_SELECTORS: tuple[str, ...] = (
        "main model-response",
        "main [data-response-id]",
        "main div[data-response-id]",
        "model-response",
    )
    _GEMINI_FALLBACK_SELECTORS: tuple[str, ...] = (
        "main [data-message-author-role='assistant']",
        "main [data-message-role='assistant']",
        "main [data-role='assistant']",
        "main [class*='assistant' i] [class*='markdown' i]",
        "main [class*='model-response' i]",
    )
    _GEMINI_PROMPT_HINTS: tuple[str, ...] = (
        "你在多人群聊中发言",
        "你在多人群聊中继续讨论",
        "群主最新话题：",
        "上一位发言（",
        "直接说你在群里要发的话，不要复述规则",
        "只输出一句给群里的正文，不要复述规则",
        "先回应群主话题",
        "先回应用户",
        "最终发言要求：",
        "如暂不发言，仅回复 [PASS]",
        "只输出 [PASS]",
        "可回应对象：",
        "可点名对象：",
        "只依据以下群聊消息回复",
        "不要复述提示词",
        "不要复述本提示",
        "[[PUBLIC_REPLY]]",
        "[[/PUBLIC_REPLY]]",
        "<<<PUBLIC_REPLY>>>",
        "<<<END_PUBLIC_REPLY>>>",
    )
    _GEMINI_UI_NOISE_PAT = re.compile(
        r"(?:about gemini|gemini apps?|open in new window|enterprise|subscription|"
        r"关于\s*gemini|gemini\s*应用|与\s*gemini\s*对话|在新窗口中打开|企业应用场景|订阅|"
        r"须遵守.?google\s*条款|须遵守.?google\s*隐私权政策|认识\s*gemini|你的私人\s*ai\s*助理|"
        r"gemini\s*是.?款\s*ai\s*工具|写作|计划|研究|学习|工具|历史记录|新对话|发现)",
        re.I,
    )
    _GEMINI_THINK_TOGGLE_PAT = re.compile(
        r"^\s*(?:显示思路|显示思考|显示推理|思路展开|show\s*(?:thinking|reasoning)|thinking)\s*$",
        re.I,
    )
    _GEMINI_SPEAKER_LABEL_PAT = re.compile(
        r"^\s*gemini\s*(?:说|says|said)?\s*[:：]?\s*$",
        re.I,
    )

    def find_input(self) -> Optional[Locator]:
        if self.page is None:
            return None
        return core.find_gemini_input(self.page)

    def send_user_text(self, text: str) -> None:
        if self.page is None:
            raise RuntimeError("page not ready")
        core.send_message(self.page, "Gemini", text)

    @staticmethod
    def _line_key(text: str) -> str:
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", core.normalize_text(text).lower())

    def _clean_candidate_text(self, text: str) -> str:
        t = core.normalize_text(text)
        if not t:
            return ""
        m = re.search(r"(?is)\[\[\s*PUBLIC_REPLY\s*\]\]\s*(.*?)\s*\[\[\s*/\s*PUBLIC_REPLY\s*\]\]", t, re.I)
        if m:
            return core.normalize_text(m.group(1))
        # Drop Gemini UI labels that are sometimes copied into message text.
        kept: list[str] = []
        for ln in t.splitlines():
            s = ln.strip()
            if not s:
                kept.append("")
                continue
            if self._GEMINI_THINK_TOGGLE_PAT.match(s):
                continue
            if self._GEMINI_SPEAKER_LABEL_PAT.match(s):
                continue
            kept.append(ln)

        t = core.normalize_text("\n".join(kept))
        t = re.sub(r"^\s*Gemini\s*(?:说|says|said)?\s*[:：]?\s*", "", t, flags=re.I)
        return core.normalize_text(t)

    def _salvage_noisy_candidate(self, text: str) -> str:
        t = self._clean_candidate_text(text)
        if not t:
            return ""
        out: list[str] = []
        seen: set[str] = set()
        for ln in t.splitlines():
            s = core.normalize_text(ln)
            if not s:
                continue
            if any(h in s for h in self._GEMINI_PROMPT_HINTS):
                continue
            if self._GEMINI_UI_NOISE_PAT.search(s) and len(self._line_key(s)) <= 18:
                continue
            k = self._line_key(s)
            if k and k in seen:
                continue
            if k:
                seen.add(k)
            out.append(s)
        return core.normalize_text("\n".join(out))

    def _looks_like_ui_noise(self, text: str) -> bool:
        t = self._clean_candidate_text(text)
        if not t:
            return True
        if any(h in t for h in self._GEMINI_PROMPT_HINTS):
            return True
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        if not lines:
            return True
        key_len = len(self._line_key(t))
        if len(lines) == 1 and self._GEMINI_UI_NOISE_PAT.search(lines[0]) and key_len <= 220:
            return True
        hit = sum(1 for ln in lines if self._GEMINI_UI_NOISE_PAT.search(ln))
        if hit >= max(2, len(lines) // 2):
            return True
        if "|" in t and hit >= 1 and key_len <= 64:
            return True
        if hit >= 1 and key_len <= 42:
            return True
        return False

    def _iter_reply_selectors(self) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for sel in self._GEMINI_STRICT_SELECTORS:
            if sel not in seen:
                seen.add(sel)
                out.append(sel)
        for sel in self._GEMINI_FALLBACK_SELECTORS:
            if sel not in seen:
                seen.add(sel)
                out.append(sel)
        return out

    def _extract_last_reply_candidate(self) -> str:
        if self.page is None:
            return ""
        best = ""
        best_score = float("-inf")
        seen_keys: set[str] = set()
        for sel in self._iter_reply_selectors():
            try:
                loc = self.page.locator(sel)
                cnt = loc.count()
            except Exception:
                continue
            if cnt <= 0:
                continue
            start = max(0, cnt - 10)
            for i in range(cnt - 1, start - 1, -1):
                node = loc.nth(i)
                try:
                    txt = self._clean_candidate_text(node.inner_text())
                except Exception:
                    continue
                if not txt:
                    continue
                k = self._line_key(txt)
                if not k or k in seen_keys:
                    continue
                seen_keys.add(k)
                if self._looks_like_ui_noise(txt):
                    salv = self._salvage_noisy_candidate(txt)
                    if not salv or self._looks_like_ui_noise(salv):
                        continue
                    txt = salv
                    k = self._line_key(txt)
                    if not k:
                        continue
                score = float(len(k))
                if cnt > 1:
                    score += (i / max(1, cnt - 1)) * 80.0
                if len(k) >= 18:
                    score += 36.0
                elif len(k) < 10:
                    score -= 48.0
                if score > best_score or (score == best_score and len(txt) > len(best)):
                    best = txt
                    best_score = score
        return core.normalize_text(best)

    def wait_reply_and_extract(self, before_snapshot: str, timeout_s: int = 600) -> str:
        if self.page is None:
            return ""
        prev_count = core.count_gemini_responses(self.page)
        before_last = core.normalize_text(self._extract_last_reply_candidate() or core.extract_gemini_last_reply(self.page))

        timed_out = False
        try:
            core.wait_gemini_generation_done(self.page, prev_count, timeout_s=timeout_s)
        except TimeoutError:
            timed_out = True
            core.warn("Gemini 等待生成超时，进入提取兜底流程。")

        stable = core.normalize_text(
            core.read_stable_text(lambda: self._extract_last_reply_candidate() or core.extract_gemini_last_reply(self.page), "Gemini", rounds=10)
        )
        if stable and stable != before_last and not self._looks_like_ui_noise(stable):
            return stable

        # If the "last reply" cursor did not move but conversation snapshot already changed,
        # extract by snapshot diff to avoid losing the first turn.
        diff_now = core.normalize_text(self._diff_reply(before_snapshot, self.snapshot_conversation()))
        diff_now = self._salvage_noisy_candidate(diff_now) or diff_now
        if diff_now and not self._looks_like_ui_noise(diff_now):
            return diff_now

        # Some Gemini layouts lag the final render after stop/send state settles.
        if timed_out:
            grace_begin = time.time()
            while time.time() - grace_begin < 15:
                cur = core.normalize_text(self._extract_last_reply_candidate() or core.extract_gemini_last_reply(self.page))
                cur = self._salvage_noisy_candidate(cur) or cur
                if cur and cur != before_last and not self._looks_like_ui_noise(cur):
                    return cur
                time.sleep(0.8)

        # Last fallback: diff full snapshot to avoid losing an already-rendered answer.
        after = core.normalize_text(self.snapshot_conversation())
        diff = core.normalize_text(self._diff_reply(before_snapshot, after))
        diff = self._salvage_noisy_candidate(diff) or diff
        if diff and diff != before_last and not self._looks_like_ui_noise(diff):
            return diff
        if stable and not self._looks_like_ui_noise(stable):
            return stable
        return ""


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

    _COMMON_REPLY_SELECTORS: tuple[str, ...] = (
        "main [class*='assistant'] [class*='markdown']",
        "main [data-role='assistant']",
        "main [class*='message'] [class*='markdown']",
        "main [class*='message-content']",
        "main [class*='answer']",
        "main [class*='response']",
        "main article",
        "main div.markdown",
        "main div[class*='markdown']",
    )

    _KEY_REPLY_SELECTORS: dict[str, tuple[str, ...]] = {
        "doubao": (
            "main [data-role='assistant']",
            "main [data-message-role='assistant']",
            "main [data-message-author-role='assistant']",
            "main [data-testid*='assistant' i]",
            "main [class*='assistant-message']",
            "main [class*='chat-item'] [class*='markdown']",
            "main [class*='answer-content']",
        ),
        "qwen": (
            "main [data-role='assistant']",
            "main [data-message-role='assistant']",
            "main [data-message-author-role='assistant']",
            "main [data-role='assistant']",
            "main [class*='assistant'] [class*='bubble']",
            "main [class*='assistant-message']",
            "main [class*='chat-item'] [class*='content']",
            "main [class*='message-item'] [class*='markdown']",
        ),
        "kimi": (
            "main [class*='assistant'] [class*='markdown']",
            "main [class*='message'] [class*='markdown']",
        ),
    }

    _NOISE_LINE_PAT = re.compile(
        r"^(新对话|内容由 ?AI ?生成|今天|昨天|7 ?天内|30 ?天内|历史记录|会话列表|设置|发现|快速|免费|更多|Beta|超能模式|PPT ?生成|图像生成|帮我写作|正在思考…*|思考中…*|已完成思考|跳过|人工智能生成的内容可能不准确。?)$",
        re.I,
    )
    _NOISE_TEXT_PAT = re.compile(
        r"(新对话|内容由 ?AI ?生成|历史记录|会话列表|重命名|删除会话|设置|发现|帮助中心|反馈|升级会员|找到\d+篇资料|PPT ?生成|超能模式|图像生成|帮我写作|Beta|免费体验|正在思考|思考中|已完成思考|人工智能生成的内容可能不准确)",
        re.I,
    )

    _PROMPT_ECHO_HINTS: tuple[str, ...] = (
        "<<<PUBLIC_REPLY>>>",
        "<<<END_PUBLIC_REPLY>>>",
        "<<<PRIVATE_REPLY>>>",
        "<<<END_PRIVATE_REPLY>>>",
        "[[PUBLIC_REPLY]]",
        "[[/PUBLIC_REPLY]]",
        "[[PRIVATE_REPLY]]",
        "[[/PRIVATE_REPLY]]",
        "请把公开发言放在",
        "公开发言放在",
        "如需隐藏想法，可放在",
        "如需隐藏想法，仅使用",
        "你在多人群聊中发言",
        "你在多人群聊中继续讨论",
        "群主最新话题：",
        "优先级：必须先回应“群主最新话题”",
        "最终发言要求：",
        "只输出给群里的正文",
        "不要写“先给区间/然后补充/基于当前信息提出”",
        "如暂不发言，仅回复 [PASS]",
        "群主刚刚插话：",
        "下面是群聊新消息，请直接给出对外发言",
        "只依据以下群聊消息回复",
        "不要复述本提示",
        "不要复述提示词",
        "不要复述题目",
        "直接发你在群里的这条回复",
        "1-2句",
        "实质内容",
        "附一个追问",
        "请优先回应这条插话",
        "请继续群聊",
        "请先回应群主插话",
        "再基于上一位",
        "像真人群聊一样自然接话",
        "可点名对象：",
        "可回应对象：",
        "群主：",
        "请直接在群里自然回复",
        "请基于这条消息继续群聊",
        "继续群聊，给出你想补充",
        "如果你这轮不想发言",
        "可选：使用",
        "如果你想加入当前讨论",
        "如果你暂时不加入",
        "只输出 [PASS]",
        "群聊旁听同步",
        "群聊消息同步",
        "这是群聊第",
        "上一位发言（",
    )

    def __init__(self, meta: ModelMeta) -> None:
        super().__init__(meta)
        self._last_sent_text: str = ""

    def find_input(self) -> Optional[Locator]:
        if self.page is None:
            return None

        candidates = [
            self.page.locator("rich-textarea [contenteditable='true']"),
            self.page.locator("textarea"),
            self.page.locator("[contenteditable='true']"),
            self.page.get_by_role("textbox"),
            self.page.get_by_placeholder(re.compile(r"send|message|输入|发送|提问|聊天|Ask", re.I)),
        ]
        for loc in candidates:
            node = core.pick_visible(loc, prefer_last=True)
            if node is not None:
                return node
        return None

    def send_user_text(self, text: str) -> None:
        self._last_sent_text = core.normalize_text(text)
        super().send_user_text(text)

    def _reply_selectors(self) -> list[str]:
        key = (self.meta.key or "").strip().lower()
        picked: list[str] = []
        for sel in self._KEY_REPLY_SELECTORS.get(key, ()):
            if sel not in picked:
                picked.append(sel)
        for sel in self._COMMON_REPLY_SELECTORS:
            if sel not in picked:
                picked.append(sel)
        return picked

    @classmethod
    def _looks_like_ui_noise(cls, text: str) -> bool:
        t = core.normalize_text(text)
        if not t:
            return True

        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        if not lines:
            return True
        if len(lines) == 1 and cls._NOISE_LINE_PAT.search(lines[0]):
            return True

        if len(t) <= 30 and cls._NOISE_TEXT_PAT.search(t):
            return True

        # Sidebar/list style snippets: many short lines with navigation words.
        short_lines = sum(1 for ln in lines if len(ln) <= 10)
        if len(lines) >= 3 and short_lines >= len(lines) - 1 and cls._NOISE_TEXT_PAT.search(t):
            return True
        return False

    def _strip_prompt_echo(self, text: str) -> str:
        t = core.normalize_text(text)
        if not t:
            return ""

        sent = core.normalize_text(self._last_sent_text)
        sent_lines = {ln.strip() for ln in sent.splitlines() if ln.strip()}
        keep: list[str] = []
        for ln in t.splitlines():
            s = ln.strip()
            if not s:
                continue
            if s in sent_lines:
                continue
            if any(h in s for h in self._PROMPT_ECHO_HINTS):
                continue
            if self._NOISE_LINE_PAT.search(s):
                continue
            keep.append(s)

        out = core.normalize_text("\n".join(keep))
        if not out:
            return ""

        if sent:
            sent_compact = re.sub(r"\s+", "", sent)
            out_compact = re.sub(r"\s+", "", out)
            if out_compact and (out_compact in sent_compact or sent_compact in out_compact):
                return ""
        return out

    @staticmethod
    def _line_dedupe_key(line: str) -> str:
        s = core.normalize_text(line).lower()
        if not s:
            return ""
        # Keep only common alnum + CJK chars so punctuation/spacing variants collapse.
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", s)

    def _drop_repeated_blocks(self, lines: list[str]) -> list[str]:
        if len(lines) < 4:
            return lines

        keys = [self._line_dedupe_key(x) for x in lines]
        out: list[str] = []
        i = 0
        n = len(lines)
        while i < n:
            # Drop repeated blocks that already appeared earlier (common in re-rendered snapshots).
            max_block = min(8, i, n - i)
            skip = 0
            for block in range(max_block, 1, -1):
                cur = keys[i : i + block]
                if not all(cur):
                    continue
                if sum(len(x) for x in cur) < 16:
                    continue
                found = False
                for j in range(0, i - block + 1):
                    if keys[j : j + block] == cur:
                        found = True
                        break
                if found:
                    skip = block
                    break
            if skip:
                i += skip
                continue
            out.append(lines[i])
            i += 1
        return out

    def _clean_candidate_text(self, text: str) -> str:
        cleaned = self._strip_prompt_echo(text)
        if not cleaned:
            return ""

        # Remove repeated lines from streaming/virtualized re-render.
        out: list[str] = []
        prev_key = ""
        seen_exact: set[str] = set()
        seen_long: list[str] = []
        for ln in cleaned.splitlines():
            s = ln.strip()
            if not s:
                continue
            key = self._line_dedupe_key(s)
            if not key:
                continue
            if key == prev_key:
                continue
            # Drop global exact duplicates for meaningful lines.
            if len(key) >= 6 and key in seen_exact:
                continue
            # Drop short "split fragments" when a longer previous line already contains them.
            covered = False
            if len(key) >= 8:
                for old in seen_long[-24:]:
                    if len(old) >= len(key) + 8 and key in old:
                        covered = True
                        break
            if covered:
                continue
            out.append(s)
            prev_key = key
            if len(key) >= 6:
                seen_exact.add(key)
            if len(key) >= 14:
                seen_long.append(key)

        # Collapse repeated blocks (A/B/.../A/B/...) even when not adjacent.
        out = self._drop_repeated_blocks(out)

        # Collapse immediately repeated tail blocks, e.g. A/B/A/B at the end.
        changed = True
        while changed and len(out) >= 4:
            changed = False
            max_block = min(8, len(out) // 2)
            for block in range(max_block, 0, -1):
                if out[-2 * block : -block] == out[-block:]:
                    out = out[:-block]
                    changed = True
                    break
        cleaned = core.normalize_text("\n".join(out))
        if not cleaned:
            return ""
        if self._looks_like_ui_noise(cleaned):
            return ""
        return cleaned

    def _extract_last_reply_candidate(self) -> str:
        if self.page is None:
            return ""

        fallback_short = ""
        for sel in self._reply_selectors():
            try:
                loc = self.page.locator(sel)
                cnt = loc.count()
            except Exception:
                continue
            if cnt <= 0:
                continue

            for i in range(cnt - 1, -1, -1):
                node = loc.nth(i)
                try:
                    txt = self._clean_candidate_text(node.inner_text())
                except Exception:
                    continue
                if not txt:
                    continue
                if len(txt) >= 24:
                    return txt
                if not fallback_short:
                    fallback_short = txt
        return fallback_short

    def snapshot_conversation(self) -> str:
        if self.page is None:
            return ""

        parts: list[str] = []
        for sel in self._reply_selectors():
            try:
                loc = self.page.locator(sel)
                cnt = loc.count()
            except Exception:
                continue
            if cnt <= 0:
                continue

            start = max(0, cnt - 28)
            for i in range(start, cnt):
                try:
                    txt = self._clean_candidate_text(loc.nth(i).inner_text())
                except Exception:
                    continue
                if not txt:
                    continue
                parts.append(txt)
            if parts:
                break

        if parts:
            return core.normalize_text("\n\n".join(parts[-24:]))
        return super().snapshot_conversation()

    def wait_reply_and_extract(self, before_snapshot: str, timeout_s: int = 600) -> str:
        if self.page is None:
            return ""

        before_last = core.normalize_text(self._extract_last_reply_candidate())
        begin = time.time()
        while time.time() - begin < timeout_s:
            cur_last = core.normalize_text(self._extract_last_reply_candidate())
            if cur_last and cur_last != before_last:
                stable = core.normalize_text(
                    core.read_stable_text(lambda: self._extract_last_reply_candidate(), self.meta.name, rounds=8)
                )
                if stable and stable != before_last and not self._looks_like_ui_noise(stable):
                    return stable

            cur_snapshot = core.normalize_text(self.snapshot_conversation())
            if cur_snapshot and cur_snapshot != before_snapshot:
                diff = self._clean_candidate_text(self._diff_reply(before_snapshot, cur_snapshot))
                if diff:
                    return diff
            time.sleep(0.8)

        stable_snapshot = core.normalize_text(core.read_stable_text(lambda: self.snapshot_conversation(), self.meta.name, rounds=10))
        reply = self._clean_candidate_text(self._diff_reply(before_snapshot, stable_snapshot))
        if reply:
            return reply

        last = core.normalize_text(self._extract_last_reply_candidate())
        if last and last != before_last:
            return last
        return ""


class QwenAdapter(GenericWebChatAdapter):
    """Qwen-specific adapter with stricter reply extraction preference."""

    _QWEN_STRICT_SELECTORS: tuple[str, ...] = (
        "main #chat-message-container div.qwen-chat-message.qwen-chat-message-assistant",
        "main #chat-message-container div[id^='qwen-chat-message-assistant-']",
        "main #chat-message-container div.chat-response-message",
        "main #chat-message-container div[id^='chat-response-message-']",
        "main #chat-messages-scroll-container div.qwen-chat-message.qwen-chat-message-assistant",
        "main #chat-messages-scroll-container div.chat-response-message",
        "main [data-role='assistant']",
        "main [data-message-role='assistant']",
        "main [data-message-author-role='assistant']",
        "main [data-testid*='assistant' i]",
    )
    _QWEN_FALLBACK_SELECTORS: tuple[str, ...] = (
        "main [class*='qwen-chat-message-assistant' i]",
        "main [id^='qwen-chat-message-assistant-']",
        "main [class*='chat-response-message' i]",
        "main [id^='chat-response-message-']",
        "main [class*='assistant-message']",
        "main [class*='assistant'] [class*='bubble']",
        "main [class*='chat-item'] [class*='content']",
        "main [class*='message-item'] [class*='markdown']",
    )
    _QWEN_DOM_REPLY_ROOT_SELECTORS: tuple[str, ...] = (
        "main #chat-message-container div.qwen-chat-message.qwen-chat-message-assistant",
        "main #chat-message-container div[id^='qwen-chat-message-assistant-']",
        "main #chat-message-container div.chat-response-message",
        "main #chat-message-container div[id^='chat-response-message-']",
        "main #chat-messages-scroll-container div.qwen-chat-message.qwen-chat-message-assistant",
        "main #chat-messages-scroll-container div.chat-response-message",
    )
    _QWEN_DOM_REPLY_TEXT_SELECTORS: tuple[str, ...] = (
        ".chat-response-message-content",
        ".chat-response-message [class*='markdown' i]",
        ".qwen-chat-message-assistant [class*='markdown' i]",
        ".qwen-chat-message-assistant [class*='content' i]",
        ".chat-response-message p",
        ".chat-response-message li",
        ".qwen-chat-message-assistant p",
        ".qwen-chat-message-assistant li",
    )
    _QWEN_DOM_DROP_SELECTORS: tuple[str, ...] = (
        ".qwen-chat-status-card",
        ".qwen-chat-status-card-title",
        ".qwen-chat-status-card-title-animate",
        ".chat-response-message-right",
        "[class*='status-card' i]",
        "[class*='status-title' i]",
        ".qwen-chat-status-card-answer-now",
        "button",
        "[role='button']",
    )
    _QWEN_STOP_PAT = re.compile(r"停止生成|停止回答|停止输出|stop generating|stop response", re.I)
    _QWEN_THINKING_PAT = re.compile(r"正在思考|思考中|生成中|responding|writing", re.I)
    _QWEN_STATUS_PAT = re.compile(
        r"^\s*(?:已?完成|已经完成|完成思考|已完成思考|思考|思考中|正在思考|继续思考|生成中|回答中|正在构思|构思中)\s*$",
        re.I,
    )
    _QWEN_PASS_PAT = re.compile(r"^\s*(?:\[?\s*pass\s*\]?|跳过|旁听|继续旁听)\s*$", re.I)
    _QWEN_PROMPT_LINE_HINTS: tuple[str, ...] = (
        "<<<PUBLIC_REPLY>>>",
        "<<<END_PUBLIC_REPLY>>>",
        "<<<PRIVATE_REPLY>>>",
        "<<<END_PRIVATE_REPLY>>>",
        "[[PUBLIC_REPLY]]",
        "[[/PUBLIC_REPLY]]",
        "[[PRIVATE_REPLY]]",
        "[[/PRIVATE_REPLY]]",
        "请把公开发言放在",
        "公开发言放在",
        "如需隐藏想法，可放在",
        "如需隐藏想法，仅使用",
        "这是群聊第",
        "上一位发言（",
        "下面是群聊新消息，请直接给出对外发言",
        "只依据以下群聊消息回复",
        "不要复述本提示",
        "不要复述提示词",
        "不要复述题目",
        "直接发你在群里的这条回复",
        "1-2句",
        "实质内容",
        "附一个追问",
        "请你基于当前对话",
        "请先回应群主插话",
        "再基于上一位",
        "像真人群聊一样自然接话",
        "可点名对象：",
        "可回应对象：",
        "群主：",
        "请直接在群里自然回复",
        "请基于这条消息继续群聊",
        "继续群聊，给出你想补充",
        "如果你这轮不想发言",
        "可选：使用",
        "如果你认为应让某个模型加入",
        "群聊同步",
        "群聊旁听同步",
        "群聊消息同步",
        "请你直接回复这一轮消息",
        "先回应用户，再补充你对上一位的看法",
        "如暂不发言，仅回复 [PASS]",
        "请务必给出一条实际观点",
        "只输出 [PASS]",
        "如需隐藏想法，仅使用",
        "不会公开",
        "最终发言要求：",
        "只输出给群里的正文",
        "你在多人群聊中发言",
        "SELFTEST-",
    )
    _QWEN_HEADING_FRAGMENT_PAT = re.compile(r"(回应|思路|结论|总结|要点|分析|建议|步骤)", re.I)
    _QWEN_SHORT_COMPLETE_PAT = re.compile(
        r"^\s*[\u4e00-\u9fff]{4,10}(?:\s*\n\s*@[\w\u4e00-\u9fff-]{1,24})?\s*$",
        re.I,
    )
    _QWEN_LOW_VALUE_PROCESS_PAT = re.compile(
        r"^\s*(?:继续推进(?:接龙)?游戏(?:进程)?|继续推进对话进程|继续成语接龙(?:的游戏)?|继续推进接龙游戏|"
        r"专注推进对话进程|聚焦于生成符合要求的回应|忽略初始回合的起始声明，专注回应核心请求|"
        r"继续推进游戏|聚焦于回应|专注回应核心请求|继续推进游戏进程|专注推进(?:游戏|对话)进程|"
        r"聚焦于(?:生成|回应|推进)(?:符合要求的)?(?:回应|内容|流程)?|"
        r"专注解析对话(?:情境|语境)?|承接对话(?:脉络)?(?:，|,)?(?:自然)?(?:回应|延续语意流|延续语义流)?|"
        r"保持专注与清晰的表达|保持流畅对话的节奏|"
        r"(?:优化|组织|梳理)(?:回应|表达)(?:以契合语境)?|"
        r"(?:正在)?理解对话(?:情境|语境)与语义内涵|"
        r"正在权衡上下文限制与用户期待之间的矛盾|权衡上下文限制与用户期待之间的矛盾|"
        r"用心体会用户传递的情感与需求|营造和谐共处的对话氛围|"
        r"表达观点|肯定这一创意的趣味性与教育意义|"
        r"评估(?:人民币计价)?黄金价格走势|评估.*(?:走势|区间|情境|语境|话题)|"
        r"(?:预测|推演)(?:人民币计价)?黄金价格(?:走势|趋势|区间)|(?:预测|推演).*(?:走势|趋势|区间|情境|语境|话题)|"
        r"开始成语接龙游戏|启动成语接龙游戏|接龙游戏正酣.*|"
        r"我需要[:：].*|让我(?:构思|想一下|数一下|再确认)|检查字数|数一下|好，就这个)\s*[。.!！~～]*\s*$",
        re.I,
    )
    _QWEN_TOPIC_PROCESS_PAT = re.compile(
        r"(根据(?:群主)?最新话题.*(?:给出|回应|讨论)|给出一个简洁.*核心假设|给出.*人民币.*黄金.*价格区间.*假设)",
        re.I,
    )
    _QWEN_PROCESS_LABEL_PAT = re.compile(
        r"^\s*(?:开始|继续|承接|专注|聚焦|保持|优化|营造|理解|体会|推动|组织|梳理|确认|明确|延续|对齐)\s*"
        r".{0,24}(?:对话|语境|语义|流程|节奏|进程|脉络|情境|回应|请求|需求|观点|氛围|主题|衔接|表达)\s*"
        r"[。.!！~～]?\s*$",
        re.I,
    )

    def __init__(self, meta: ModelMeta) -> None:
        super().__init__(meta)
        self._last_effective_reply: str = ""
        self._recent_sent_line_keys: list[set[str]] = []

    def _is_process_title_line(self, text: str) -> bool:
        t = core.normalize_text(text)
        if not t:
            return False
        if self._QWEN_LOW_VALUE_PROCESS_PAT.match(t) or self._QWEN_PROCESS_LABEL_PAT.match(t):
            return True
        if (
            "\n" not in t
            and len(self._line_dedupe_key(t)) <= 26
            and not re.search(r"[。！？?!,，；;:：]", t)
            and re.search(r"(?:评估|预测|推演|判断|分析|权衡|理解|整理|总结)", t)
            and re.search(r"(?:趋势|走势|区间|情境|语境|语义|流程|进程|脉络|回应|回复)", t)
        ):
            return True
        if "\n" in t:
            return False
        klen = len(self._line_dedupe_key(t))
        if klen < 6 or klen > 34:
            return False
        if not re.search(r"(开始|启动|继续|承接|推进|分析|理解|权衡|优化|聚焦|专注|整理|总结|接龙|回应|流程|进程)", t):
            return False
        if re.search(r"(我|你|他|她|我们|建议|同意|反对|认为|可以|应该|因为|所以)", t):
            return False
        return True

    @staticmethod
    def _prompt_like_penalty(text: str) -> int:
        t = core.normalize_text(text)
        if not t:
            return 0
        bad = 0
        hints = (
            "<<<PUBLIC_REPLY>>>",
            "<<<END_PUBLIC_REPLY>>>",
            "<<<PRIVATE_REPLY>>>",
            "<<<END_PRIVATE_REPLY>>>",
            "[[PUBLIC_REPLY]]",
            "[[/PUBLIC_REPLY]]",
            "[[PRIVATE_REPLY]]",
            "[[/PRIVATE_REPLY]]",
            "请把公开发言放在",
            "公开发言放在",
            "如需隐藏想法，可放在",
            "如需隐藏想法，仅使用",
            "这是群聊第",
            "上一位发言（",
            "下面是群聊新消息，请直接给出对外发言",
            "只依据以下群聊消息回复",
            "不要复述本提示",
            "不要复述提示词",
            "不要复述题目",
            "直接发你在群里的这条回复",
            "1-2句",
            "实质内容",
            "附一个追问",
            "请你基于当前对话",
            "可点名对象：",
            "可回应对象：",
            "群主：",
            "请直接在群里自然回复",
            "请基于这条消息继续群聊",
            "继续群聊，给出你想补充",
            "如果你这轮不想发言",
            "可选：使用",
            "如果你认为应让某个模型加入",
            "群聊同步",
            "群聊旁听同步",
            "群聊消息同步",
            "请你直接回复这一轮消息",
            "先回应用户，再补充你对上一位的看法",
            "如暂不发言，仅回复 [PASS]",
            "请务必给出一条实际观点",
            "只输出 [PASS]",
            "SELFTEST-",
        )
        for h in hints:
            if h in t:
                bad += 1
        return bad

    def _is_qwen_generating(self) -> bool:
        if self.page is None:
            return False

        stop_candidates = (
            self.page.get_by_role("button", name=re.compile(r"^(停止生成|停止回答|停止输出|Stop generating|Stop response)$", re.I)),
            self.page.locator(
                "button:has-text('停止生成'),"
                "button:has-text('停止回答'),"
                "button:has-text('停止输出'),"
                "button:has-text('Stop generating'),"
                "button:has-text('Stop response')"
            ),
        )
        for loc in stop_candidates:
            node = core.pick_visible(loc, prefer_last=True)
            if node is not None:
                return True

        # Avoid text-based "thinking" markers here: they are noisy and can stay visible after completion.
        return False

    def _clean_candidate_text(self, text: str) -> str:
        cleaned = super()._clean_candidate_text(text)
        if not cleaned:
            return ""

        lines: list[str] = []
        for ln in cleaned.splitlines():
            s = ln.strip()
            if not s:
                continue
            if self._QWEN_STATUS_PAT.match(s):
                continue
            if re.search(r"(谁先来|我们这就开始|先来(?:出)?第一个|我先来抛砖引玉)", s, re.I) and len(self._line_dedupe_key(s)) <= 56:
                continue
            if re.search(r"(?:好(?:的|嘞)?|收到|明白|ok|OK)[，,。!！~～\s]{0,3}(?:群主|主持人|老大|老板)", s, re.I) and len(self._line_dedupe_key(s)) <= 56:
                continue
            if self._QWEN_LOW_VALUE_PROCESS_PAT.match(s):
                continue
            if self._QWEN_TOPIC_PROCESS_PAT.search(s):
                continue
            if self._QWEN_PROCESS_LABEL_PAT.match(s):
                continue
            if self._is_process_title_line(s):
                continue
            if re.search(r"</?\s*think\s*>", s, re.I):
                continue
            if re.search(r"(?:我需要\s*[:：]|让我(?:构思|想一下|数一下|再确认)|检查字数|字数符合要求|好，就这个)", s):
                continue
            if s in {"自动", "自动模式"} or (s.startswith("自动") and len(s) <= 6):
                continue
            if any(h in s for h in self._QWEN_PROMPT_LINE_HINTS):
                continue
            if self._looks_like_recent_prompt_echo(s):
                continue
            if s.startswith("【群聊") or s.startswith("【用户】"):
                continue
            if re.match(r"^【[^】]{1,12}】$", s):
                continue
            if re.match(r"^@[\w\u4e00-\u9fff-]{1,20}\s*[:：]\s*$", s):
                continue
            lines.append(s)

        if len(lines) > 18 and self._prompt_like_penalty("\n".join(lines)) >= 1:
            lines = lines[-18:]

        out = core.normalize_text("\n".join(lines))
        if not out:
            return ""
        if (
            "\n" not in out
            and len(self._line_dedupe_key(out)) <= 12
            and not re.search(r"[。？！?!，,；;:：]", out)
            and self._QWEN_HEADING_FRAGMENT_PAT.search(out)
        ):
            # Often a transient section heading, not the final answer body.
            return ""
        if self._QWEN_PASS_PAT.match(out):
            return ""
        if self._QWEN_STATUS_PAT.match(out):
            return ""
        if self._QWEN_LOW_VALUE_PROCESS_PAT.match(out):
            return ""
        if self._QWEN_TOPIC_PROCESS_PAT.search(out):
            return ""
        if self._QWEN_PROCESS_LABEL_PAT.match(out):
            return ""
        if self._is_process_title_line(out):
            return ""
        if self._prompt_echo_overlap_ratio(out) >= 0.5:
            return ""
        if self._prompt_like_penalty(out) >= 2:
            return ""
        return out

    def send_user_text(self, text: str) -> None:
        self._remember_recent_sent_lines(text)
        super().send_user_text(text)

    def _remember_recent_sent_lines(self, text: str) -> None:
        t = core.normalize_text(text)
        if not t:
            return
        keys: set[str] = set()
        for ln in t.splitlines():
            s = core.normalize_text(ln)
            if not s:
                continue
            k = self._line_dedupe_key(s)
            if k:
                keys.add(k)
            # Keep the payload tail as an echo-key too:
            # "用户：xxx" / "上一位发言（Qwen）：xxx" should remember "xxx".
            for sep in ("：", ":"):
                if sep not in s:
                    continue
                head, tail = s.split(sep, 1)
                hk = self._line_dedupe_key(head)
                tk = self._line_dedupe_key(tail)
                if not tk:
                    continue
                if len(hk) <= 16 or re.search(
                    r"(用户|群主|上一位发言|user|assistant|model|chatgpt|gemini|deepseek|qwen|doubao|kimi|豆包|千问|通义|Gemini|ChatGPT|DeepSeek|Qwen|Kimi)",
                    head,
                    re.I,
                ):
                    keys.add(tk)
        if not keys:
            return
        self._recent_sent_line_keys.append(keys)
        if len(self._recent_sent_line_keys) > 20:
            self._recent_sent_line_keys = self._recent_sent_line_keys[-20:]

    def _looks_like_recent_prompt_echo(self, line: str) -> bool:
        k = self._line_dedupe_key(line)
        if not k:
            return False
        for hist in self._recent_sent_line_keys:
            if k in hist:
                return True
        return False

    def _prompt_echo_overlap_ratio(self, text: str) -> float:
        t = core.normalize_text(text)
        if not t:
            return 0.0
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        if not lines:
            return 0.0

        recent_keys: set[str] = set()
        for hist in self._recent_sent_line_keys[-4:]:
            recent_keys.update(hist)
        recent_list = [k for k in recent_keys if len(k) >= 8]
        total = 0
        hit = 0
        for ln in lines:
            k = self._line_dedupe_key(ln)
            if not k:
                continue
            total += 1
            if any(h in ln for h in self._QWEN_PROMPT_LINE_HINTS):
                hit += 1
                continue
            if "上下文边界" in ln or "只依据以下群聊消息回复" in ln or "不要复述本提示" in ln:
                hit += 1
                continue
            if k in recent_keys:
                hit += 1
                continue
            if len(k) >= 12:
                for old in recent_list[-48:]:
                    if k == old:
                        hit += 1
                        break
                    if len(old) >= 12 and (k in old or old in k):
                        hit += 1
                        break
        if total <= 0:
            return 0.0
        return hit / float(total)

    def _iter_reply_selectors(self) -> list[tuple[str, bool]]:
        out: list[tuple[str, bool]] = []
        seen: set[str] = set()
        for sel in self._QWEN_STRICT_SELECTORS:
            if sel not in seen:
                seen.add(sel)
                out.append((sel, True))
        for sel in self._QWEN_FALLBACK_SELECTORS:
            if sel not in seen:
                seen.add(sel)
                out.append((sel, False))
        for sel in super()._reply_selectors():
            if sel not in seen:
                seen.add(sel)
                out.append((sel, False))
        return out

    def _extract_last_reply_by_dom_path(self) -> str:
        if self.page is None:
            return ""

        root_selectors_js = ", ".join(json.dumps(s) for s in self._QWEN_DOM_REPLY_ROOT_SELECTORS)
        text_selectors_js = ", ".join(json.dumps(s) for s in self._QWEN_DOM_REPLY_TEXT_SELECTORS)
        drop_selectors_js = ", ".join(json.dumps(s) for s in self._QWEN_DOM_DROP_SELECTORS)
        try:
            raw = self.page.evaluate(
                f"""
() => {{
  const rootSelectors = [{root_selectors_js}];
  const textSelectors = [{text_selectors_js}];
  const dropSelectors = [{drop_selectors_js}];
  const dropLinePat = /^(?:跳过|skip|pass)$/i;

  const norm = (s) => {{
    if (!s) return '';
    return String(s)
      .replace(/\\u200b/g, '')
      .replace(/\\r/g, '')
      .split('\\n')
      .map((x) => x.trim())
      .filter(Boolean)
      .join('\\n')
      .trim();
  }};

  const uniqueRoots = [];
  const seenRoots = new Set();
  for (const sel of rootSelectors) {{
    for (const el of document.querySelectorAll(sel)) {{
      if (!seenRoots.has(el)) {{
        seenRoots.add(el);
        uniqueRoots.push(el);
      }}
    }}
  }}
  if (!uniqueRoots.length) return '';

  const collectFromRoot = (root) => {{
    const picked = [];
    const seenText = new Set();

    const pushText = (rawText) => {{
      let t = norm(rawText);
      if (!t) return;
      const lines = t
        .split('\\n')
        .map((x) => x.trim())
        .filter((x) => x && !dropLinePat.test(x));
      t = norm(lines.join('\\n'));
      if (!t || seenText.has(t)) return;
      seenText.add(t);
      picked.push(t);
    }};

    for (const sel of textSelectors) {{
      for (const el of root.querySelectorAll(sel)) {{
        let blocked = false;
        for (const ds of dropSelectors) {{
          if (el.matches(ds) || el.closest(ds)) {{
            blocked = true;
            break;
          }}
        }}
        if (blocked) continue;
        pushText(el.innerText || '');
      }}
    }}

    if (!picked.length) {{
      const clone = root.cloneNode(true);
      for (const ds of dropSelectors) {{
        for (const bad of clone.querySelectorAll(ds)) bad.remove();
      }}
      pushText(clone.innerText || '');
    }}

    if (!picked.length) return '';
    picked.sort((a, b) => a.length - b.length);
    return picked[picked.length - 1];
  }};

  let last = '';
  for (const root of uniqueRoots) {{
    const t = collectFromRoot(root);
    if (t) last = t;
  }}
  return last || '';
}}
"""
            )
        except Exception:
            return ""

        return self._clean_candidate_text(core.normalize_text(str(raw or "")))

    def _extract_last_reply_candidate(self) -> str:
        if self.page is None:
            return ""

        dom_path_reply = self._extract_last_reply_by_dom_path()
        if dom_path_reply:
            return dom_path_reply

        best = ""
        best_score = float("-inf")
        seen: set[str] = set()
        for sel, strict in self._iter_reply_selectors():
            try:
                loc = self.page.locator(sel)
                cnt = loc.count()
            except Exception:
                continue
            if cnt <= 0:
                continue

            start = max(0, cnt - (6 if strict else 4))
            for i in range(cnt - 1, start - 1, -1):
                node = loc.nth(i)
                try:
                    txt = self._clean_candidate_text(node.inner_text())
                except Exception:
                    continue
                if not txt:
                    continue
                key = self._line_dedupe_key(txt)
                if not key or key in seen:
                    continue
                seen.add(key)

                penalty = self._prompt_like_penalty(txt)
                if penalty >= 2:
                    continue
                if len(key) >= 1800:
                    # Usually means the selector hit a whole-thread container.
                    continue
                line_cnt = len([ln for ln in txt.splitlines() if ln.strip()])
                if line_cnt >= 40:
                    continue
                if len(key) < 12:
                    continue

                if strict and len(key) >= 34 and not self._QWEN_PASS_PAT.match(txt):
                    return txt

                score = float(len(key))
                if len(key) >= 24:
                    score += 60.0
                elif len(key) < 24:
                    score -= 120.0
                if cnt > 1:
                    score += (i / max(1, cnt - 1)) * 120.0
                if self._QWEN_PASS_PAT.match(txt):
                    score -= 180.0
                score -= float(penalty * 120)
                if strict:
                    score += 120.0

                if score > best_score or (score == best_score and len(txt) > len(best)):
                    best_score = score
                    best = txt
        return best

    def wait_reply_and_extract(self, before_snapshot: str, timeout_s: int = 600) -> str:
        if self.page is None:
            return ""

        before_keys = {
            self._line_dedupe_key(ln)
            for ln in core.normalize_text(before_snapshot).splitlines()
            if self._line_dedupe_key(ln) and len(self._line_dedupe_key(ln)) >= 6
        }
        before_last = core.normalize_text(self._extract_last_reply_candidate())
        before_main = ""
        try:
            before_main = core.normalize_text(self.page.locator("main").first.inner_text())
        except Exception:
            before_main = ""
        prev_candidates: list[str] = []
        for cand in (core.normalize_text(self._last_effective_reply), before_last):
            if cand and cand not in prev_candidates:
                prev_candidates.append(cand)

        def _strip_by_prev(cur: str, prev: str) -> str:
            cur = core.normalize_text(cur)
            prev = core.normalize_text(prev)
            if not cur:
                return ""
            if not prev:
                return cur
            if cur == prev:
                return ""
            if cur.startswith(prev):
                tail = core.normalize_text(cur[len(prev) :])
                if tail:
                    return tail

            prev_lines = [ln.strip() for ln in prev.splitlines() if ln.strip()]
            cur_lines = [ln.strip() for ln in cur.splitlines() if ln.strip()]
            prev_keys = [self._line_dedupe_key(ln) for ln in prev_lines]
            cur_keys = [self._line_dedupe_key(ln) for ln in cur_lines]
            prev_keys = [k for k in prev_keys if k]
            cur_keys = [k for k in cur_keys if k]
            if prev_keys and cur_keys:
                common = 0
                max_common = min(len(prev_keys), len(cur_keys))
                while common < max_common and prev_keys[common] == cur_keys[common]:
                    common += 1
                if common >= max(2, len(prev_keys) - 1) and len(cur_lines) > common:
                    tail = core.normalize_text("\n".join(cur_lines[common:]))
                    if tail:
                        return tail
            return cur

        def _trim_old_prefix(text: str) -> str:
            lines = [ln.strip() for ln in core.normalize_text(text).splitlines() if ln.strip()]
            if len(lines) <= 1:
                return core.normalize_text(text)
            idx = 0
            while idx < len(lines) - 1:
                k = self._line_dedupe_key(lines[idx])
                if not k or k not in before_keys:
                    break
                idx += 1
            if idx <= 0:
                return core.normalize_text(text)
            tail = core.normalize_text("\n".join(lines[idx:]))
            return tail or core.normalize_text(text)

        def _drop_history_lines(text: str) -> str:
            lines = [ln.strip() for ln in core.normalize_text(text).splitlines() if ln.strip()]
            if len(lines) < 3:
                return core.normalize_text(text)
            kept: list[str] = []
            dropped = 0
            for ln in lines:
                k = self._line_dedupe_key(ln)
                if k and len(k) >= 8 and k in before_keys:
                    dropped += 1
                    continue
                kept.append(ln)
            if kept and (dropped >= 2 or (dropped * 1.0 / max(1, len(lines))) >= 0.45):
                out = core.normalize_text("\n".join(kept))
                if out:
                    return out
            return core.normalize_text(text)

        def _to_incremental(text: str) -> str:
            cur = core.normalize_text(self._clean_candidate_text(text))
            if not cur:
                return ""
            for prev in prev_candidates:
                if cur == prev:
                    return ""
            best_inc = cur
            for prev in prev_candidates:
                tail = core.normalize_text(self._clean_candidate_text(_strip_by_prev(cur, prev)))
                if tail and tail != cur:
                    if not best_inc or len(tail) < len(best_inc):
                        best_inc = tail
            snap_tail = core.normalize_text(self._clean_candidate_text(self._diff_reply(before_snapshot, cur)))
            if snap_tail and snap_tail != cur:
                if not best_inc or len(snap_tail) < len(best_inc):
                    best_inc = snap_tail
            best_inc = _trim_old_prefix(best_inc)
            best_inc = _drop_history_lines(best_inc)
            if self._QWEN_PASS_PAT.match(best_inc):
                return ""
            if self._QWEN_LOW_VALUE_PROCESS_PAT.match(best_inc):
                return ""
            if self._QWEN_TOPIC_PROCESS_PAT.search(best_inc):
                return ""
            if self._QWEN_PROCESS_LABEL_PAT.match(best_inc):
                return ""
            if self._is_process_title_line(best_inc):
                return ""
            return best_inc

        def _commit(reply: str) -> str:
            out = core.normalize_text(reply)
            if out:
                self._last_effective_reply = out
            return out

        def _looks_incomplete(reply: str) -> bool:
            t = core.normalize_text(reply)
            if not t:
                return True
            if re.search(r"</?\s*think\s*>", t, re.I):
                return True
            if re.search(r"(?:我需要\s*[:：]|让我(?:构思|想一下|数一下|再确认)|检查字数|字数符合要求|好，就这个)", t):
                return True
            if self._QWEN_SHORT_COMPLETE_PAT.match(t):
                if self._QWEN_LOW_VALUE_PROCESS_PAT.match(t) or self._QWEN_PROCESS_LABEL_PAT.match(t):
                    return True
                if self._is_process_title_line(t):
                    return True
                return False
            klen = len(self._line_dedupe_key(t))
            if klen < 12:
                if re.fullmatch(r"[\u4e00-\u9fff]{4,10}", re.sub(r"\s+", "", t)):
                    if self._QWEN_LOW_VALUE_PROCESS_PAT.match(t) or self._QWEN_PROCESS_LABEL_PAT.match(t):
                        return True
                    if self._is_process_title_line(t):
                        return True
                    return False
                return True
            if "\n" not in t and klen < 28 and self._QWEN_HEADING_FRAGMENT_PAT.search(t):
                return True
            if self._QWEN_LOW_VALUE_PROCESS_PAT.match(t) or self._QWEN_PROCESS_LABEL_PAT.match(t):
                return True
            if self._QWEN_TOPIC_PROCESS_PAT.search(t):
                return True
            if self._is_process_title_line(t):
                return True
            return False

        def _looks_partial(reply: str) -> bool:
            t = core.normalize_text(reply)
            if not t:
                return True
            if self._QWEN_SHORT_COMPLETE_PAT.match(t):
                return False
            if t.endswith(("。", "！", "？", "!", "?", "…", "）", ")", "】", "]", "\"", "'")):
                return False
            lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
            if len(lines) >= 2 and any(
                ln.endswith(("。", "！", "？", "!", "?", "…")) for ln in lines[-2:]
            ):
                return False
            return len(self._line_dedupe_key(t)) < 42

        best = ""
        best_at = 0.0
        last_change_at = 0.0
        force_return_after = 1.4
        saw_main_progress = False
        last_stable_probe_at = 0.0

        begin = time.time()
        while time.time() - begin < timeout_s:
            cur_last = core.normalize_text(self._extract_last_reply_candidate())
            incr = _to_incremental(cur_last)
            if incr:
                now = time.time()
                if incr != best and (len(incr) > len(best) or len(incr) >= max(8, len(best) - 8)):
                    best = incr
                    best_at = now
                    last_change_at = now
                elif not last_change_at:
                    last_change_at = now

                # Fast path: if candidate stays unchanged for a short window, return early.
                if best and len(best) >= 8 and (now - last_change_at) >= 0.65 and not _looks_incomplete(best):
                    if _looks_partial(best) and (self._is_qwen_generating() or (now - best_at) < 2.2):
                        pass
                    else:
                        return _commit(best)

                # Partial-output probe: run less frequently to reduce extraction latency jitter.
                if (
                    best
                    and _looks_partial(best)
                    and not self._is_qwen_generating()
                    and (time.time() - last_stable_probe_at) >= 2.8
                ):
                    last_stable_probe_at = time.time()
                    stable = _to_incremental(
                        core.read_stable_text(lambda: self._extract_last_reply_candidate(), "Qwen", rounds=1)
                    )
                    if stable and stable != before_last and len(stable) >= max(8, len(best) - 8):
                        best = stable
                        best_at = time.time()
                        last_change_at = best_at
                    if best and time.time() - best_at >= 0.18 and not _looks_incomplete(best):
                        if _looks_partial(best) and (time.time() - best_at) < 2.2:
                            pass
                        else:
                            return _commit(best)

                # Fallback against false-positive generating state.
                if best and len(best) >= 24 and time.time() - best_at >= force_return_after:
                    return _commit(best)

            # Fallback: selector may miss transiently on some Qwen UI versions.
            if before_main:
                try:
                    cur_main = core.normalize_text(self.page.locator("main").first.inner_text())
                except Exception:
                    cur_main = ""
                if cur_main and cur_main != before_main:
                    saw_main_progress = True
                    main_incr = _to_incremental(self._diff_reply(before_main, cur_main))
                    if main_incr and self._prompt_like_penalty(main_incr) <= 1:
                        now2 = time.time()
                        if main_incr != best and (len(main_incr) > len(best) or len(main_incr) >= max(8, len(best) - 8)):
                            best = main_incr
                            best_at = now2
                            last_change_at = now2
                        # Fast return from main-diff when stable briefly.
                        if best and len(best) >= 8 and (now2 - last_change_at) >= 0.5 and not _looks_incomplete(best):
                            if _looks_partial(best) and (self._is_qwen_generating() or (now2 - best_at) < 2.2):
                                pass
                            else:
                                return _commit(best)

            elif best and not self._is_qwen_generating() and (time.time() - best_at) >= 0.45 and not _looks_incomplete(best):
                if _looks_partial(best) and (time.time() - best_at) < 2.2:
                    pass
                else:
                    return _commit(best)

            # No useful progress detected for too long: fail fast to avoid stalling the whole group loop.
            if not best and (time.time() - begin) >= min(float(timeout_s) * 0.78, 16.0):
                if not self._is_qwen_generating() and not saw_main_progress:
                    probe = _to_incremental(
                        core.read_stable_text(lambda: self._extract_last_reply_candidate(), "Qwen", rounds=1)
                    )
                    if probe and not _looks_incomplete(probe):
                        return _commit(probe)
                    return ""

            time.sleep(0.22)

        if best and not _looks_incomplete(best):
            return _commit(best)

        last = _to_incremental(self._extract_last_reply_candidate())
        if last and not _looks_incomplete(last):
            return _commit(last)

        # Last-resort diff (kept small and strict to avoid whole-thread contamination).
        stable_snapshot = core.normalize_text(core.read_stable_text(lambda: self.snapshot_conversation(), "Qwen", rounds=2))
        reply = _to_incremental(self._diff_reply(before_snapshot, stable_snapshot))
        if (
            reply
            and self._prompt_like_penalty(reply) <= 1
            and not self._QWEN_STATUS_PAT.match(reply)
            and len(self._line_dedupe_key(reply)) >= 12
        ):
            return _commit(reply)
        return ""

    def _reply_selectors(self) -> list[str]:
        picked: list[str] = []
        for sel in self._QWEN_STRICT_SELECTORS:
            if sel not in picked:
                picked.append(sel)
        for sel in self._QWEN_FALLBACK_SELECTORS:
            if sel not in picked:
                picked.append(sel)
        for sel in super()._reply_selectors():
            if sel not in picked:
                picked.append(sel)
        return picked


class DoubaoAdapter(GenericWebChatAdapter):
    """Doubao-specific adapter: try entering chat surface and enabling deep-think mode."""

    _ENTER_CHAT_PAT = re.compile(r"新对话|开始对话|进入对话|继续对话|开始聊天|立即体验", re.I)
    _DEEP_THINK_PAT = re.compile(r"深度思考|深度模式|深度推理|深度", re.I)
    _THINK_OPTION_PAT = re.compile(r"思考|深度思考", re.I)
    _QUICK_MODE_PAT = re.compile(r"快速|标准|普通|默认", re.I)
    _MODE_PICKER_PAT = re.compile(r"快速|思考|专家", re.I)
    _THINK_STATUS_PAT = re.compile(r"^(思考|思考中|正在思考|已完成思考|完成思考)$", re.I)
    _THINK_BLOCK_HINTS: tuple[str, ...] = (
        "已完成思考",
        "用户现在需要",
        "请优先回应",
        "请先回应群主插话",
        "再基于上一位",
        "像真人群聊一样自然接话",
        "再基于当前讨论",
        "首先，组织语言",
        "这样符合要求",
        "整理一下",
        "整理成自然",
        "简洁自然",
    )
    _SUGGEST_CHIP_PAT = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9，,、\s]{2,34}[。？！?!]?$")
    _INTRO_LINE_PAT = re.compile(r"^我是[\u4e00-\u9fffA-Za-z0-9，,、\s]{6,}")
    _THOUGHT_FRAGMENT_PAT = re.compile(
        r"(深度思考|思考中|正在思考|用户现|用户现在需要|跳过思考|直接回答|回顾|稍等|等等|先想|整理一下|整理成自然|简洁自然|首先[，,:：].{0,18}(?:回应|回复|组织语言)|这样符合要求)",
        re.I,
    )
    _DOUBAO_HOST_CHATTER_PAT = re.compile(
        r"(?:^|[，,。!！~～\s])(?:好(?:的|嘞)?|收到|明白|行(?:吧)?|ok|OK)(?:[，,。!！~～\s]{0,3})(?:群主|主持人|老大|老板)|"
        r"(谁先来|我们这就开始|先来(?:出)?第一个|我先来抛砖引玉|接龙(?:开始|走起)|继续接龙)",
        re.I,
    )
    _DOUBAO_SHORT_COMPLETE_PAT = re.compile(
        r"^\s*[\u4e00-\u9fff]{4,10}(?:\s*\n\s*@[\w\u4e00-\u9fff-]{1,24})?\s*$",
        re.I,
    )
    _DOUBAO_STOP_PAT = re.compile(r"停止生成|停止回答|停止输出|stop generating|stop response|stop", re.I)
    _DOUBAO_GENERATING_PAT = re.compile(r"深度思考中|思考中|正在思考|生成中|写作中|回答中", re.I)
    _DOUBAO_STRICT_SELECTORS: tuple[str, ...] = (
        "main [data-role='assistant']",
        "main [data-message-role='assistant']",
        "main [data-message-author-role='assistant']",
        "main [data-testid*='assistant' i]",
    )
    _DOUBAO_FALLBACK_SELECTORS: tuple[str, ...] = (
        "main [class*='assistant-message']",
        "main [class*='assistant'] [class*='markdown']",
        "main [class*='answer-content']",
        "main [class*='chat-item'] [class*='markdown']",
    )
    _CHIP_HINTS: tuple[str, ...] = (
        "造个句",
        "有哪些技巧",
        "有哪些规则",
        "推荐一些",
        "提供一些",
        "成语接龙",
        "生成器",
        "视频生成",
        "图片生成",
        "图像生成",
        "PPT生成",
        "PPT 生成",
        "帮我写作",
        "文案",
        "海报",
        "插画",
        "头像",
        "周报",
        "大纲",
        "翻译",
        "超能模式",
        "偏好话题",
        "你可以",
        "帮我",
        "可以吗",
    )

    def __init__(self, meta: ModelMeta) -> None:
        super().__init__(meta)
        self._deepthink_enabled = False
        self._deepthink_attempts = 0
        self._deepthink_last_probe_ts = 0.0
        self._deepthink_last_warn_ts = 0.0
        self._recent_sent_line_keys: list[set[str]] = []
        self._last_effective_reply: str = ""

    def _click_visible(self, locator: Locator, *, prefer_last: bool = False) -> bool:
        node = core.pick_visible(locator, prefer_last=prefer_last)
        if node is None:
            return False
        try:
            node.click(timeout=2500)
            time.sleep(0.5)
            return True
        except Exception:
            return False

    def _ensure_chat_entry(self) -> None:
        if self.page is None:
            return
        self._dismiss_restore_popup()
        if super().find_input() is not None:
            return
        # Some Doubao pages open at a launcher/landing panel before chat textarea appears.
        self._click_visible(self.page.get_by_role("button", name=self._ENTER_CHAT_PAT))
        if super().find_input() is not None:
            return
        self._click_visible(self.page.get_by_text(self._ENTER_CHAT_PAT))
        if super().find_input() is not None:
            return
        try:
            self.page.goto(self.meta.url, wait_until="domcontentloaded")
            time.sleep(0.6)
        except Exception:
            pass

    def _dismiss_restore_popup(self) -> None:
        if self.page is None:
            return
        marker = core.pick_visible(
            self.page.get_by_text(re.compile(r"要恢复页面吗|Chromium ?未正确关闭|Restore pages", re.I)),
            prefer_last=False,
        )
        if marker is None:
            return
        try:
            self.page.keyboard.press("Escape")
            time.sleep(0.2)
        except Exception:
            pass
        # Try close/dismiss; avoid clicking "恢复".
        for loc in (
            self.page.get_by_role("button", name=re.compile(r"关闭|取消|dismiss|close|x", re.I)),
            self.page.locator("button[aria-label*='close' i], button[aria-label*='关闭']"),
        ):
            if self._click_visible(loc, prefer_last=False):
                return

    @staticmethod
    def _is_toggle_on(node: Locator) -> Optional[bool]:
        try:
            pressed = (node.get_attribute("aria-pressed") or "").strip().lower()
            if pressed in {"true", "1"}:
                return True
            if pressed in {"false", "0"}:
                return False
            checked = (node.get_attribute("aria-checked") or "").strip().lower()
            if checked in {"true", "1"}:
                return True
            if checked in {"false", "0"}:
                return False
            for attr in ("data-state", "data-status", "data-selected", "data-active", "aria-current"):
                v = (node.get_attribute(attr) or "").strip().lower()
                if v in {"true", "1", "on", "active", "selected", "checked", "yes", "page"}:
                    return True
                if v in {"false", "0", "off", "inactive", "unselected", "unchecked", "no"}:
                    return False
            cls = (node.get_attribute("class") or "").lower()
            if any(x in cls for x in ("active", "selected", "checked", "on")):
                return True
            if any(x in cls for x in ("inactive", "unselected", "off", "disabled")):
                return False
            return None
        except Exception:
            return None

    def _pick_mode_button(self, pat: re.Pattern[str], *, prefer_last: bool = True) -> Optional[Locator]:
        if self.page is None:
            return None
        candidates = (
            self.page.locator("button,[role='button'],[role='switch'],label", has_text=pat),
            self.page.get_by_role("button", name=pat),
            self.page.get_by_text(pat),
        )
        for loc in candidates:
            node = core.pick_visible(loc, prefer_last=prefer_last)
            if node is not None:
                return node
        return None

    def _reply_selectors(self) -> list[str]:
        # Doubao page has many toolbars/chips; avoid overly broad generic selectors.
        picked: list[str] = []
        for sel in self._DOUBAO_STRICT_SELECTORS:
            if sel not in picked:
                picked.append(sel)
        for sel in self._DOUBAO_FALLBACK_SELECTORS:
            if sel not in picked:
                picked.append(sel)
        return picked

    def _iter_reply_selectors(self) -> list[tuple[str, bool]]:
        out: list[tuple[str, bool]] = []
        seen: set[str] = set()
        for sel in self._DOUBAO_STRICT_SELECTORS:
            if sel not in seen:
                seen.add(sel)
                out.append((sel, True))
        for sel in self._DOUBAO_FALLBACK_SELECTORS:
            if sel not in seen:
                seen.add(sel)
                out.append((sel, False))
        return out

    def _remember_recent_sent_lines(self, text: str) -> None:
        t = core.normalize_text(text)
        if not t:
            return
        keys: set[str] = set()
        for ln in t.splitlines():
            k = self._line_dedupe_key(ln)
            if k:
                keys.add(k)
        if not keys:
            return
        self._recent_sent_line_keys.append(keys)
        if len(self._recent_sent_line_keys) > 4:
            self._recent_sent_line_keys = self._recent_sent_line_keys[-4:]

    def _is_doubao_thought_like(self, text: str) -> bool:
        t = core.normalize_text(text)
        if not t:
            return True
        if self._THINK_STATUS_PAT.match(t):
            return True
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        if not lines:
            return True
        thought_like = 0
        for ln in lines:
            if self._THINK_STATUS_PAT.match(ln):
                thought_like += 1
                continue
            if self._THOUGHT_FRAGMENT_PAT.search(ln):
                thought_like += 1
                continue
            if any(h in ln for h in self._THINK_BLOCK_HINTS):
                thought_like += 1
                continue
        compact = re.sub(r"\s+", "", t)
        if len(compact) <= 8 and "\n" not in t and "@" not in t:
            if not re.search(r"[。？！?!]", t):
                return True
        if thought_like >= len(lines):
            return True
        if thought_like >= max(2, int(len(lines) * 0.75)) and len(compact) <= 140:
            return True
        return False

    def _looks_like_recent_prompt_echo(self, line: str) -> bool:
        k = self._line_dedupe_key(line)
        if not k:
            return False
        for hist in self._recent_sent_line_keys:
            if k in hist:
                return True
        return False

    def _prompt_echo_overlap_ratio(self, text: str) -> float:
        t = core.normalize_text(text)
        if not t:
            return 0.0
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        if not lines:
            return 0.0

        recent_keys: set[str] = set()
        for hist in self._recent_sent_line_keys[-3:]:
            recent_keys.update(hist)
        recent_list = [k for k in recent_keys if len(k) >= 8]

        total = 0
        hit = 0
        for ln in lines:
            k = self._line_dedupe_key(ln)
            if not k:
                continue
            total += 1
            if any(h in ln for h in self._PROMPT_ECHO_HINTS):
                hit += 1
                continue
            if "上下文边界" in ln or "只依据以下群聊消息回复" in ln or "不要复述本提示" in ln:
                hit += 1
                continue
            if k in recent_keys:
                hit += 1
                continue
            if len(k) >= 12:
                for old in recent_list[-40:]:
                    if k == old:
                        hit += 1
                        break
                    if len(old) >= 12 and (k in old or old in k):
                        hit += 1
                        break
        if total <= 0:
            return 0.0
        return hit / float(total)

    @classmethod
    def _is_chip_line(cls, line: str) -> bool:
        s = core.normalize_text(line)
        if not s:
            return False
        if "|" in s or "｜" in s:
            segs = [x.strip() for x in re.split(r"[|｜]+", s) if x.strip()]
            if len(segs) >= 2:
                chip_like = sum(1 for x in segs if cls._is_chip_line(x))
                if chip_like >= max(2, len(segs) - 1):
                    return True
        if "→" in s or "->" in s or "›" in s:
            return True
        k = cls._line_dedupe_key(s)
        if not k:
            return False
        if len(k) > 30:
            return False
        if any(h in s for h in cls._CHIP_HINTS):
            return True
        if re.match(r"^(询问|提问|追问)[A-Za-z0-9_\-\u4e00-\u9fff]{0,18}(?:新功能|话题|问题|偏好|能力|类型)$", s):
            return True
        if len(k) <= 24 and not re.search(r"[。？！?!，,；;:：]", s):
            if any(h in s for h in ("推荐", "分享", "生成器", "文案", "技巧", "规则", "话题", "示例")):
                return True
            if s.endswith(("推荐", "分享", "问题", "类型")):
                return True
        if len(k) <= 24 and not re.search(r"[。？！?!，,；;:：]", s):
            if re.match(r"^(?:写一份|生成|推荐|提供|帮我)", s):
                return True
            if re.search(r"(?:视频|图片|图像|海报|插画|头像|PPT|文案)?生成$", s):
                return True
        # very short question-like tail prompts
        if len(k) <= 22 and ("吗" in s or "？" in s or "?" in s):
            return True
        return False

    def _clean_candidate_text(self, text: str) -> str:
        cleaned = super()._clean_candidate_text(text)
        if not cleaned:
            return ""

        lines: list[str] = []
        for ln in cleaned.splitlines():
            s = ln.strip()
            if not s:
                continue
            if any(
                h in s
                for h in (
                    "<<<PUBLIC_REPLY>>>",
                    "<<<END_PUBLIC_REPLY>>>",
                    "<<<PRIVATE_REPLY>>>",
                    "<<<END_PRIVATE_REPLY>>>",
                    "[[PUBLIC_REPLY]]",
                    "[[/PUBLIC_REPLY]]",
                    "[[PRIVATE_REPLY]]",
                    "[[/PRIVATE_REPLY]]",
                )
            ):
                continue
            if re.search(r"(请把公开发言放在|公开发言放在|如需隐藏想法|暂不支持该消息类型)", s):
                continue
            if self._is_chip_line(s):
                continue
            if self._DOUBAO_HOST_CHATTER_PAT.search(s) and len(self._line_dedupe_key(s)) <= 56:
                continue
            if re.match(r"^(?:@?[\w\u4e00-\u9fff-]{1,20}\s*)?(?:接|你接|请接|来接)\s*[~～!！。\.]*$", s):
                continue
            if self._THINK_STATUS_PAT.match(s):
                continue
            if self._THOUGHT_FRAGMENT_PAT.search(s):
                continue
            if self._looks_like_recent_prompt_echo(s):
                continue
            if any(h in s for h in self._PROMPT_ECHO_HINTS):
                continue
            if re.search(r"(多人群聊中发言|群主最新话题|最终发言要求|只输出给群里的正文|如暂不发言)", s):
                continue
            lines.append(s)

        # Drop heading-like title line often rendered above answer body.
        if lines and len(lines[0]) <= 18 and ("介绍" in lines[0] or "建议" in lines[0] or "总结" in lines[0]):
            lines = lines[1:]

        # Drop trailing suggestion chips like "豆包，你可以...？" when they appear in a batch.
        tail_idx = len(lines)
        chip_count = 0
        while tail_idx > 0:
            s = lines[tail_idx - 1]
            if self._SUGGEST_CHIP_PAT.match(s) or self._is_chip_line(s):
                tail_idx -= 1
                chip_count += 1
                continue
            break
        if chip_count >= 2:
            lines = lines[:tail_idx]

        # If what remains still looks like chip-only cards, discard this candidate.
        if lines:
            chip_like = sum(1 for s in lines if self._is_chip_line(s))
            if chip_like >= max(1, len(lines) - 1):
                return ""

        # If multiple intro-like blocks are concatenated, keep the latest block only.
        intro_idxs = [i for i, ln in enumerate(lines) if self._INTRO_LINE_PAT.match(ln)]
        if len(intro_idxs) >= 2:
            last_intro = intro_idxs[-1]
            if last_intro < len(lines) - 1:
                lines = lines[last_intro:]

        cleaned = core.normalize_text("\n".join(lines))
        if not cleaned:
            return ""
        if self._prompt_echo_overlap_ratio(cleaned) >= 0.45:
            return ""
        if self._is_doubao_thought_like(cleaned):
            return ""
        return cleaned

    def _is_doubao_generating(self) -> bool:
        if self.page is None:
            return False

        stop_candidates = (
            self.page.get_by_role("button", name=self._DOUBAO_STOP_PAT),
            self.page.locator("button[aria-label*='停止'], button[title*='停止']"),
            self.page.get_by_text(re.compile(r"停止生成|停止回答", re.I)),
        )
        for loc in stop_candidates:
            node = core.pick_visible(loc, prefer_last=True)
            if node is not None:
                return True

        marker = core.pick_visible(self.page.get_by_text(self._DOUBAO_GENERATING_PAT), prefer_last=True)
        return marker is not None

    def _extract_last_reply_candidate(self) -> str:
        if self.page is None:
            return ""

        fallback = ""
        fallback_score = float("-inf")
        seen: set[str] = set()
        for sel, strict in self._iter_reply_selectors():
            try:
                loc = self.page.locator(sel)
                cnt = loc.count()
            except Exception:
                continue
            if cnt <= 0:
                continue

            start = max(0, cnt - 8)
            # Prefer the latest visible assistant block instead of longest historical one.
            for i in range(cnt - 1, start - 1, -1):
                node = loc.nth(i)
                try:
                    txt = self._clean_candidate_text(node.inner_text())
                except Exception:
                    continue
                if not txt:
                    continue

                key = self._line_dedupe_key(txt)
                if not key or key in seen:
                    continue
                seen.add(key)
                if self._prompt_echo_overlap_ratio(txt) >= 0.5:
                    continue
                if self._is_doubao_thought_like(txt):
                    continue
                line_cnt = len([ln for ln in txt.splitlines() if ln.strip()])
                if len(key) >= 1500 or line_cnt >= 28:
                    # Likely selected a whole-thread container.
                    continue
                if len(key) < 10:
                    continue

                if strict and len(key) >= 16:
                    return txt
                if len(key) >= 16:
                    return txt

                score = float(len(key))
                if cnt > 1:
                    score += (i / max(1, cnt - 1)) * 40.0
                if strict:
                    score += 60.0
                if score > fallback_score:
                    fallback_score = score
                    fallback = txt
        return fallback

    def wait_reply_and_extract(self, before_snapshot: str, timeout_s: int = 600) -> str:
        if self.page is None:
            return ""

        before_keys = {
            self._line_dedupe_key(ln)
            for ln in core.normalize_text(before_snapshot).splitlines()
            if self._line_dedupe_key(ln) and len(self._line_dedupe_key(ln)) >= 6
        }
        before_last = core.normalize_text(self._extract_last_reply_candidate())
        prev_candidates: list[str] = []
        for cand in (core.normalize_text(self._last_effective_reply), before_last):
            if cand and cand not in prev_candidates:
                prev_candidates.append(cand)

        def _strip_by_prev(cur: str, prev: str) -> str:
            cur = core.normalize_text(cur)
            prev = core.normalize_text(prev)
            if not cur:
                return ""
            if not prev:
                return cur
            if cur == prev:
                return ""
            if cur.startswith(prev):
                tail = core.normalize_text(cur[len(prev) :])
                if tail:
                    return tail
            prev_lines = [ln.strip() for ln in prev.splitlines() if ln.strip()]
            cur_lines = [ln.strip() for ln in cur.splitlines() if ln.strip()]
            prev_keys = [self._line_dedupe_key(ln) for ln in prev_lines]
            cur_keys = [self._line_dedupe_key(ln) for ln in cur_lines]
            prev_keys = [k for k in prev_keys if k]
            cur_keys = [k for k in cur_keys if k]
            if prev_keys and cur_keys:
                # Strong line-prefix overlap: cur is likely cumulative block = prev + delta.
                common = 0
                max_common = min(len(prev_keys), len(cur_keys))
                while common < max_common and prev_keys[common] == cur_keys[common]:
                    common += 1
                if common >= max(2, len(prev_keys) - 1) and len(cur_lines) > common:
                    tail = core.normalize_text("\n".join(cur_lines[common:]))
                    if tail:
                        return tail

                # Fallback: prev tail matches cur head (handles minor format shifts).
                max_overlap = min(len(prev_keys), len(cur_keys) - 1)
                for overlap in range(max_overlap, 1, -1):
                    if prev_keys[-overlap:] == cur_keys[:overlap]:
                        tail = core.normalize_text("\n".join(cur_lines[overlap:]))
                        if tail:
                            return tail
            prev_compact = re.sub(r"\s+", "", prev)
            cur_compact = re.sub(r"\s+", "", cur)
            if prev_compact and cur_compact.startswith(prev_compact):
                # Compact-prefix fallback: keep tail by line diff when spacing changed.
                prev_lines = [ln for ln in prev.splitlines() if ln.strip()]
                cur_lines = [ln for ln in cur.splitlines() if ln.strip()]
                if len(cur_lines) >= len(prev_lines) and cur_lines[: len(prev_lines)] == prev_lines:
                    tail = core.normalize_text("\n".join(cur_lines[len(prev_lines) :]))
                    if tail:
                        return tail
            return cur

        def _trim_old_prefix(text: str) -> str:
            lines = [ln.strip() for ln in core.normalize_text(text).splitlines() if ln.strip()]
            if len(lines) <= 1:
                return core.normalize_text(text)
            idx = 0
            while idx < len(lines) - 1:
                k = self._line_dedupe_key(lines[idx])
                if not k or k not in before_keys:
                    break
                idx += 1
            if idx <= 0:
                return core.normalize_text(text)
            tail = core.normalize_text("\n".join(lines[idx:]))
            return tail or core.normalize_text(text)

        def _drop_history_lines(text: str) -> str:
            lines = [ln.strip() for ln in core.normalize_text(text).splitlines() if ln.strip()]
            if len(lines) < 3:
                return core.normalize_text(text)
            kept: list[str] = []
            dropped = 0
            for ln in lines:
                k = self._line_dedupe_key(ln)
                if k and len(k) >= 8 and k in before_keys:
                    dropped += 1
                    continue
                kept.append(ln)
            if kept and (dropped >= 2 or (dropped * 1.0 / max(1, len(lines))) >= 0.45):
                out = core.normalize_text("\n".join(kept))
                if out:
                    return out
            return core.normalize_text(text)

        def _to_incremental(text: str) -> str:
            cur = core.normalize_text(text)
            if not cur:
                return ""
            if re.search(
                r"(多人群聊中发言|群主最新话题|最终发言要求|只输出给群里的正文|如暂不发言|"
                r"请把公开发言放在|公开发言放在|如需隐藏想法|暂不支持该消息类型|"
                r"PUBLIC_REPLY|PRIVATE_REPLY)",
                cur,
            ):
                return ""
            if self._prompt_echo_overlap_ratio(cur) >= 0.5:
                return ""
            for prev in prev_candidates:
                if cur == prev:
                    return ""

            best = cur
            for prev in prev_candidates:
                tail = _strip_by_prev(cur, prev)
                if tail and tail != cur:
                    if not best or len(tail) < len(best):
                        best = tail
            snap_tail = core.normalize_text(self._clean_candidate_text(self._diff_reply(before_snapshot, cur)))
            if snap_tail and snap_tail != cur:
                if not best or len(snap_tail) < len(best):
                    best = snap_tail
            if best:
                return _drop_history_lines(_trim_old_prefix(best))
            return cur

        def _commit(reply: str) -> str:
            out = core.normalize_text(reply)
            if out:
                self._last_effective_reply = out
            return out

        def _looks_partial(reply: str) -> bool:
            t = core.normalize_text(reply)
            if not t:
                return True
            if self._DOUBAO_SHORT_COMPLETE_PAT.match(t):
                return False
            if t.endswith(("。", "！", "？", "!", "?", "…", "）", ")", "】", "]", "\"", "'")):
                return False
            lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
            if len(lines) >= 2 and any(
                ln.endswith(("。", "！", "？", "!", "?", "…")) for ln in lines[-2:]
            ):
                return False
            return len(self._line_dedupe_key(t)) < 56

        def _loose_diff_extract(cur_snapshot: str) -> str:
            raw = core.normalize_text(self._diff_reply(before_snapshot, cur_snapshot))
            if not raw:
                return ""
            lines: list[str] = []
            for ln in raw.splitlines():
                s = ln.strip()
                if not s:
                    continue
                if self._NOISE_LINE_PAT.search(s):
                    continue
                if self._is_chip_line(s):
                    continue
                if self._THINK_STATUS_PAT.match(s):
                    continue
                if self._THOUGHT_FRAGMENT_PAT.search(s) and len(s) <= 36:
                    continue
                lines.append(s)
            if not lines:
                return ""
            return core.normalize_text("\n".join(lines[-10:]))

        best = ""
        best_at = 0.0
        last_stable_probe_at = 0.0

        begin = time.time()
        while time.time() - begin < timeout_s:
            cur_last = core.normalize_text(self._extract_last_reply_candidate())
            incr_cur = _to_incremental(cur_last)
            if incr_cur and not self._is_doubao_thought_like(incr_cur):
                if len(incr_cur) >= max(8, len(best) - 8):
                    best = incr_cur
                    best_at = time.time()

                if (time.time() - last_stable_probe_at) >= 1.1:
                    last_stable_probe_at = time.time()
                    stable = core.normalize_text(
                        core.read_stable_text(lambda: self._extract_last_reply_candidate(), "Doubao", rounds=2)
                    )
                    incr = _to_incremental(stable)
                    if incr and not self._is_doubao_thought_like(incr):
                        if len(incr) >= max(8, len(best) - 8):
                            best = incr
                            best_at = time.time()

            if not best:
                cur_snapshot = core.normalize_text(self.snapshot_conversation())
                if cur_snapshot and cur_snapshot != before_snapshot:
                    raw_diff = self._clean_candidate_text(self._diff_reply(before_snapshot, cur_snapshot))
                    diff = _to_incremental(raw_diff) or core.normalize_text(raw_diff)
                    if not diff:
                        diff = _loose_diff_extract(cur_snapshot)
                    if diff and not self._is_doubao_thought_like(diff):
                        line_cnt = len([ln for ln in diff.splitlines() if ln.strip()])
                        if len(diff) <= 1200 and line_cnt <= 18:
                            best = diff
                            best_at = time.time()

            if best and not self._is_doubao_generating() and (time.time() - best_at) >= 0.7:
                if _looks_partial(best):
                    cur_snapshot = core.normalize_text(self.snapshot_conversation())
                    raw_diff = self._clean_candidate_text(self._diff_reply(before_snapshot, cur_snapshot))
                    diff = _to_incremental(raw_diff) or core.normalize_text(raw_diff)
                    if diff and len(self._line_dedupe_key(diff)) >= len(self._line_dedupe_key(best)) + 8:
                        best = diff
                        best_at = time.time()
                    if (time.time() - best_at) < 5.5:
                        pass
                    else:
                        return _commit(best)
                else:
                    return _commit(best)
            if not best and (time.time() - begin) >= min(float(timeout_s) * 0.80, 16.0):
                if not self._is_doubao_generating():
                    probe = _to_incremental(
                        core.read_stable_text(lambda: self._extract_last_reply_candidate(), "Doubao", rounds=2)
                    )
                    if probe and not self._is_doubao_thought_like(probe):
                        return _commit(probe)
                    return ""
            time.sleep(0.28)

        if best:
            return _commit(best)

        last = core.normalize_text(self._extract_last_reply_candidate())
        incr_last = _to_incremental(last)
        if incr_last:
            return _commit(incr_last)
        # Last resort only: strict snapshot diff to reduce contamination.
        stable_snapshot = core.normalize_text(core.read_stable_text(lambda: self.snapshot_conversation(), "Doubao", rounds=3))
        raw_reply = self._clean_candidate_text(self._diff_reply(before_snapshot, stable_snapshot))
        reply = _to_incremental(raw_reply) or core.normalize_text(raw_reply)
        if not reply:
            reply = _loose_diff_extract(stable_snapshot)
        if reply and not self._is_doubao_thought_like(reply):
            return _commit(reply)
        return ""

    def _try_select_think_from_mode_picker(self) -> bool:
        if self.page is None:
            return False

        picker = self._pick_mode_button(self._MODE_PICKER_PAT, prefer_last=True)
        if picker is None:
            return False

        try:
            picker_text = core.normalize_text(picker.inner_text())
        except Exception:
            picker_text = ""
        if "思考" in picker_text and "快速" not in picker_text:
            self._deepthink_enabled = True
            return True

        try:
            picker.click(timeout=2500)
            time.sleep(0.25)
        except Exception:
            return False

        opt = core.pick_visible(
            self.page.locator("button,[role='button'],[role='menuitem'],[role='option'],label", has_text=self._THINK_OPTION_PAT),
            prefer_last=True,
        )
        if opt is None:
            return False
        try:
            opt.click(timeout=2500)
            time.sleep(0.35)
        except Exception:
            return False

        picker2 = self._pick_mode_button(self._MODE_PICKER_PAT, prefer_last=True)
        if picker2 is not None:
            try:
                text2 = core.normalize_text(picker2.inner_text())
            except Exception:
                text2 = ""
            if "思考" in text2 and "快速" not in text2:
                self._deepthink_enabled = True
                core.log("豆包：已通过模式下拉切换到“思考”。")
                return True

        self._deepthink_enabled = True
        core.log("豆包：已尝试通过模式下拉切换到“思考”。")
        return True

    def _try_enable_deepthink_once(self) -> None:
        if self.page is None:
            return
        now = time.time()
        if now - self._deepthink_last_probe_ts < 0.6:
            return
        self._deepthink_last_probe_ts = now

        # Keep probing across rounds; Doubao may reset mode when opening/creating a new chat.
        if self._deepthink_attempts >= 120:
            self._deepthink_attempts = 0
        self._deepthink_attempts += 1

        # New Doubao UI uses a mode picker: 快速 / 思考 / 专家.
        if self._try_select_think_from_mode_picker():
            return

        deep_btn = self._pick_mode_button(self._DEEP_THINK_PAT, prefer_last=True)
        if deep_btn is None:
            if now - self._deepthink_last_warn_ts >= 10:
                core.warn("豆包：未找到“思考/深度思考”模式入口，本轮跳过，后续会继续尝试。")
                self._deepthink_last_warn_ts = now
            return
        quick_btn = self._pick_mode_button(self._QUICK_MODE_PAT, prefer_last=True)

        deep_state = self._is_toggle_on(deep_btn)
        quick_state = self._is_toggle_on(quick_btn) if quick_btn is not None else None
        if deep_state is True and quick_state is not True:
            self._deepthink_enabled = True
            return

        should_click = False
        if deep_state is False or quick_state is True:
            should_click = True
        elif deep_state is None:
            # If state cannot be detected, still try once to avoid being stuck in quick mode.
            should_click = True

        if not should_click:
            return
        try:
            deep_btn.click(timeout=2500)
            time.sleep(0.45)
            deep_state2 = self._is_toggle_on(deep_btn)
            quick_state2 = self._is_toggle_on(quick_btn) if quick_btn is not None else None
            if deep_state2 is True or quick_state2 is False:
                self._deepthink_enabled = True
                core.log("豆包：已切换到深度思考模式。")
                return
            # Unknown but click succeeded: treat as enabled and continue.
            self._deepthink_enabled = True
            core.log("豆包：已尝试切换深度思考（状态未知，按成功处理）。")
        except Exception:
            if now - self._deepthink_last_warn_ts >= 10:
                core.warn("豆包：点击“深度思考”失败，本轮将继续尝试。")
                self._deepthink_last_warn_ts = now

    def find_input(self) -> Optional[Locator]:
        self._ensure_chat_entry()
        box = super().find_input()
        if box is not None:
            self._try_enable_deepthink_once()
        return box

    def send_user_text(self, text: str) -> None:
        self._ensure_chat_entry()
        self._try_enable_deepthink_once()
        self._remember_recent_sent_lines(text)
        super().send_user_text(text)
