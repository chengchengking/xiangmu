import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from playwright.sync_api import Locator, Page, TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


# 目标网站
CHATGPT_URL = "https://chatgpt.com/"
GEMINI_URL = "https://gemini.google.com/app"

# 反机械化随机延迟
def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except Exception:
        return default
    if value < 0:
        return default
    return value


ACTION_DELAY_MIN = _env_float("AI_DUEL_ACTION_DELAY_MIN", 0.15)
ACTION_DELAY_MAX = _env_float("AI_DUEL_ACTION_DELAY_MAX", 0.35)
if ACTION_DELAY_MAX < ACTION_DELAY_MIN:
    ACTION_DELAY_MAX = ACTION_DELAY_MIN

# 轮询参数
POLL_SECONDS = _env_float("AI_DUEL_POLL_SECONDS", 0.4)
MAX_WAIT_SECONDS = 600


def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def disable_windows_console_quickedit() -> None:
    """
    Windows 传统控制台有个坑：用鼠标在控制台里“选中文本”会进入 QuickEdit/标记模式，
    此时控制台输出会被暂停，看起来像脚本“卡住”了（其实脚本还在跑）。

    这里默认禁用 QuickEdit，避免误触；如需保留 QuickEdit，可设置环境变量：
    - AI_DUEL_QUICKEDIT=1
    """
    if os.name != "nt":
        return
    if os.environ.get("AI_DUEL_QUICKEDIT", "").strip() == "1":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        STD_INPUT_HANDLE = -10
        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080

        h = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        if not h or h == wintypes.HANDLE(-1).value:
            return

        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            return

        new_mode = mode.value | ENABLE_EXTENDED_FLAGS
        new_mode &= ~ENABLE_QUICK_EDIT_MODE
        kernel32.SetConsoleMode(h, new_mode)
    except Exception:
        return


def jitter(phase: str = "") -> None:
    """每次关键动作前后加入随机延时。"""
    delay = random.uniform(ACTION_DELAY_MIN, ACTION_DELAY_MAX)
    if phase:
        log(f"{phase}，随机等待 {delay:.2f}s")
    time.sleep(delay)


def normalize_text(text: Optional[str]) -> str:
    return (text or "").replace("\u200b", "").strip()


def pick_visible(locator: Locator, prefer_last: bool = True) -> Optional[Locator]:
    """在一个 Locator 集合中挑出可见元素，默认优先最后一个（通常是底部输入框）。"""
    try:
        count = locator.count()
    except Exception:
        return None

    if count <= 0:
        return None

    indexes = range(count - 1, -1, -1) if prefer_last else range(count)
    for i in indexes:
        node = locator.nth(i)
        try:
            if node.is_visible(timeout=300):
                return node
        except Exception:
            continue
    return None


def safe_is_enabled(locator: Optional[Locator]) -> bool:
    if locator is None:
        return False
    try:
        return locator.is_enabled(timeout=300)
    except Exception:
        return False


def find_chatgpt_input(page: Page) -> Optional[Locator]:
    """
    ChatGPT 输入框选择策略：
    1) 优先官方稳定 ID/结构
    严禁依赖随机 hash class。
    """
    candidates = [
        page.locator("textarea#prompt-textarea"),
        page.locator("textarea[data-testid='prompt-textarea']"),
        page.locator("div#prompt-textarea[contenteditable='true']"),
        page.locator("div[data-testid='prompt-textarea'][contenteditable='true']"),
        # placeholder 在中英文环境都有可能出现，作为兜底，但避免使用过于宽泛的 textbox 选择器
        page.get_by_placeholder(re.compile(r"message|发送消息|发消息|向.*发送消息", re.I)),
        page.locator("form textarea#prompt-textarea"),
    ]
    for locator in candidates:
        node = pick_visible(locator, prefer_last=True)
        if node is not None:
            return node
    return None


def ensure_input_ready(page: Page, site_name: str) -> Locator:
    """
    发送前兜底：如果输入框消失（UI 卡住/只读页面/欢迎弹窗遮挡），尝试自动恢复：
    - Esc 关闭弹窗
    - reload
    - goto 站点入口页
    """
    finder = find_chatgpt_input if site_name == "ChatGPT" else find_gemini_input
    url = CHATGPT_URL if site_name == "ChatGPT" else GEMINI_URL

    input_box = finder(page)
    if input_box is not None:
        return input_box

    warn(f"{site_name} 未检测到对话输入框，尝试自动恢复（Esc / 刷新 / 重新打开入口页）...")
    for attempt in range(1, 4):
        try:
            page.keyboard.press("Escape")
        except Exception:
            pass

        try:
            if attempt == 1:
                page.reload(wait_until="domcontentloaded")
            else:
                page.goto(url, wait_until="domcontentloaded")
        except Exception as exc:
            warn(f"{site_name} 自动恢复动作失败: {exc}")

        time.sleep(2)
        input_box = finder(page)
        if input_box is not None:
            log(f"{site_name} 输入框已恢复")
            return input_box

    raise RuntimeError(
        f"{site_name} 对话输入框不可用。"
        "可能原因：只读分享页/页面未渲染完成/风控或人机验证/弹窗遮挡。"
        "请手动刷新页面或点“新聊天”，确认输入框出现后再继续。"
    )


def find_gemini_input(page: Page) -> Optional[Locator]:
    """
    Gemini 输入框常见形态：rich-textarea / contenteditable / textbox。
    这里做多候选兜底，避免单点失效。
    """
    candidates = [
        page.locator("rich-textarea [contenteditable='true']"),
        page.locator("div[contenteditable='true'][aria-label*='prompt' i]"),
        page.locator("div[contenteditable='true'][aria-label*='输入']"),
        page.get_by_role("textbox"),
        page.locator("textarea"),
    ]
    for locator in candidates:
        node = pick_visible(locator, prefer_last=True)
        if node is not None:
            return node
    return None


def wait_for_login(page: Page, site_name: str) -> None:
    """
    人工干预登录检查：
    - 持续探测输入框（textarea 或 contenteditable / textbox）
    - 如果找不到，循环提示用户手动登录并关闭弹窗
    - 找到输入框后才继续
    """
    log(f"{site_name} 开始登录状态检测...")
    last_warn = 0.0
    while True:
        input_box = find_chatgpt_input(page) if site_name == "ChatGPT" else find_gemini_input(page)
        if input_box is not None:
            log(f"{site_name} 检测到输入框，登录检查通过。")
            return

        now = time.time()
        if now - last_warn > 6:
            warn("检测到未登录或有弹窗，请手动登录并关闭所有欢迎弹窗...")
            last_warn = now
        time.sleep(2)


def find_chatgpt_send_button(page: Page) -> Optional[Locator]:
    candidates = [
        page.locator("button[data-testid='send-button']"),
        page.get_by_role("button", name=re.compile(r"send|发送", re.I)),
        page.locator("form button[type='submit']"),
    ]
    for locator in candidates:
        node = pick_visible(locator, prefer_last=True)
        if node is not None:
            return node
    return None


def find_chatgpt_stop_button(page: Page) -> Optional[Locator]:
    candidates = [
        page.locator("button[data-testid='stop-button']"),
        page.get_by_role("button", name=re.compile(r"stop generating|停止生成|stop", re.I)),
        page.locator("button[aria-label*='Stop']"),
    ]
    for locator in candidates:
        node = pick_visible(locator, prefer_last=True)
        if node is not None:
            return node
    return None


def find_chatgpt_continue_button(page: Page) -> Optional[Locator]:
    candidates = [
        page.get_by_role("button", name=re.compile(r"continue generating|继续生成", re.I)),
        page.locator("button:has-text('Continue generating')"),
        page.locator("button:has-text('继续生成')"),
    ]
    for locator in candidates:
        node = pick_visible(locator, prefer_last=False)
        if node is not None:
            return node
    return None


def find_gemini_send_button(page: Page) -> Optional[Locator]:
    candidates = [
        page.get_by_role("button", name=re.compile(r"send|发送", re.I)),
        page.locator("button[aria-label*='Send']"),
        page.locator("button[aria-label*='发送']"),
        page.locator("button[mattooltip*='Send']"),
    ]
    for locator in candidates:
        node = pick_visible(locator, prefer_last=True)
        if node is not None:
            return node
    return None


def find_gemini_stop_button(page: Page) -> Optional[Locator]:
    candidates = [
        page.get_by_role("button", name=re.compile(r"stop generating|停止生成|stop", re.I)),
        page.locator("button[aria-label*='Stop']"),
        page.locator("button[aria-label*='停止']"),
        page.locator("button:has-text('Stop')"),
    ]
    for locator in candidates:
        node = pick_visible(locator, prefer_last=True)
        if node is not None:
            return node
    return None


def clear_and_fill_input(page: Page, input_box: Locator, content: str, site_name: str) -> None:
    """
    同时兼容 textarea 和 contenteditable：
    - textarea/input 直接 fill
    - contenteditable 用 Ctrl+A + Backspace + insert_text
    """
    jitter(f"{site_name} 聚焦输入框前")
    input_box.click(timeout=5000)
    jitter(f"{site_name} 聚焦输入框后")

    is_contenteditable = bool(input_box.evaluate("el => el.isContentEditable"))
    tag_name = input_box.evaluate("el => el.tagName.toLowerCase()")

    if tag_name in ("textarea", "input") and not is_contenteditable:
        input_box.fill(content, timeout=15000)
    else:
        page.keyboard.press("Control+A")
        page.keyboard.press("Backspace")
        page.keyboard.insert_text(content)

    jitter(f"{site_name} 输入完成后")


def send_message(page: Page, site_name: str, content: str) -> None:
    """发送消息：填充输入框后优先点击发送按钮，失败时 Enter 兜底。"""
    text = normalize_text(content)
    if not text:
        raise ValueError(f"{site_name} 待发送内容为空")

    # 避免超长文本触发 UI 性能问题
    max_chars = 12000
    if len(text) > max_chars:
        warn(f"{site_name} 文本长度 {len(text)} 超过 {max_chars}，已截断")
        text = text[:max_chars]

    input_box = ensure_input_ready(page, site_name)

    log(f"正在发送给 {site_name} ...")
    clear_and_fill_input(page, input_box, text, site_name)

    send_button = find_chatgpt_send_button(page) if site_name == "ChatGPT" else find_gemini_send_button(page)
    jitter(f"{site_name} 发送前")
    if safe_is_enabled(send_button):
        send_button.click(timeout=8000)
        jitter(f"{site_name} 点击发送后")
    else:
        page.keyboard.press("Enter")
        jitter(f"{site_name} Enter 发送后")


def count_chatgpt_assistant_messages(page: Page) -> int:
    # 兼容 ChatGPT DOM 变更：有时 assistant 节点不一定是 div（可能是 article/section 等）。
    # 这里优先用属性选择器，避免被标签类型卡住。
    return page.locator("[data-message-author-role='assistant']").count()


_GEMINI_REPLY_SELECTORS: list[str] = [
    # Prefer explicit model reply containers; avoid broad "message-content" which can include sidebars/nav.
    "main model-response",
    "model-response",
    "main [data-response-id]",
    "main div[data-response-id]",
]

_GEMINI_UI_NOISE_PAT = re.compile(
    r"(关于 Gemini|Gemini 应用|在新窗口中打开|订阅|企业应用场景|认识 Gemini|你的私人 AI 助理|"
    r"写作|计划|研究|学习|工具|快速|历史记录|新对话|发现)",
    re.I,
)


def _clean_gemini_candidate(text: str) -> str:
    t = normalize_text(text)
    if not t:
        return ""
    # Some layouts prepend speaker labels.
    t = re.sub(r"^\s*Gemini\s*说\s*[:：]?\s*", "", t, flags=re.I)
    # Prefer explicitly wrapped public payload if present.
    m = re.search(r"(?is)\[\[\s*PUBLIC_REPLY\s*\]\]\s*(.*?)\s*\[\[\s*/\s*PUBLIC_REPLY\s*\]\]", t, re.I)
    if m:
        return normalize_text(m.group(1))
    return t


def _looks_like_gemini_ui_noise(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return True
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return True
    if len(lines) == 1 and len(lines[0]) <= 20 and _GEMINI_UI_NOISE_PAT.search(lines[0]):
        return True
    hit = sum(1 for ln in lines if _GEMINI_UI_NOISE_PAT.search(ln))
    if hit >= max(2, len(lines) // 2):
        return True
    if "|" in t and hit >= 2:
        return True
    key_len = len(re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", t.lower()))
    if key_len < 6 and hit >= 1:
        return True
    return False


def count_gemini_responses(page: Page) -> int:
    """
    Gemini 回答节点统计：优先计算“看起来像真实回答”的节点数。
    """
    best = 0
    for sel in _GEMINI_REPLY_SELECTORS:
        try:
            loc = page.locator(sel)
            cnt = loc.count()
        except Exception:
            continue
        if cnt <= 0:
            continue
        good = 0
        start = max(0, cnt - 24)
        for i in range(start, cnt):
            try:
                txt = _clean_gemini_candidate(loc.nth(i).inner_text())
            except Exception:
                continue
            if txt and not _looks_like_gemini_ui_noise(txt):
                good += 1
        best = max(best, good if good > 0 else cnt)
    return best


def extract_chatgpt_last_reply(page: Page) -> str:
    """
    按要求读取最后一条 ChatGPT assistant 回复。
    """
    replies = page.locator("[data-message-author-role='assistant']")
    n = replies.count()
    if n <= 0:
        return ""
    return normalize_text(replies.nth(n - 1).inner_text())


def extract_gemini_last_reply(page: Page) -> str:
    """
    Gemini 最后一条回复提取：
    逐个候选结构逆序扫描，拿到最后一个非空文本。
    """
    best = ""
    for sel in _GEMINI_REPLY_SELECTORS:
        loc = page.locator(sel)
        count = loc.count()
        if count <= 0:
            continue
        for i in range(count - 1, -1, -1):
            node = loc.nth(i)
            try:
                if not node.is_visible(timeout=300):
                    continue
                text = _clean_gemini_candidate(node.inner_text())
                if not text:
                    continue
                if _looks_like_gemini_ui_noise(text):
                    continue
                # Prefer substantial semantic blocks first.
                if len(text) >= 24:
                    return text
                if len(text) > len(best):
                    best = text
            except Exception:
                continue
    return best


def read_stable_text(reader: Callable[[], str], site_name: str, rounds: int = 12) -> str:
    """
    文本稳定性检查（防止流式输出中途截断）：
    1) 读取一次文本 A
    2) sleep(1) 后读取文本 B
    3) 如果 len(A) == len(B) 且非空，认为已稳定
    4) 不稳定则继续最多 rounds 次
    """
    log(f"正在读取 {site_name} 回复...")
    text_a = normalize_text(reader())
    for i in range(rounds):
        time.sleep(1)
        text_b = normalize_text(reader())
        log(f"{site_name} 稳定性检查 {i + 1}/{rounds}: len {len(text_a)} -> {len(text_b)}")
        if text_b and len(text_a) == len(text_b):
            return text_b
        text_a = text_b
    warn(f"{site_name} 在限定轮次内仍可能输出中，返回最后一次结果")
    return text_a


def wait_chatgpt_generation_done(page: Page, previous_count: int, timeout_s: int = MAX_WAIT_SECONDS) -> None:
    """
    ChatGPT 生成结束判断（核心逻辑）：
    - 仅靠 wait_for_timeout 不可靠，因此通过 UI 状态组合判定：
      1) 如果出现 Stop 按钮，说明仍在生成
      2) 观察到 Stop 消失 + Send 恢复可见，并且 assistant 消息数增加
      3) 连续命中多次（stable_hits）才判定结束，防止按钮抖动误判
    - 特例：如果出现 Continue generating，会自动点击并继续等待
    """
    log("等待 ChatGPT 生成完成（Stop/Send 状态机）...")
    begin = time.time()
    started = False
    stable_hits = 0
    last_len: Optional[int] = None
    len_stable_hits = 0
    last_status_log = 0.0

    # 兜底：有些情况下 assistant 数量计数不稳定（虚拟列表/DOM 变化），
    # 因此额外记录“发送前最后一条 assistant 文本”，用“文本变化”判断是否出现新回复。
    prev_last_text = normalize_text(extract_chatgpt_last_reply(page))

    while time.time() - begin < timeout_s:
        continue_btn = find_chatgpt_continue_button(page)
        if safe_is_enabled(continue_btn):
            warn("检测到 Continue generating，自动点击继续")
            jitter("点击 Continue generating 前")
            continue_btn.click(timeout=8000)
            jitter("点击 Continue generating 后")
            started = True
            stable_hits = 0
            time.sleep(POLL_SECONDS)
            continue

        current_count = count_chatgpt_assistant_messages(page)
        stop_btn = find_chatgpt_stop_button(page)
        send_btn = find_chatgpt_send_button(page)
        stop_visible = stop_btn is not None
        send_visible = send_btn is not None
        send_enabled = safe_is_enabled(send_btn)

        # 新回复判定：assistant 数量增加 或 最后一条 assistant 文本发生变化（且非空）
        last_text = normalize_text(extract_chatgpt_last_reply(page))
        has_new_assistant = (current_count > previous_count) or (last_text and last_text != prev_last_text)

        if stop_visible or has_new_assistant:
            started = True

        # 关键：不能强依赖 Send 按钮是否可见。
        # 现实中 ChatGPT 的输入区偶尔会因为 UI 卡顿/重渲染暂时不可见（你截图里就是这种情况），
        # 这会导致 send_visible=False，从而旧逻辑永远无法累计 stable_hits，最终“卡死”在等待生成结束。
        if stop_visible:
            stable_hits = 0
        elif started and has_new_assistant:
            stable_hits += 1
        else:
            stable_hits = 0

        # 文本长度稳定性：避免 Stop 消失的瞬间误判（尤其在 UI 抖动/流式输出末尾）
        if has_new_assistant and last_text:
            cur_len = len(last_text)
            if last_len is not None and cur_len == last_len:
                len_stable_hits += 1
            else:
                len_stable_hits = 0
            last_len = cur_len

        # 定期打点输出状态，避免用户误以为卡死
        now = time.time()
        if now - last_status_log > 5:
            log(
                "ChatGPT 状态: "
                f"assistant_count={current_count} prev={previous_count} "
                f"has_new_assistant={has_new_assistant} "
                f"stop_visible={stop_visible} send_visible={send_visible} send_enabled={send_enabled} "
                f"stable_hits={stable_hits} len_stable_hits={len_stable_hits}"
            )
            last_status_log = now

        # 完成条件（组合判断）：
        # 1) 必须已经开始（出现 stop 或出现新 assistant 迹象）
        # 2) stop 不可见
        # 3) 已确认出现新 assistant（数量增长或文本变化）
        # 4) 最后一条文本长度连续稳定 2 次以上，避免还在流式追加
        if started and (not stop_visible) and has_new_assistant and len_stable_hits >= 2 and stable_hits >= 2:
            log("ChatGPT 判定生成完成")
            return

        time.sleep(POLL_SECONDS)

    raise TimeoutError("等待 ChatGPT 生成超时")


def wait_gemini_generation_done(page: Page, previous_count: int, timeout_s: int = MAX_WAIT_SECONDS) -> None:
    """
    Gemini 生成结束判断（核心逻辑）：
    - 观察 Stop 按钮与 Send 状态变化
    - 只有在“回答条数增长”后，才允许进入完成判定
    - 使用 stable_hits 连续命中避免瞬态误判
    """
    log("等待 Gemini 生成完成（Stop/Send 状态机）...")
    begin = time.time()
    started = False
    stable_hits = 0

    while time.time() - begin < timeout_s:
        current_count = count_gemini_responses(page)
        stop_btn = find_gemini_stop_button(page)
        send_btn = find_gemini_send_button(page)

        stop_visible = stop_btn is not None
        send_visible = send_btn is not None

        if current_count > previous_count:
            started = True

        if stop_visible:
            started = True
            stable_hits = 0
        elif started and current_count > previous_count and (send_visible or not stop_visible):
            stable_hits += 1
        else:
            stable_hits = 0

        if started and stable_hits >= 3:
            log("Gemini 判定生成完成")
            return

        time.sleep(POLL_SECONDS)

    raise TimeoutError("等待 Gemini 生成超时")


def run_duel_loop() -> None:
    user_data_dir = str(Path("./user_data").resolve())
    Path(user_data_dir).mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=False,
            viewport={"width": 1920, "height": 1080},
            args=["--disable-blink-features=AutomationControlled"],
        )
        context.set_default_timeout(15000)

        page_chatgpt = context.pages[0] if context.pages else context.new_page()
        page_gemini = context.new_page()

        log("打开 ChatGPT 页面...")
        page_chatgpt.goto(CHATGPT_URL, wait_until="domcontentloaded")
        log("打开 Gemini 页面...")
        page_gemini.goto(GEMINI_URL, wait_until="domcontentloaded")

        # 阶段一：人工登录等待
        page_chatgpt.bring_to_front()
        wait_for_login(page_chatgpt, "ChatGPT")
        page_gemini.bring_to_front()
        wait_for_login(page_gemini, "Gemini")

        # 第一条消息由群主输入（不再自动发送固定 seed）
        seed_message = ""
        while not seed_message:
            try:
                seed_message = input("请输入群主第一句话（回车提交）: ").strip()
            except EOFError:
                seed_message = ""
            if not seed_message:
                warn("第一句话不能为空，请重新输入。")

        log(f"群主初始化发送 -> ChatGPT: {seed_message}")
        chatgpt_prev = count_chatgpt_assistant_messages(page_chatgpt)
        send_message(page_chatgpt, "ChatGPT", seed_message)

        round_index = 1
        while True:
            log(f"========== 回合 {round_index} ==========")

            # A: 等待并读取 ChatGPT
            wait_chatgpt_generation_done(page_chatgpt, chatgpt_prev)
            chatgpt_reply = read_stable_text(lambda: extract_chatgpt_last_reply(page_chatgpt), "ChatGPT")
            if not chatgpt_reply:
                warn("ChatGPT 回复为空，使用兜底语句继续")
                chatgpt_reply = "请继续你的论述。"

            # A -> B
            gemini_prev = count_gemini_responses(page_gemini)
            log("正在发送给 Gemini...")
            send_message(page_gemini, "Gemini", chatgpt_reply)

            # B: 等待并读取 Gemini
            wait_gemini_generation_done(page_gemini, gemini_prev)
            gemini_reply = read_stable_text(lambda: extract_gemini_last_reply(page_gemini), "Gemini")
            if not gemini_reply:
                warn("Gemini 回复为空，使用兜底语句继续")
                gemini_reply = "我没有接收到你的完整回复，请重述一次并更具体。"

            # B -> A
            chatgpt_prev = count_chatgpt_assistant_messages(page_chatgpt)
            log("正在发送给 ChatGPT...")
            send_message(page_chatgpt, "ChatGPT", gemini_reply)

            round_index += 1


def main() -> None:
    try:
        disable_windows_console_quickedit()
        run_duel_loop()
    except KeyboardInterrupt:
        print("\n[EXIT] 收到 Ctrl+C，脚本已停止。")
    except PlaywrightTimeoutError as exc:
        print(f"[ERROR] Playwright 超时: {exc}")
    except Exception as exc:
        print(f"[ERROR] 脚本异常: {exc}")


if __name__ == "__main__":
    main()
