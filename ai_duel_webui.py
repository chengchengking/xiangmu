"""
一个轻量的本地 Web UI，用来把 ChatGPT + Gemini 的“互怼”对话汇总到一个页面里，
并允许你作为第三位成员插话（群主）。

设计原则：
- 不引入额外依赖（仅使用标准库 + playwright + 现有 ai_duel.py 逻辑）
- UI 通过轮询 /api/messages 获取新消息；通过 /api/send 发送你的消息
- 底层仍然是 Playwright 操控网页端（因此 chatgpt.com / gemini.google.com 仍会打开，但你不需要盯着它们）
"""

from __future__ import annotations

import base64
import ast
import atexit
import hashlib
import json
import os
import queue
import re
from difflib import SequenceMatcher
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
    DoubaoAdapter,
    GeminiAdapter,
    GenericWebChatAdapter,
    ModelAdapter,
    ModelMeta,
    QwenAdapter,
    default_avatar_svg,
)
from orchestrator import (
    Mode,
    TurnContext,
    TurnErrorType,
    TurnResult,
    apply_turn_guards,
    error_like,
    is_valid_ack_token,
    is_valid_packet_hash_token,
    map_turn_error_to_reject_reason,
)
from orchestrator.evidence import has_valid_evidence_hook, should_enter_evidence
from orchestrator.receipt import Receipt, ReceiptStatus, RejectReason
from protocol import parse_envelope

SHADOW_SCOPE_DEFAULT = (os.environ.get("AI_DUEL_SHADOW_SCOPE_DEFAULT") or "shared").strip().lower()
if SHADOW_SCOPE_DEFAULT not in {"shared", "local"}:
    SHADOW_SCOPE_DEFAULT = "shared"


def _env_int(name: str, default: int, *, min_v: int = 0, max_v: int = 86400) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except Exception:
        return default
    if v < min_v:
        return min_v
    if v > max_v:
        return max_v
    return v


def _protocol_wrap_instruction_suffix(
    *,
    mode: str,
    hidden_reply_hint: bool,
    turn_id: Optional[int] = None,
    reply_to: str = "",
    ack_in: str = "",
    packet_hash: str = "",
) -> str:
    # Keep this header short/stable to reduce prompt echo on web UIs.
    private_line = (
        "[[PRIVATE_REPLY]]可选：仅写不公开的想法[[/PRIVATE_REPLY]]\n"
        if hidden_reply_hint
        else "[[PRIVATE_REPLY]]可选[[/PRIVATE_REPLY]]\n"
    )
    return (
        "\n\n请按以下格式输出（不要复述本提示）：\n"
        "[[META]]\n"
        f"{(f'turn_id={turn_id}\n') if turn_id is not None else ''}"
        f"mode={mode}\n"
        f"{(f'reply_to={reply_to}\n') if reply_to else ''}"
        f"{(f'ack_in={ack_in}\n') if ack_in else ''}"
        f"{(f'packet_hash={packet_hash}\n') if packet_hash else ''}"
        "[[/META]]\n"
        "[[PUBLIC_REPLY]]\n[[/PUBLIC_REPLY]]\n"
        f"{private_line}"
    )


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_MESSAGES = 1500
MAX_CONTEXT_MESSAGES = 120
MAX_MODEL_PROMPT_CHARS = 12000
MODEL_REPLY_TIMEOUT_S = 180
MODEL_REPLY_TIMEOUT_OVERRIDES: dict[str, int] = {
    "gemini": 90,
    "qwen": 42,
}
TURN_TRACE_ENABLED = os.environ.get("AI_DUEL_TURN_TRACE", "1").strip().lower() not in {"0", "false", "off", "no"}
GROUP_CONTINUOUS_MAX_ROUNDS = 60
GROUP_DEFAULT_ROUNDS = -1  # -1 means continuous rounds until manual stop/safety cap
FOCUS_RECOVERY_ROUNDS = 4
TOPIC_LOCK_MAX_ATTEMPTS_PER_MODEL = 2
TOPIC_INTERJECT_PENDING_KEEP = _env_int("AI_DUEL_TOPIC_INTERJECT_PENDING_KEEP", 2, min_v=2, max_v=80)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
TRACE_DIR = Path(".tmp")
MESSAGE_LOG_FILE = Path(
    os.environ.get("AI_DUEL_MSG_LOG_FILE", str(TRACE_DIR / f"ai_group_messages_{RUN_ID}.jsonl"))
)
TURN_LOG_FILE = Path(
    os.environ.get("AI_DUEL_TURN_LOG_FILE", str(TRACE_DIR / f"ai_group_turns_{RUN_ID}.jsonl"))
)
WORKER_STATE_LOG_FILE = Path(
    os.environ.get("AI_DUEL_WORKER_STATE_LOG_FILE", str(TRACE_DIR / f"ai_worker_states_{RUN_ID}.jsonl"))
)
SELECTED_MODELS_FILE = Path(os.environ.get("AI_DUEL_SELECTED_MODELS_FILE", str(TRACE_DIR / "selected_models.json")))
_JSONL_LOCK = threading.Lock()
GROUP_PUBLIC_TURN_SOFT_TIMEOUT_S = {
    "qwen": _env_int("AI_DUEL_GROUP_TIMEOUT_SOFT_QWEN", 30, min_v=12, max_v=240),
    "doubao": _env_int("AI_DUEL_GROUP_TIMEOUT_SOFT_DOUBAO", 32, min_v=12, max_v=240),
    "default": _env_int("AI_DUEL_GROUP_TIMEOUT_SOFT_DEFAULT", 26, min_v=10, max_v=180),
}
GROUP_PUBLIC_ROUND_CAP_BY_COUNT = {
    2: _env_int("AI_DUEL_GROUP_TIMEOUT_CAP_2", 34, min_v=12, max_v=240),
    3: _env_int("AI_DUEL_GROUP_TIMEOUT_CAP_3", 38, min_v=12, max_v=260),
    4: _env_int("AI_DUEL_GROUP_TIMEOUT_CAP_4", 42, min_v=12, max_v=280),
    5: _env_int("AI_DUEL_GROUP_TIMEOUT_CAP_5", 46, min_v=12, max_v=320),
}
GROUP_TIMEOUT_GROWTH_ROUND_STEP = _env_int("AI_DUEL_GROUP_TIMEOUT_GROWTH_ROUND_STEP", 3, min_v=1, max_v=20)
GROUP_TIMEOUT_GROWTH_PER_STEP_S = _env_int("AI_DUEL_GROUP_TIMEOUT_GROWTH_PER_STEP_S", 3, min_v=1, max_v=30)
GROUP_TIMEOUT_GROWTH_MAX_ROUND_BONUS_S = _env_int(
    "AI_DUEL_GROUP_TIMEOUT_GROWTH_MAX_ROUND_BONUS_S", 28, min_v=0, max_v=180
)
GROUP_TIMEOUT_CONTEXT_UNIT_CHARS = _env_int("AI_DUEL_GROUP_TIMEOUT_CONTEXT_UNIT_CHARS", 900, min_v=200, max_v=5000)
GROUP_TIMEOUT_CONTEXT_PER_UNIT_S = _env_int("AI_DUEL_GROUP_TIMEOUT_CONTEXT_PER_UNIT_S", 2, min_v=0, max_v=20)
GROUP_TIMEOUT_CONTEXT_MAX_BONUS_S = _env_int("AI_DUEL_GROUP_TIMEOUT_CONTEXT_MAX_BONUS_S", 20, min_v=0, max_v=180)
GROUP_TIMEOUT_DYNAMIC_MAX_S = _env_int("AI_DUEL_GROUP_TIMEOUT_DYNAMIC_MAX_S", 120, min_v=20, max_v=360)
GROUP_TIMEOUT_CONTEXT_WINDOW_MSGS = _env_int("AI_DUEL_GROUP_TIMEOUT_CONTEXT_WINDOW_MSGS", 20, min_v=6, max_v=100)
GROUP_TIMEOUT_CONTEXT_PER_MSG_CHAR_CAP = _env_int(
    "AI_DUEL_GROUP_TIMEOUT_CONTEXT_PER_MSG_CHAR_CAP", 420, min_v=80, max_v=4000
)
GROUP_PUBLIC_REPLY_MAX_CHARS = _env_int("AI_DUEL_GROUP_PUBLIC_REPLY_MAX_CHARS", 420, min_v=120, max_v=2000)
GROUP_PUBLIC_REPLY_MAX_LINES = _env_int("AI_DUEL_GROUP_PUBLIC_REPLY_MAX_LINES", 8, min_v=2, max_v=24)
GROUP_PUBLIC_REPLY_MAX_SENTENCES = _env_int("AI_DUEL_GROUP_PUBLIC_REPLY_MAX_SENTENCES", 6, min_v=2, max_v=16)
GROUP_BROADCAST_MAX_MSGS = _env_int("AI_DUEL_GROUP_BROADCAST_MAX_MSGS", 20, min_v=6, max_v=120)
GROUP_BROADCAST_MAX_CHARS = _env_int("AI_DUEL_GROUP_BROADCAST_MAX_CHARS", 3600, min_v=1000, max_v=12000)
GROUP_BROADCAST_PER_MSG_CHAR_CAP = _env_int("AI_DUEL_GROUP_BROADCAST_PER_MSG_CHAR_CAP", 320, min_v=100, max_v=2400)
GROUP_BROADCAST_RECENT_MAX_MSGS = _env_int("AI_DUEL_GROUP_BROADCAST_RECENT_MAX_MSGS", 12, min_v=4, max_v=80)
GROUP_BROADCAST_RECENT_MAX_CHARS = _env_int("AI_DUEL_GROUP_BROADCAST_RECENT_MAX_CHARS", 2200, min_v=600, max_v=12000)
CONTEXT_MODEL_DIGEST_MAX_CHARS = _env_int("AI_DUEL_CONTEXT_MODEL_DIGEST_MAX_CHARS", 560, min_v=240, max_v=2400)
CONTEXT_MODEL_DIGEST_MAX_LINES = _env_int("AI_DUEL_CONTEXT_MODEL_DIGEST_MAX_LINES", 10, min_v=4, max_v=28)
CONTEXT_USER_DIGEST_MAX_CHARS = _env_int("AI_DUEL_CONTEXT_USER_DIGEST_MAX_CHARS", 420, min_v=180, max_v=1800)
CONTEXT_USER_DIGEST_MAX_LINES = _env_int("AI_DUEL_CONTEXT_USER_DIGEST_MAX_LINES", 8, min_v=3, max_v=24)


def _group_round_timeout_cap_s(model_count: int, *, round_no: int = 1, context_chars: int = 0) -> int:
    if model_count <= 2:
        base = int(GROUP_PUBLIC_ROUND_CAP_BY_COUNT[2])
    elif model_count == 3:
        base = int(GROUP_PUBLIC_ROUND_CAP_BY_COUNT[3])
    elif model_count == 4:
        base = int(GROUP_PUBLIC_ROUND_CAP_BY_COUNT[4])
    else:
        base = int(GROUP_PUBLIC_ROUND_CAP_BY_COUNT[5])

    rn = max(1, int(round_no))
    round_steps = max(0, rn - 1) // max(1, int(GROUP_TIMEOUT_GROWTH_ROUND_STEP))
    round_bonus = min(
        int(GROUP_TIMEOUT_GROWTH_MAX_ROUND_BONUS_S),
        round_steps * int(GROUP_TIMEOUT_GROWTH_PER_STEP_S),
    )

    ctx = max(0, int(context_chars))
    if GROUP_TIMEOUT_CONTEXT_PER_UNIT_S <= 0:
        context_bonus = 0
    else:
        context_units = ctx // max(1, int(GROUP_TIMEOUT_CONTEXT_UNIT_CHARS))
        context_bonus = min(
            int(GROUP_TIMEOUT_CONTEXT_MAX_BONUS_S),
            context_units * int(GROUP_TIMEOUT_CONTEXT_PER_UNIT_S),
        )

    cap = base + round_bonus + context_bonus
    cap = min(int(GROUP_TIMEOUT_DYNAMIC_MAX_S), cap)
    return max(base, int(cap))


def _topic_lock_rounds(model_count: int) -> int:
    # Keep topic lock short: enough to force a visible reaction, but not so long that
    # normal chat flow gets over-constrained and appears "stuck".
    return max(1, min(3, int(model_count)))

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
- 语气像正常人聊天：自然、有温度，避免空话套话。
- 讨论要有“立场 + 依据 + 推论”：先给结论，再给关键理由。
- 可以结构化表达（短段落/少量要点均可），不要机械模板化复读。
- 允许联网：仅在需要“最新事实/具体数据/可核验信息”时联网；联网只补事实，不把检索结果直接当结论。
- 尽量给出依据：可使用数字、区间、时间点、对比、边界条件（至少 1 个）。
- 如存在不确定性，要明确写出假设与风险，而不是回避表态。
- 字数不要卡死：常规建议 120-360 字；复杂议题可到 500 字，但保持聚焦。
""".strip()

# 每轮轻量提醒：避免模型“跑偏”回长文/检索式回答
RULES_REMINDER = "【提醒】像真实辩论：先结论，后依据（可含数据/假设/边界）；可适度展开，但别跑题。"

# 单条转发上限：只做安全截断，不做摘要拼装/历史重打包。
FORWARD_MAX_CHARS = 3200
INCREMENTAL_MSG_MAX_LINES = 20
INCREMENTAL_MSG_MAX_CHARS = 3200

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
    if meta.key == "doubao":
        return DoubaoAdapter(meta)
    if meta.key == "qwen":
        return QwenAdapter(meta)
    return GenericWebChatAdapter(meta)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False)
        with _JSONL_LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # Logging must never block chat pipeline.
        pass


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
    shadow_scope: str = "shared"  # shadow: shared | local


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
    runtime_state: str = "INIT"  # INIT|IDLE|GENERATING|STALE|CAPTCHA_BLOCKED|NEEDS_HUMAN|COOLING_DOWN|DEAD
    runtime_reason: str = ""
    runtime_fail_count: int = 0
    runtime_cooldown_until: float = 0.0
    runtime_updated_ts: str = ""


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
        self._load_selected_models()

        # 后端动作队列（Playwright 只能在同一线程执行）
        self.inbox: "queue.Queue[dict[str, Any]]" = queue.Queue()

    def _load_selected_models(self) -> None:
        try:
            if not SELECTED_MODELS_FILE.exists():
                return
            data = json.loads(SELECTED_MODELS_FILE.read_text(encoding="utf-8", errors="replace"))
            selected = data.get("selected_keys") if isinstance(data, dict) else None
            if not isinstance(selected, list):
                return
            wanted = {(str(x or "").strip().lower()) for x in selected if str(x or "").strip()}
            for k, m in self._models.items():
                if not m.integrated:
                    continue
                m.selected = k in wanted
        except Exception:
            return

    def _save_selected_models_locked(self) -> None:
        try:
            SELECTED_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
            selected = [k for k, m in self._models.items() if m.selected]
            selected.sort(key=lambda k: self._models[k].slot if k in self._models else 9999)
            payload = {"selected_keys": selected}
            SELECTED_MODELS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            return

    # --- messages ------------------------------------------------------

    def add_message(
        self,
        role: str,
        speaker: str,
        text: str,
        *,
        visibility: str,
        model_key: Optional[str] = None,
        shadow_scope: str = "shared",
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
                    shadow_scope=(shadow_scope or "shared"),
                )
            )
            if len(self._messages) > MAX_MESSAGES:
                self._messages = self._messages[-MAX_MESSAGES:]
            _append_jsonl(
                MESSAGE_LOG_FILE,
                {
                    "run_id": RUN_ID,
                    "id": mid,
                    "ts": self._messages[-1].ts,
                    "role": role,
                    "speaker": speaker,
                    "visibility": visibility,
                    "model_key": model_key,
                    "shadow_scope": (shadow_scope or "shared"),
                    "text": t,
                },
            )
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

    def set_model_runtime_state(
        self,
        key: str,
        new_state: str,
        *,
        reason: str = "",
        fail_delta: int = 0,
        reset_fail: bool = False,
        cooldown_s: float = 0.0,
        turn_id: int = 0,
    ) -> None:
        key = (key or "").strip().lower()
        with self._lock:
            m = self._models.get(key)
            if not m:
                return
            old_state = m.runtime_state or "INIT"
            if reset_fail:
                m.runtime_fail_count = 0
            if fail_delta:
                m.runtime_fail_count = max(0, int(m.runtime_fail_count) + int(fail_delta))
            now = time.time()
            if cooldown_s and float(cooldown_s) > 0:
                m.runtime_cooldown_until = max(float(m.runtime_cooldown_until or 0.0), now + float(cooldown_s))
            elif new_state != "COOLING_DOWN":
                m.runtime_cooldown_until = 0.0
            m.runtime_state = str(new_state or m.runtime_state or "INIT")
            m.runtime_reason = core.normalize_text(reason)
            m.runtime_updated_ts = _now_iso()
            _append_jsonl(
                WORKER_STATE_LOG_FILE,
                {
                    "run_id": RUN_ID,
                    "ts": m.runtime_updated_ts,
                    "model_key": key,
                    "model_name": m.name,
                    "old_state": old_state,
                    "new_state": m.runtime_state,
                    "reason": m.runtime_reason,
                    "fail_count": int(m.runtime_fail_count),
                    "cooldown_until": float(m.runtime_cooldown_until or 0.0),
                    "turn_id": int(turn_id or 0),
                },
            )

    def get_model_runtime_state(self, key: str) -> dict[str, Any]:
        key = (key or "").strip().lower()
        with self._lock:
            m = self._models.get(key)
            if not m:
                return {"ok": False, "error": "unknown_model"}
            now = time.time()
            remain = max(0.0, float(m.runtime_cooldown_until or 0.0) - now)
            return {
                "ok": True,
                "key": m.key,
                "state": m.runtime_state,
                "reason": m.runtime_reason,
                "fail_count": int(m.runtime_fail_count),
                "cooldown_remaining_s": round(remain, 2),
                "updated_ts": m.runtime_updated_ts,
            }

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
                self._save_selected_models_locked()
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
            self._save_selected_models_locked()
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
                self._save_selected_models_locked()
            return True

    # --- status / stop -------------------------------------------------

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            runtimes = {
                k: {
                    "state": m.runtime_state,
                    "reason": m.runtime_reason,
                    "fail_count": int(m.runtime_fail_count),
                    "cooldown_remaining_s": round(max(0.0, float(m.runtime_cooldown_until or 0.0) - time.time()), 2),
                    "updated_ts": m.runtime_updated_ts,
                }
                for k, m in self._models.items()
            }
            return {"ok": True, "status": self._status, "stop": self._stop, "worker_runtime": runtimes}

    def request_stop(self) -> None:
        with self._lock:
            self._stop = True

    def clear_stop(self) -> None:
        with self._lock:
            self._stop = False

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
      .recoverBtn {
        position: absolute;
        right: 6px;
        top: 6px;
        width: 22px;
        height: 22px;
        border-radius: 9px;
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(127,29,29,0.30);
        color: rgba(255,255,255,0.96);
        display: none;
        place-items: center;
        font-size: 12px;
        cursor: pointer;
        user-select: none;
      }
      .recoverBtn:hover { background: rgba(127,29,29,0.45); }
      .modelBtn.needs-human .recoverBtn,
      .modelBtn.cooling .recoverBtn,
      .modelBtn.stale .recoverBtn { display: grid; }
      .runtimePill {
        position: absolute;
        left: 50%;
        bottom: -8px;
        transform: translateX(-50%);
        min-width: 26px;
        height: 14px;
        padding: 0 4px;
        border-radius: 999px;
        display: none;
        align-items: center;
        justify-content: center;
        font-size: 9px;
        font-weight: 900;
        color: rgba(255,255,255,0.98);
        border: 1px solid rgba(0,0,0,0.18);
        background: rgba(30,41,59,0.78);
        letter-spacing: 0.1px;
      }
      .runtimePill.show { display: flex; }
      .runtimePill.idle { background: rgba(22,163,74,0.88); }
      .runtimePill.generating { background: rgba(59,130,246,0.90); }
      .runtimePill.cooling { background: rgba(234,88,12,0.92); }
      .runtimePill.needs-human { background: rgba(220,38,38,0.92); }
      .runtimePill.dead { background: rgba(71,85,105,0.94); }
      .runtimePill.stale { background: rgba(168,85,247,0.92); }
      .runtimePill.init { background: rgba(14,116,144,0.90); }
      .modelBtn.needs-human { box-shadow: 0 0 0 3px rgba(239,68,68,0.16), 0 12px 22px rgba(13,43,64,0.20); }
      .modelBtn.cooling { box-shadow: 0 0 0 3px rgba(249,115,22,0.16), 0 12px 22px rgba(13,43,64,0.20); }
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
            <div>- 过长会被截断，建议聚焦“立场 + 依据 + 数据/假设”，不用刻意压到很短。</div>
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
        let stopArmUntil = 0;
        let isComposing = false;

        const authMask = $('authMask');
        const authTitle = $('authTitle');
        const authHelp = $('authHelp');
        const authClose = $('authClose');
        const authOpen = $('authOpen');
        const authCheck = $('authCheck');

        const st = {
          models: [],
          workerRuntime: {},
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

        const getRuntimeInfo = (key) => {
          const wr = st.workerRuntime || {};
          return wr[String(key || '').toLowerCase()] || null;
        };

        const runtimeCssClass = (stateName) => {
          const s = String(stateName || '').toUpperCase();
          if (!s) return '';
          if (s === 'IDLE') return 'idle';
          if (s === 'GENERATING') return 'generating';
          if (s === 'COOLING_DOWN') return 'cooling';
          if (s === 'NEEDS_HUMAN' || s === 'CAPTCHA_BLOCKED') return 'needs-human';
          if (s === 'DEAD') return 'dead';
          if (s === 'STALE') return 'stale';
          return 'init';
        };

        const runtimeShortLabel = (stateName, runtime) => {
          const s = String(stateName || '').toUpperCase();
          if (!s) return '';
          if (s === 'IDLE') return 'OK';
          if (s === 'GENERATING') return 'RUN';
          if (s === 'COOLING_DOWN') {
            const remain = Number(runtime && runtime.cooldown_remaining_s || 0);
            if (Number.isFinite(remain) && remain > 0) return 'CD' + String(Math.min(99, Math.round(remain)));
            return 'CD';
          }
          if (s === 'NEEDS_HUMAN') return 'HUM';
          if (s === 'CAPTCHA_BLOCKED') return 'CAP';
          if (s === 'DEAD') return 'DEAD';
          if (s === 'STALE') return 'STL';
          return 'INIT';
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
            const rt = getRuntimeInfo(m.key);
            const rtState = String((rt && rt.state) || '').toUpperCase();
            const rtCls = runtimeCssClass(rtState);
            const btn = document.createElement('button');
            btn.className = 'modelBtn' + (m.integrated ? (m.selected ? ' on' : ' off') : ' disabled');
            if (m.authenticated) btn.classList.add('authed');
            if (st.viewTarget === m.key) btn.classList.add('active');
            if (rtCls) btn.classList.add(rtCls);

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

            const recover = document.createElement('div');
            recover.className = 'recoverBtn';
            recover.textContent = '↻';
            recover.title = '恢复/探活';
            recover.addEventListener('click', async (ev) => {
              ev.stopPropagation();
              const resp = await apiPost('/api/models/recover', { key: m.key });
              await pollState();
              await pollModels();
              rebuildTargets();
              if (resp && resp.ok && resp.needs_human) openAuth(m.key);
            });

            const rtPill = document.createElement('div');
            rtPill.className = 'runtimePill';
            const rtLabel = runtimeShortLabel(rtState, rt);
            if (rtLabel) {
              rtPill.classList.add('show');
              if (rtCls) rtPill.classList.add(rtCls);
              rtPill.textContent = rtLabel;
            }

            btn.appendChild(num);
            btn.appendChild(authDot);
            btn.appendChild(lock);
            btn.appendChild(nudge);
            btn.appendChild(recover);
            btn.appendChild(rtPill);

            const rtTitle = rt
              ? ('\\n状态：' + String(rt.state || 'UNKNOWN') + (rt.reason ? ('\\n原因：' + rt.reason) : '')
                  + ((rt.cooldown_remaining_s || 0) > 0 ? ('\\n冷却剩余：' + String(rt.cooldown_remaining_s) + 's') : '')
                  + ('\\n失败计数：' + String(rt.fail_count || 0)))
              : '';
            btn.title = m.integrated
              ? (m.name + '\\n点击：加入/退出群聊\\n绿点：已登录\\n💬：让TA发言\\n↻：恢复/探活' + rtTitle)
              : (m.name + '\\n未接入（锁定）');

            btn.addEventListener('click', async () => {
              if (!m.integrated) return;
              // 一键切换：后端会自动做登录态检查。
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
          const hideRouteMentions = (raw) => {
            let t = String(raw || '');
            if (msg.role !== 'model' || msg.visibility !== 'public') return t;
            // Hide routing mentions in UI while keeping backend routing logic.
            t = t.replace(/(^|\\n)\\s*@[-_0-9A-Za-z\\u4e00-\\u9fff]{1,24}\\s*(?=\\n|$)/g, '$1');
            t = t.replace(/\\s+@[-_0-9A-Za-z\\u4e00-\\u9fff]{1,24}\\b/g, '');
            t = t.replace(/\\n{3,}/g, '\\n\\n');
            t = t.trim();
            return t || String(raw || '');
          };
          text.innerHTML = esc(hideRouteMentions(msg.text || ''));

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
          st.workerRuntime = res.worker_runtime || {};
          renderModels();
        };

        const toUtf8Base64 = (s) => {
          try {
            const bytes = new TextEncoder().encode(String(s || ''));
            let bin = '';
            const chunk = 0x8000;
            for (let i = 0; i < bytes.length; i += chunk) {
              bin += String.fromCharCode(...bytes.slice(i, i + chunk));
            }
            return btoa(bin);
          } catch (_) {
            try {
              return btoa(unescape(encodeURIComponent(String(s || ''))));
            } catch (_) {
              return '';
            }
          }
        };
        const toUriText = (s) => {
          try {
            return encodeURIComponent(String(s || ''));
          } catch (_) {
            return '';
          }
        };
        const textKeyLen = (s) => {
          try {
            return String(s || '').replace(/[^0-9A-Za-z\u4e00-\u9fff]+/g, '').length;
          } catch (_) {
            return 0;
          }
        };
        const looksGarbled = (s) => {
          const t = String(s || '');
          if (!t) return false;
          const q = (t.match(/[?？]/g) || []).length;
          const ratio = q / Math.max(1, t.length);
          return ratio >= 0.35 && textKeyLen(t) < 6;
        };

        const sendNow = async () => {
          const v = sendTarget.value;
          if (!v || v === '__sep__') {
            st.viewTarget = '';
            updateComposerEnabled();
            return;
          }
          const original = String(input.value || '');
          if (!original.trim()) return;
          let text = original;
          // IME edge case: Enter may arrive right before composition commits.
          // Re-read once shortly, then reject obvious garbled payload.
          if (looksGarbled(text)) {
            await new Promise((r) => setTimeout(r, 120));
            const late = String(input.value || '');
            if (textKeyLen(late) > textKeyLen(text)) text = late;
          }
          if (looksGarbled(text)) {
            hintLeft.textContent = '检测到疑似乱码输入，请重新输入后发送';
            hintLeft.style.color = 'var(--danger)';
            return;
          }
          const payload = { target: v, text, text_b64: toUtf8Base64(text), text_uri: toUriText(text) };
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
          const now = Date.now();
          if (now > stopArmUntil) {
            stopArmUntil = now + 1800;
            hintLeft.textContent = '再点一次“停止轮聊”以确认';
            hintLeft.style.color = 'var(--warn)';
            return;
          }
          stopArmUntil = 0;
          await apiPost('/api/debate/stop', {});
          hintLeft.textContent = '已请求停止自动轮聊';
          hintLeft.style.color = 'var(--warn)';
        });
        clearBtn && clearBtn.addEventListener('click', () => {
          input.value = '';
          input.focus();
        });
        sendBtn && sendBtn.addEventListener('click', sendNow);
        input && input.addEventListener('compositionstart', () => { isComposing = true; });
        input && input.addEventListener('compositionend', () => { isComposing = false; });
        input && input.addEventListener('keydown', (ev) => {
          if (isComposing || ev.isComposing) return;
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
          const enc = (() => {
            try {
              const bytes = new TextEncoder().encode(text);
              let bin = '';
              for (let i = 0; i < bytes.length; i += 0x8000) {
                bin += String.fromCharCode(...bytes.slice(i, i + 0x8000));
              }
              return btoa(bin);
            } catch (_) {
              try {
                return btoa(unescape(encodeURIComponent(text)));
              } catch (_) {
                return '';
              }
            }
          })();
          let textUri = '';
          try {
            textUri = encodeURIComponent(text);
          } catch (_) {
            textUri = '';
          }
          await apiPost('/api/send', { text, text_b64: enc, text_uri: textUri, to });
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

    def _ensure_worker_running(self) -> None:
        w = getattr(self, "worker", None)
        if w is None:
            return
        try:
            # New user actions should resume the backend loop after /api/stop.
            self.state.clear_stop()
        except Exception:
            pass
        try:
            w.start()
        except Exception:
            pass

    def _sync_login_check(self, key: str, *, timeout_s: float = 45.0) -> dict[str, Any]:
        self._ensure_worker_running()
        ev = threading.Event()
        box: dict[str, Any] = {}
        self.state.inbox.put({"kind": "login_check", "key": key, "_ev": ev, "_reply": box})
        ev.wait(timeout=timeout_s)
        if box:
            return box
        return {"ok": False, "authenticated": False, "error": "login_check_timeout"}

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
        raw = self.rfile.read(length)
        ctype = self.headers.get("Content-Type") or ""
        m = re.search(r"charset\s*=\s*([-\w.]+)", ctype, re.I)
        candidates: list[str] = []
        if m:
            candidates.append(m.group(1).strip().strip("\"'"))
        candidates.extend(["utf-8", "utf-8-sig", "gb18030", "gbk"])
        body = ""
        tried: set[str] = set()
        for enc in candidates:
            e = (enc or "").strip().lower()
            if not e or e in tried:
                continue
            tried.add(e)
            try:
                body = raw.decode(e)
                break
            except Exception:
                continue
        if not body:
            body = raw.decode("utf-8", errors="replace")
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
            key = str(data.get("key") or "").strip().lower()
            m = self.state.get_model(key)
            # 一键启用：若是“未选中且未认证”，先自动检查登录态；
            # 已登录则直接加入，不再要求手动点“打开登录窗口”。
            if m and m.integrated and (not m.selected) and (not m.authenticated):
                self._sync_login_check(key, timeout_s=45.0)
            self._send_json(self.state.toggle_selected(key))
            return

        if parsed.path == "/api/models/login/open":
            self._ensure_worker_running()
            key = str(data.get("key") or "")
            ev = threading.Event()
            box: dict[str, Any] = {}
            self.state.inbox.put({"kind": "login_open", "key": key, "_ev": ev, "_reply": box})
            ev.wait(timeout=25)
            self._send_json(box or {"ok": True})
            return

        if parsed.path == "/api/models/login/check":
            key = str(data.get("key") or "")
            box = self._sync_login_check(key, timeout_s=45.0)
            self._send_json(box or {"ok": True, "authenticated": False})
            return

        if parsed.path == "/api/models/nudge":
            self._ensure_worker_running()
            key = str(data.get("key") or "")
            self.state.inbox.put({"kind": "nudge", "key": key})
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/models/recover":
            self._ensure_worker_running()
            key = str(data.get("key") or "")
            ev = threading.Event()
            box: dict[str, Any] = {}
            self.state.inbox.put({"kind": "recover_model", "key": key, "_ev": ev, "_reply": box})
            ev.wait(timeout=45)
            self._send_json(box or {"ok": False, "error": "recover_timeout"})
            return

        if parsed.path == "/api/send":
            self._ensure_worker_running()
            text_plain = core.normalize_text(str(data.get("text") or ""))
            text = text_plain
            text_b64 = core.normalize_text(str(data.get("text_b64") or ""))
            text_uri = core.normalize_text(str(data.get("text_uri") or data.get("text_urlenc") or ""))
            decoded_b64 = ""
            if text_b64:
                try:
                    decoded_b64 = base64.b64decode(text_b64.encode("ascii"), validate=False).decode(
                        "utf-8", errors="replace"
                    )
                    decoded_b64 = core.normalize_text(decoded_b64)
                except Exception:
                    decoded_b64 = ""
            decoded_uri = ""
            if text_uri:
                try:
                    decoded_uri = core.normalize_text(urllib.parse.unquote(text_uri, encoding="utf-8", errors="replace"))
                except Exception:
                    decoded_uri = ""
            text = _pick_best_transport_text(text_plain, decoded_b64, decoded_uri)
            target = str(data.get("target") or data.get("to") or "").strip().lower()
            if target in {"next", "groupchat", "public"}:
                target = "group"
            if not text:
                self._send_json({"ok": False, "error": "empty text"}, status=400)
                return
            if not target:
                self._send_json({"ok": False, "error": "missing target"}, status=400)
                return
            rounds_raw = data.get("rounds", GROUP_DEFAULT_ROUNDS)
            shadow_scope = str(data.get("shadow_scope") or "").strip().lower()
            self.state.inbox.put(
                {
                    "kind": "send",
                    "target": target,
                    "text": text,
                    "rounds": rounds_raw,
                    "shadow_scope": shadow_scope,
                }
            )
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/debate/stop":
            self.state.request_round_stop()
            self.state.add_system("收到停止轮聊请求。")
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
    if len(summary) > 320:
        summary = summary[:320].rstrip() + "…"
    return main, summary


def _clip_text(s: str, max_chars: int) -> str:
    s = core.normalize_text(s)
    if len(s) <= max_chars:
        return s
    head = s[: int(max_chars * 0.7)].rstrip()
    tail = s[-int(max_chars * 0.2) :].lstrip()
    return f"{head}…（略）…{tail}".strip()


def _enqueue_startup_login_refresh(state: SharedState) -> int:
    queued = 0
    for key in state.selected_keys():
        m = state.get_model(key)
        if not m or not m.integrated:
            continue
        state.inbox.put({"kind": "login_check", "key": key})
        queued += 1
    return queued


def _transport_text_score(text: str) -> float:
    t = core.normalize_text(text)
    if not t:
        return -1e9
    key = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", t.lower())
    score = float(len(key))
    q_cnt = t.count("?")
    repl_cnt = t.count("�")
    q_ratio = (q_cnt / max(1, len(t)))
    if q_ratio >= 0.30:
        score -= 180.0
    elif q_ratio >= 0.15:
        score -= 80.0
    elif q_ratio >= 0.08:
        score -= 30.0
    score -= float(repl_cnt * 22)
    cjk_cnt = len(re.findall(r"[\u4e00-\u9fff]", t))
    if cjk_cnt:
        score += min(64.0, float(cjk_cnt) * 0.65)
    if re.search(r"[A-Za-z]{3,}", t):
        score += 10.0
    if re.search(r"\d", t):
        score += 6.0
    if len(key) < 4:
        score -= 18.0
    if re.search(r"(?:%[0-9a-f]{2}){3,}", t, re.I):
        score -= 160.0
    # Common UTF-8<->GBK mojibake fingerprints; strongly down-rank these payloads.
    weird_hits = len(re.findall(r"[浣鍦鍙鍐鍑锛銆鏈€闂瑕缇缁鏂鏃闄鎯璇鎴寮鎺纭]", t))
    cjk_total = len(re.findall(r"[\u4e00-\u9fff]", t))
    if cjk_total >= 8 and weird_hits >= 4:
        if (weird_hits / max(1, cjk_total)) >= 0.16:
            score -= 260.0
    if re.search(r"(?:浣犲|鏈€|闂棰|璇峰|鍙洖|濡傛殏|缇や富)", t):
        score -= 140.0
    return score


def _pick_best_transport_text(*candidates: str) -> str:
    normed: list[str] = []
    best = ""
    best_score = -1e9
    first_non_empty = ""
    for cand in candidates:
        txt = core.normalize_text(cand)
        normed.append(txt)
        if txt and not first_non_empty:
            first_non_empty = txt
        sc = _transport_text_score(txt)
        if sc > best_score:
            best_score = sc
            best = txt

    picked = core.normalize_text(best or first_non_empty)
    # Safety net: if the picked text is mostly "?" but another candidate contains
    # meaningful letters/CJK, prefer the meaningful candidate.
    if picked:
        q_like = picked.count("?") + picked.count("？")
        if q_like >= max(4, int(len(picked) * 0.35)):
            for cand in normed:
                if not cand or cand == picked:
                    continue
                key_len = len(re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", cand.lower()))
                if key_len >= 6:
                    cand_q = cand.count("?") + cand.count("？")
                    if cand_q <= max(1, int(len(cand) * 0.12)):
                        picked = cand
                        break
    # If picked payload still looks mojibake-like, prefer any cleaner candidate.
    if picked:
        picked_score = _transport_text_score(picked)
        for cand in normed:
            if not cand or cand == picked:
                continue
            cand_score = _transport_text_score(cand)
            if cand_score >= (picked_score + 80):
                picked = cand
                picked_score = cand_score
    return core.normalize_text(picked)


_TOPIC_ZH_STOPWORDS = {
    "我们",
    "你们",
    "大家",
    "这个",
    "那个",
    "然后",
    "如果",
    "因为",
    "所以",
    "就是",
    "以及",
    "需要",
    "可以",
    "应该",
    "但是",
    "不过",
    "还是",
    "一下",
    "一个",
    "一些",
    "可能",
    "开始",
    "继续",
    "讨论",
    "回复",
}

_TOPIC_EN_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "that",
    "this",
    "is",
    "are",
    "be",
    "as",
    "we",
    "you",
}


def _topic_terms(text: str, *, max_terms: int = 40) -> set[str]:
    t = core.normalize_text(text).lower()
    if not t:
        return set()
    out: list[str] = []
    seen: set[str] = set()

    for en in re.findall(r"[a-z]{3,}", t):
        if en in _TOPIC_EN_STOPWORDS:
            continue
        if en not in seen:
            seen.add(en)
            out.append(en)
        if len(out) >= max_terms:
            return set(out)

    for zh in re.findall(r"[\u4e00-\u9fff]{2,24}", t):
        chunks: list[str]
        if len(zh) <= 6:
            chunks = [zh]
        else:
            chunks = [zh[i : i + 2] for i in range(0, len(zh) - 1)]
        for c in chunks:
            if len(c) < 2:
                continue
            if c in _TOPIC_ZH_STOPWORDS:
                continue
            if c not in seen:
                seen.add(c)
                out.append(c)
            if len(out) >= max_terms:
                return set(out)
    return set(out)


def _topic_overlap_score(reply: str, topic: str) -> float:
    rep = core.normalize_text(reply).lower()
    top = core.normalize_text(topic).lower()
    if not top:
        return 1.0
    if not rep:
        return 0.0

    rt = _topic_terms(rep)
    tt = _topic_terms(top)
    score_term = 0.0
    if tt:
        score_term = len(rt & tt) / max(1, len(tt))

    r_chars = set(re.findall(r"[0-9a-z\u4e00-\u9fff]", rep))
    t_chars = set(re.findall(r"[0-9a-z\u4e00-\u9fff]", top))
    score_char = 0.0
    if t_chars:
        score_char = len(r_chars & t_chars) / max(1, len(t_chars))
    return max(score_term, score_char * 0.65)


def _required_topic_anchors(topic: str) -> set[str]:
    """
    对仓库/群聊调试类问题，抽取一组“必须至少命中一个”的领域锚点，
    防止模型只抓到 Repo/框架等泛词后跑偏到网络/TCP等无关方向。
    """
    t = core.normalize_text(topic).lower()
    if not t:
        return set()
    anchors: set[str] = set()
    pairs = [
        ("parser", ["parser", "parse", "解析", "提取", "抽取", "爬取", "抓取"]),
        ("broadcast", ["broadcast", "广播", "packet_hash", "一致性"]),
        ("topic_switch", ["topic-switch", "topic switch", "切题", "话题切换", "插话", "interject", "interrupt"]),
        ("ui", ["ui", "webui", "窗口", "界面"]),
        ("log", ["log", "日志"]),
        (
            "runtime",
            [
                "timeout",
                "超时",
                "captcha",
                "验证码",
                "风控",
                "recover",
                "恢复",
                "reload",
                "worker",
                "state",
                "状态机",
                "熔断",
                "cooldown",
                "冷却",
            ],
        ),
        (
            "quality",
            [
                "parser",
                "broadcast",
                "topic",
                "reply",
                "回复",
                "抓取",
                "解析",
                "日志",
                "公平",
                "fairness",
                "重复",
                "刷屏",
                "维护",
                "架构",
                "扩展",
            ],
        ),
    ]
    for _k, kws in pairs:
        if any(kw in t for kw in kws):
            anchors.update(kws)
    return anchors


def _is_repo_debug_topic(topic: str) -> bool:
    t = core.normalize_text(topic).lower()
    if not t:
        return False
    if "github.com/" in t or "repo:" in t or "仓库" in t or "xiangmu" in t:
        return True
    return bool(
        re.search(
            r"(ai\s+groupchat|群聊产品|webui|parser|broadcast|packet_hash|topic[- ]?switch|插话|日志|captcha|验证码)",
            t,
            re.I,
        )
    )


def _reply_hits_any_anchor(reply: str, anchors: set[str]) -> bool:
    if not anchors:
        return True
    r = core.normalize_text(reply).lower()
    if not r:
        return False
    return any(a and a in r for a in anchors)


def _looks_redundant_model_repeat(reply: str, previous_public_texts: list[str]) -> bool:
    cur = core.normalize_text(reply)
    if not cur:
        return False
    cur_key = _line_dedupe_key(cur)
    if len(cur_key) < 20:
        return False
    prevs = [core.normalize_text(x) for x in (previous_public_texts or []) if core.normalize_text(x)]
    for p in prevs[-3:]:
        pk = _line_dedupe_key(p)
        if not pk:
            continue
        if cur_key == pk:
            return True
        # High containment = same point repeated with minor wording changes.
        if len(cur_key) >= 24 and (cur_key in pk or pk in cur_key):
            return True
        # Simple overlap ratio on dedupe keys.
        a = set(re.findall(r"[0-9a-z\u4e00-\u9fff]", cur_key))
        b = set(re.findall(r"[0-9a-z\u4e00-\u9fff]", pk))
        if a and b:
            ov = len(a & b) / max(1, len(a | b))
            if ov >= 0.88:
                return True
    return False


def _looks_like_clarify_reply(text: str) -> bool:
    t = core.normalize_text(text)
    if not t:
        return False
    if not re.search(r"[?？]$", t):
        return False
    return bool(
        re.search(r"(再|请|具体|说明|澄清|哪个|哪一个|如何|what|which|clarify|specify|details?)", t, re.I)
    )


def _looks_provider_error_reply(text: str) -> bool:
    t = core.normalize_text(text)
    if not t:
        return False
    # Web UI transient/server error banners that should not enter group dialogue.
    if re.search(r"something went wrong", t, re.I) and re.search(r"help\.openai\.com", t, re.I):
        return True
    if re.search(r"(internal server error|service unavailable|temporarily unavailable)", t, re.I):
        return True
    if re.search(r"(网络错误|请求失败|服务暂时不可用|服务器错误)", t):
        return True
    if re.search(r"(重试|retry)$", t, re.I) and len(_line_dedupe_key(t)) <= 120:
        return True
    return False


def _extract_expected_short_literal(topic: str) -> str:
    """
    从用户题目里提取“只输出某个固定短词”的期望值。
    例：
    - output only BLUE
    - answer only 105
    - 仅输出 BLUE
    - 只回复 105
    """
    t = core.normalize_text(topic)
    if not t:
        return ""
    english_placeholder_tokens = {"this", "that", "it", "one", "number", "result", "answer"}
    patterns = [
        re.compile(r"(?is)\b(?:output|answer|reply)\s+only\s+([A-Za-z0-9_-]{1,24})\b"),
        re.compile(r"(?is)\b(?:output|answer|reply)\s+(?:exactly\s+)?([A-Za-z0-9_-]{1,24})\s+only\b"),
        re.compile(r"(?is)\bonly\s+(?:output|answer|reply)\s+([A-Za-z0-9_-]{1,24})\b"),
        re.compile(r"(?is)(?:仅输出|只输出|仅回复|只回复)\s*([A-Za-z0-9_-]{1,24})"),
    ]
    for pat in patterns:
        m = pat.search(t)
        if not m:
            continue
        token = core.normalize_text(m.group(1) or "")
        token = re.sub(r"^[`'\"“”‘’\[\](){}<>]+|[`'\"“”‘’\[\](){}<>]+$", "", token)
        token_l = token.lower()
        if token_l in english_placeholder_tokens:
            continue
        # "answer only this: ..." should not lock to the word "this".
        tail = t[m.end() : m.end() + 2]
        if token_l.isalpha() and re.match(r"\s*[:：]", tail or ""):
            continue
        if token:
            return token
    return ""


def _should_pass_no_public_tagged(envelope_status: str, envelope_public: str) -> bool:
    status = core.normalize_text(envelope_status).upper()
    pub = core.normalize_text(envelope_public)
    return (not pub) and status != "NO_TAGS"


def _should_wait_for_login_recheck(runtime_state: str, runtime_reason: str) -> bool:
    st = core.normalize_text(runtime_state).upper()
    rs = core.normalize_text(runtime_reason).lower()
    if st in {"INITIALIZING", "AUTH_CHECKING"}:
        return True
    if "login_check" in rs and st in {"INIT", "INITIALIZING", "IDLE"}:
        return True
    return False


def _extract_expected_numeric_answer(topic: str) -> Optional[float]:
    """
    从题目里提取算式的期望数值结果。
    支持括号与多步运算；失败时退回旧的简单二元算式提取。
    """
    t = core.normalize_text(topic)
    if not t:
        return None
    # Normalize common operator variants first.
    t_norm = (
        t.replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("（", "(")
        .replace("）", ")")
    )

    def _safe_eval(expr: str) -> Optional[float]:
        expr2 = (expr or "").strip()
        if not expr2:
            return None
        if len(expr2) > 120:
            return None
        if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr2):
            return None
        # Avoid malformed repeated operators that often come from noisy matches.
        if re.search(r"[\+\*\/]{2,}", expr2):
            return None
        try:
            tree = ast.parse(expr2, mode="eval")
        except Exception:
            return None

        def _ev(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return _ev(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return float(node.value)
                raise ValueError("bad const")
            if isinstance(node, ast.Num):  # py<3.8 compat node shape
                return float(node.n)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                v = _ev(node.operand)
                return v if isinstance(node.op, ast.UAdd) else -v
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                a = _ev(node.left)
                b = _ev(node.right)
                if isinstance(node.op, ast.Add):
                    return a + b
                if isinstance(node.op, ast.Sub):
                    return a - b
                if isinstance(node.op, ast.Mult):
                    return a * b
                if abs(b) < 1e-12:
                    raise ZeroDivisionError()
                return a / b
            raise ValueError("bad node")

        try:
            v = _ev(tree)
        except Exception:
            return None
        if abs(v) > 1e12:
            return None
        return float(v)

    # Try longest arithmetic-looking fragment first (supports "(73*89) - (47*31) + 125").
    cands = re.findall(r"[\d\.\s\+\-\*\/\(\)]{7,}", t_norm)
    cands = sorted({c.strip() for c in cands if c and re.search(r"\d", c)}, key=len, reverse=True)
    for expr in cands:
        # Require at least 2 numbers and 2 operators for "full expression" path.
        if len(re.findall(r"\d+(?:\.\d+)?", expr)) < 2:
            continue
        if len(re.findall(r"[\+\-\*\/]", expr)) < 2:
            continue
        v = _safe_eval(expr)
        if v is not None:
            return v

    # Fallback: simple binary operation.
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-−*/xX×÷])\s*(-?\d+(?:\.\d+)?)", t)
    if not m:
        return None
    try:
        a = float(m.group(1))
        op = m.group(2)
        b = float(m.group(3))
    except Exception:
        return None

    if op in {"*", "x", "X", "×"}:
        return a * b
    if op in {"+", "＋"}:
        return a + b
    if op in {"-", "−"}:
        return a - b
    if op in {"/", "÷"}:
        if abs(b) < 1e-12:
            return None
        return a / b
    return None


def _extract_expected_numeric_literal_if_requested(topic: str) -> str:
    t = core.normalize_text(topic)
    if not t:
        return ""
    if not re.search(r"(number\s+only|digits?\s+only|只(?:给|回|输)?出?数字|仅(?:给|回|输)?出?数字)", t, re.I):
        return ""
    val = _extract_expected_numeric_answer(t)
    if val is None:
        return ""
    if abs(val - int(val)) <= 1e-9:
        return str(int(val))
    return str(val)


def _strict_topic_consensus_threshold(active_model_count: int) -> int:
    """
    严格短答案题（如“只输出 323”）达到少量模型一致后即可视为收敛，
    否则会长时间占住后续真实讨论话题。
    """
    n = max(1, int(active_model_count or 0))
    # 2 模型时双方都答对；>=3 模型时收到任意 3 个一致短答即可收敛。
    return 2 if n <= 2 else 3


def _reply_contains_expected_number(reply: str, expected: float) -> bool:
    rep = core.normalize_text(reply)
    if not rep:
        return False
    nums = re.findall(r"-?\d+(?:\.\d+)?", rep)
    if not nums:
        return False
    exp_is_int = abs(expected - round(expected)) < 1e-9
    tol = 1e-6 if not exp_is_int else 1e-9
    exp_i = int(round(expected)) if exp_is_int else None
    for s in nums:
        try:
            v = float(s)
        except Exception:
            continue
        if exp_is_int:
            if abs(v - expected) <= tol:
                return True
            try:
                if int(round(v)) == exp_i and abs(v - round(v)) < 1e-9:
                    return True
            except Exception:
                pass
        else:
            if abs(v - expected) <= 1e-3:
                return True
    return False


def _is_reply_aligned_with_user_topic(reply: str, user_text: str, *, strict: bool = False) -> bool:
    rep = core.normalize_text(reply)
    top = core.normalize_text(user_text)
    if not top:
        return True
    if not rep:
        return False
    if _TRIVIAL_PUBLIC_PAT.match(rep):
        return False
    if _LOW_VALUE_PROCESS_PAT.match(rep):
        return False
    if _QWEN_STATUS_ONLY_PAT.match(rep):
        return False
    expect_token = _extract_expected_short_literal(top)
    if expect_token:
        rep_plain = core.normalize_text(re.sub(r"[。.!！?？,，:：;；\s]+$", "", rep))
        if rep_plain.lower() == expect_token.lower():
            return True
        if strict:
            return False
    expected_numeric = _extract_expected_numeric_answer(top)
    if expected_numeric is not None and _reply_contains_expected_number(rep, expected_numeric):
        return True
    if _looks_like_clarify_reply(rep):
        return True
    score = _topic_overlap_score(rep, top)
    rep_terms = _topic_terms(rep, max_terms=40)
    top_terms = _topic_terms(top, max_terms=34)
    term_overlap = len(rep_terms & top_terms)
    top_digits = set(re.findall(r"\d+(?:\.\d+)?", top))
    rep_digits = set(re.findall(r"\d+(?:\.\d+)?", rep))
    required_anchors = _required_topic_anchors(top)
    # Hard off-topic guard: when the host switched topic, old "idiom/festival chatter"
    # should not survive if overlap is very low.
    if _IDIOM_STYLE_CHATTER_PAT.search(rep) and not _IDIOM_STYLE_CHATTER_PAT.search(top):
        if score < 0.18:
            return False
    if strict:
        if _GROUP_HOST_CHATTER_PAT.search(rep) and len(_line_dedupe_key(rep)) <= 72:
            return False
        if _GROUP_ORCHESTRATION_PAT.search(rep) and len(_line_dedupe_key(rep)) <= 72:
            return False
        if _looks_prompt_leak_reply(rep):
            return False
        if required_anchors and not _reply_hits_any_anchor(rep, required_anchors):
            # Repo/debug prompts often accept diverse but still relevant engineering fixes
            # (timeout/captcha/retry/fairness/worker-state/recover/logging) that may not
            # overlap with the exact anchor wording in the host message.
            if not (
                _is_repo_debug_topic(top)
                and re.search(
                    r"(parser|parse|解析|提取|抓取|broadcast|广播|packet_hash|topic|切题|插话|interject|"
                    r"log|日志|timeout|超时|captcha|验证码|风控|recover|恢复|reload|worker|state|状态|"
                    r"cooldown|冷却|retry|重试|fair|公平|重复|刷屏|维护|架构|扩展|selector|stale)",
                    rep,
                    re.I,
                )
            ):
                return False
        if expected_numeric is not None:
            # 算式题：若没给出正确数值，视为未对齐新话题。
            return False
        # Numeric topic: avoid hard-rejecting valid cross-language responses.
        # If host topic includes numbers, prefer "has any numeric grounding" first.
        if top_digits and not _is_repo_debug_topic(top):
            if not rep_digits:
                return False
            if top_digits & rep_digits:
                return True
            # No direct shared number, but still allow if semantic overlap is acceptable.
            if score >= 0.12 and term_overlap >= 1:
                return True
            if score >= 0.18:
                return True
            return False
        if score >= 0.12 and term_overlap >= 1:
            return True
        if term_overlap >= 2 and score >= 0.08:
            return True
        if not top_terms and score >= 0.18:
            return True
        # Cross-language or paraphrased replies can have low lexical overlap.
        # Accept unless it clearly looks like stale/group-chatter boilerplate.
        if len(_line_dedupe_key(rep)) >= 10:
            return True
        return False
    if score >= 0.14:
        return True
    if score >= 0.08 and len(_line_dedupe_key(rep)) >= 14:
        return True
    # Keep this gate conservative: only block clear stale/process chatter.
    if _GROUP_HOST_CHATTER_PAT.search(rep) and len(_line_dedupe_key(rep)) <= 72:
        return False
    if _GROUP_ORCHESTRATION_PAT.search(rep) and len(_line_dedupe_key(rep)) <= 72:
        return False
    if _looks_prompt_leak_reply(rep):
        return False
    return True


_GROUP_HOST_CHATTER_PAT = re.compile(
    r"(?:^|[，,。!！~～\s])"
    r"(?:好(?:的|嘞)?|收到|明白|行(?:吧)?|ok|OK|哈喽|hello)"
    r"(?:[，,。!！~～\s]{0,3})"
    r"(?:群主|主持人|老大|老板)",
    re.I,
)
_GROUP_ORCHESTRATION_PAT = re.compile(
    r"(谁先来|我们这就开始|先来(?:出)?第一个|我先来抛砖引玉|接龙(?:开始|走起)|继续接龙|下一位谁来)",
    re.I,
)
_IDIOM_STYLE_CHATTER_PAT = re.compile(
    r"(成语|接龙|尾字|下一位|谁来接|我接|祝大家|马年|福气|好运)",
    re.I,
)
_LOW_VALUE_PROCESS_PAT = re.compile(
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
    r"开始成语接龙游戏|启动成语接龙游戏|接龙游戏正酣.*|"
    r"我需要[:：].*|让我(?:构思|想一下|数一下|再确认)|检查字数|数一下|好，就这个|"
    r"(?:现在)?我需要.{0,120}(?:然后|再)(?:回应|补充|给出|讨论)|"
    r"基于当前信息提出预测与假设|先给区间和假设|首先[，,]先给区间和假设|"
    r"根据(?:群主)?最新话题.*(?:给出|回应|讨论)|给出一个简洁.*核心假设|"
    r"给出.*人民币.*黄金.*价格区间.*假设|"
    r"思考问题的逻辑结构|确认计算结果无误|寻找(?:符合|满足)条件的三位数|"
    r"请把公开发言放在.*之间|如需隐藏想法.*之间|"
    r"(?:\d{1,3}\s*字(?:左右)?|字数)(?:[，,、\s]{0,4})(?:符合要求|可|即可)|"
    r"符合要求(?:即可)?|"
    r"按要求(?:输出|回复)|"
    r"先结论(?:[，,、\\s\\-]*\\d+(?:-\\d+)?\\s*条?依据)?.*(?:群聊发言|不要复述规则)|"
    r"结论在前.*(?:依据|群聊发言)|"
    r"先回应用户，再补充你对上一位的看法|先回应群主话题|如暂不发言，仅回复\s*\[?\s*PASS\s*\]?)\s*[。.!！~～]*\s*$",
    re.I,
)


def _trace_turn(model_key: str, stage: str, text: str, *, elapsed_s: Optional[float] = None) -> None:
    body_console = _clip_text(text, 220).replace("\n", " | ").strip()
    if not body_console:
        body_console = "∅"
    # Keep richer prompt/reply snapshot in jsonl for postmortem debugging.
    body_json = _clip_text(text, 2400).strip()
    if not body_json:
        body_json = "∅"
    tail = f", {elapsed_s:.2f}s" if elapsed_s is not None else ""
    if TURN_TRACE_ENABLED:
        core.log(f"[TURN][{model_key}] {stage}{tail}: {body_console}")
    _append_jsonl(
        TURN_LOG_FILE,
        {
            "run_id": RUN_ID,
            "ts": _now_iso(),
            "model_key": model_key,
            "stage": stage,
            "elapsed_s": round(float(elapsed_s), 3) if elapsed_s is not None else None,
            "text": body_json,
        },
    )


def _strip_group_chatter_boilerplate(text: str) -> str:
    t = core.normalize_text(text)
    if not t:
        return ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return t

    kept: list[str] = []
    for ln in lines:
        key_len = len(_line_dedupe_key(ln))
        if key_len <= 56 and _GROUP_HOST_CHATTER_PAT.search(ln):
            continue
        if key_len <= 56 and _GROUP_ORCHESTRATION_PAT.search(ln):
            continue
        if re.match(r"^(?:@?[\w\u4e00-\u9fff-]{1,20}\s*)?(?:接|你接|请接|来接)\s*[~～!！。\.]*$", ln):
            continue
        kept.append(ln)

    if not kept:
        return t
    return core.normalize_text("\n".join(kept))


_PROMPT_ECHO_LINE_PAT = re.compile(
    r"(你在多人群聊中发言|群主最新话题|下面是群聊广播窗口|下面是你还没处理的群聊新消息|"
    r"先回应群主最新话题|直接说你在群里要发的话|最终发言要求|可回应对象|"
    r"不要输出思考过程|只输出群里的正文|优先级：必须先回应|如你这轮暂不发言)",
    re.I,
)
_PROMPT_ECHO_MSG_LINE_PAT = re.compile(r"^\[\d+\]\s*(?:群主|ChatGPT|Gemini|DeepSeek|豆包|Qwen)\s*:", re.I)


def _strip_instruction_echo_lines(text: str) -> str:
    """
    清除模型把提示词/广播窗口原样复述到回答里的噪声行。
    """
    t = core.normalize_text(text)
    if not t:
        return ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return t
    kept: list[str] = []
    for ln in lines:
        if _PROMPT_ECHO_LINE_PAT.search(ln):
            continue
        if _PROMPT_ECHO_MSG_LINE_PAT.search(ln):
            continue
        if re.search(r"^(?:先结论|结论在前).*(?:依据|条依据).*(?:群聊发言|不要复述规则)", ln):
            continue
        if re.match(r"^(?:你说|你在说|请把|如需|如果你想|不想发言)\b", ln):
            continue
        kept.append(ln)
    return core.normalize_text("\n".join(kept))


def _strip_private_thoughts(text: str) -> str:
    """
    过滤明显“思考/内心独白”段，避免在模型之间传播。
    仅用于转发与跨模型提示，不影响 UI 上原始公开回复展示。
    """
    t = core.normalize_text(text)
    if not t:
        return ""

    # Common explicit thought blocks.
    t = re.sub(r"(?is)<\s*think\s*>.*?<\s*/\s*think\s*>", "", t)
    t = re.sub(r"(?is)</?\s*think\s*>", "", t)
    t = re.sub(r"(?is)```(?:thinking|analysis|reasoning|thought).*?```", "", t)

    thought_head = re.compile(
        r"^\s*(思考(?:中)?|已思考|思路|推理过程|内心独白|thinking|analysis|reasoning|thoughts?)\b",
        re.I,
    )
    thought_timer = re.compile(r"^\s*思考[（(]用时.+?[)）]\s*$", re.I)

    out: list[str] = []
    skipping = False
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            if skipping:
                skipping = False
                continue
            out.append("")
            continue
        if thought_timer.match(s):
            continue
        if thought_head.match(s):
            skipping = True
            continue
        if re.search(r"(让我构思|让我再确认|让我数一下|检查字数|字数符合要求|好，就这个)", s):
            continue
        if re.match(r"^\s*我需要\s*[:：].{0,64}$", s):
            continue
        if skipping:
            continue
        out.append(ln)

    return core.normalize_text("\n".join(out))


_PUBLIC_HEAD_PAT = re.compile(
    r"^\s*(?:【|\[)?(?:对外|公开|外显|发言|public|output|out)(?:】|\])?\s*[:：]?\s*$",
    re.I,
)
_PRIVATE_HEAD_PAT = re.compile(
    r"^\s*(?:【|\[)?(?:内心|私下|私有|思考|想法|thought|private|inner|analysis)(?:】|\])?\s*[:：]?\s*$",
    re.I,
)
_EXPLICIT_PUBLIC_PRIVATE_MARK_PAT = re.compile(
    r"(?:"
    r"(?:【|\[)\s*(?:对外|公开|外显|发言|public|output|out|内心|私下|私有|思考|想法|thought|private|inner|analysis)\s*(?:】|\])"
    r"|(?:^|\n)\s*(?:对外|公开|外显|发言|public|output|out|内心|私下|私有|思考|想法|thought|private|inner|analysis)\s*[:：]"
    r")",
    re.I,
)
_INLINE_PUBLIC_PRIVATE_MARK_PAT = re.compile(
    r"(?:"
    r"(?:【|\[)\s*(?P<head_a>对外|公开|外显|发言|public|output|out|内心|私下|私有|思考|想法|thought|private|inner|analysis)\s*(?:】|\])\s*[:：]?"
    r"|(?:^|\n)\s*(?P<head_b>对外|公开|外显|发言|public|output|out|内心|私下|私有|思考|想法|thought|private|inner|analysis)\s*[:：]"
    r")",
    re.I,
)
_TRIVIAL_PUBLIC_PAT = re.compile(r"^\s*(?:已?完成|已经完成|done|ok|好的|收到)\s*[。.!?]?\s*$", re.I)
_QWEN_STATUS_ONLY_PAT = re.compile(
    r"^\s*(?:"
    r"已?完成(?:思考)?|已经完成(?:思考)?|思考|思考中|正在思考|继续思考|生成中|回答中|正在构思|构思中|"
    r"the chat is in progress|in progress|generating|composing"
    r")\s*[。.!！？?]?\s*$",
    re.I,
)
_PUBLIC_WRAP_OPEN = "[[PUBLIC_REPLY]]"
_PUBLIC_WRAP_CLOSE = "[[/PUBLIC_REPLY]]"
_PRIVATE_WRAP_OPEN = "[[PRIVATE_REPLY]]"
_PRIVATE_WRAP_CLOSE = "[[/PRIVATE_REPLY]]"
_PUBLIC_PLACEHOLDER_TOKEN = "<<WRITE_PUBLIC_OR_[PASS]>>"
# Qwen / Doubao are prone to echoing format instructions.
# Keep wrapper parsing support for backward compatibility, but default to no wrapper hinting.
_USE_WRAP_HINT_FOR_CN_MODELS = False
_WRAP_TOKEN_PAT = re.compile(
    r"(?:<<<\s*(?:PUBLIC_REPLY|END_PUBLIC_REPLY|PRIVATE_REPLY|END_PRIVATE_REPLY)\s*>>>"
    r"|\[\[\s*/?\s*(?:PUBLIC_REPLY|PRIVATE_REPLY)\s*\]\])",
    re.I,
)
_WRAP_INSTRUCTION_HINTS = (
    "请把公开发言放在",
    "公开发言放在",
    "如需隐藏想法",
    "你的公开回复",
    "公开回复",
    "不要复述本提示",
    "不要复述提示词",
    "不要复述题目",
)


def _wrapped_public_quality_score(text: str) -> float:
    t = core.normalize_text(text)
    if not t:
        return -1e9
    key_len = len(_line_dedupe_key(t))
    score = float(key_len)
    if key_len < 8:
        score -= 300.0
    if _LOW_VALUE_PROCESS_PAT.match(t):
        score -= 260.0
    if _QWEN_STATUS_ONLY_PAT.match(t):
        score -= 180.0
    if _looks_unfinished_public_reply(t):
        score -= 90.0
    if _looks_prompt_leak_reply(t):
        score -= 120.0
    if any(h in t for h in _WRAP_INSTRUCTION_HINTS):
        score -= 320.0
    if re.search(r"(?:<<<|>>>|PUBLIC_REPLY|END_PUBLIC_REPLY|PRIVATE_REPLY|END_PRIVATE_REPLY)", t, re.I):
        score -= 180.0
    return score


def _extract_wrapped_reply(text: str) -> tuple[str, str]:
    t = core.normalize_text(text)
    if not t:
        return "", ""
    pub = ""
    pri = ""
    # Accept both legacy angle-bracket tags and current bracket tags.
    # Some UIs sanitize "<...>" visually, so bracket tags are more stable.
    pub_patterns = (
        re.compile(r"(?is)<<<\s*PUBLIC_REPLY\s*>>>\s*(.*?)\s*<<<\s*END_PUBLIC_REPLY\s*>>>", re.I),
        re.compile(r"(?is)\[\[\s*PUBLIC_REPLY\s*\]\]\s*(.*?)\s*\[\[\s*/\s*PUBLIC_REPLY\s*\]\]", re.I),
    )
    pri_patterns = (
        re.compile(r"(?is)<<<\s*PRIVATE_REPLY\s*>>>\s*(.*?)\s*<<<\s*END_PRIVATE_REPLY\s*>>>", re.I),
        re.compile(r"(?is)\[\[\s*PRIVATE_REPLY\s*\]\]\s*(.*?)\s*\[\[\s*/\s*PRIVATE_REPLY\s*\]\]", re.I),
    )
    pub_candidates: list[str] = []
    for pat in pub_patterns:
        for m in pat.finditer(t):
            c = core.normalize_text(m.group(1) or "")
            if c:
                pub_candidates.append(c)
    if pub_candidates:
        pub = max(pub_candidates, key=_wrapped_public_quality_score)
    pri_candidates: list[str] = []
    for pat in pri_patterns:
        for m in pat.finditer(t):
            c = core.normalize_text(m.group(1) or "")
            if c:
                pri_candidates.append(c)
    if pri_candidates:
        # Private block is optional; prefer the latest one if multiple appear.
        pri = pri_candidates[-1]
    return pub, pri


def _split_public_private_reply(text: str) -> tuple[str, str]:
    """
    Parse model output into:
    - public: visible/forwardable content
    - private: hidden inner thoughts (not forwarded)
    """
    t = core.normalize_text(text)
    if not t:
        return "", ""

    wrapped_public, wrapped_private = _extract_wrapped_reply(t)
    if wrapped_public or wrapped_private:
        pub = _strip_private_thoughts(wrapped_public) or wrapped_public
        pub = core.normalize_text(_WRAP_TOKEN_PAT.sub("", pub))
        pri = core.normalize_text(_WRAP_TOKEN_PAT.sub("", wrapped_private))
        pub_key_len = len(_line_dedupe_key(pub))
        wrap_placeholder = (
            pub_key_len < 8
            or _LOW_VALUE_PROCESS_PAT.match(pub or "")
            or re.search(
                r"(你的公开回复|公开回复|在此填写|示例|格式|标记|正文|PUBLIC_REPLY|END_PUBLIC_REPLY|"
                r"群主最新话题|最终发言要求|如暂不发言|请把公开发言放在|公开发言放在|如需隐藏想法|只输出群里的正文|只输出给群里的正文)",
                pub or "",
                re.I,
            )
        )
        if pub and not wrap_placeholder:
            return pub, pri

    private_parts: list[str] = []

    def _save_private(m: re.Match[str]) -> str:
        part = core.normalize_text(m.group(1) or "")
        if part:
            private_parts.append(part)
        return ""

    body = re.sub(r"(?is)<\s*think\s*>(.*?)<\s*/\s*think\s*>", _save_private, t)
    body = _WRAP_TOKEN_PAT.sub("", body)
    body = core.normalize_text(body)
    if not body:
        return "", core.normalize_text("\n".join(private_parts))

    # Parse explicit markers like: "【对外】...【内心】..." / "内心: ...".
    inline: list[re.Match[str]] = []
    if _EXPLICIT_PUBLIC_PRIVATE_MARK_PAT.search(body):
        inline = list(_INLINE_PUBLIC_PRIVATE_MARK_PAT.finditer(body))
    if inline:
        pub_seg: list[str] = []
        pri_seg: list[str] = []
        structured_hits = 0
        first_private_pos: Optional[int] = None

        for idx, m in enumerate(inline):
            head = core.normalize_text((m.group("head_a") or m.group("head_b") or "")).lower()
            if not head:
                continue
            start = m.end()
            end = inline[idx + 1].start() if idx + 1 < len(inline) else len(body)
            seg = core.normalize_text(body[start:end])
            if not seg:
                continue
            if head in {"对外", "公开", "外显", "发言", "public", "output", "out"}:
                pub_seg.append(seg)
                structured_hits += 1
            elif head in {"内心", "私下", "私有", "思考", "想法", "thought", "private", "inner", "analysis"}:
                if first_private_pos is None:
                    first_private_pos = m.start()
                pri_seg.append(seg)
                structured_hits += 1

        if structured_hits >= 1 and (pub_seg or pri_seg):
            public = core.normalize_text("\n".join(pub_seg))
            private = core.normalize_text("\n".join(pri_seg + private_parts))
            if not public and first_private_pos is not None:
                public = core.normalize_text(body[:first_private_pos])
            if public:
                return _strip_private_thoughts(public) or public, private
            # no public block: fallback to original body to avoid empty visible reply.
            return _strip_private_thoughts(body) or body, private

    pub_lines: list[str] = []
    pri_lines: list[str] = []
    mode = "public"
    has_struct = False

    for ln in body.splitlines():
        s = ln.strip()
        if not s:
            if mode == "public":
                pub_lines.append("")
            else:
                pri_lines.append("")
            continue
        if _PUBLIC_HEAD_PAT.match(s):
            mode = "public"
            has_struct = True
            continue
        if _PRIVATE_HEAD_PAT.match(s):
            mode = "private"
            has_struct = True
            continue
        if mode == "public":
            pub_lines.append(ln)
        else:
            pri_lines.append(ln)

    public = core.normalize_text("\n".join(pub_lines))
    private = core.normalize_text("\n".join(pri_lines + private_parts))

    # If no explicit structure exists, keep original as public.
    if not has_struct:
        public = body
        # still strip explicit "内心：..." one-liners into private when possible.
        m = re.search(r"(?is)(?:^|\n)\s*(?:内心|思考|private|inner)\s*[:：]\s*(.+)$", body, re.I)
        if m:
            p = core.normalize_text(m.group(1))
            if p:
                private = core.normalize_text("\n".join([private, p]))
            public = core.normalize_text(body[: m.start()])

    public = _strip_private_thoughts(public) or public
    return core.normalize_text(public), core.normalize_text(private)


def _detail_digest(text: str, *, max_chars: int, max_lines: int) -> str:
    """
    细节保留型压缩：
    - 保留首尾句；
    - 优先保留带数字/条件/边界/结论的句子；
    - 控制总长度，避免上下文膨胀。
    """
    t = core.normalize_text(text)
    if not t:
        return ""

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return _clip_text(t, max_chars)
    if len(t) <= max_chars and len(lines) <= max_lines:
        return t

    scored: list[tuple[int, int, str]] = []
    last_idx = len(lines) - 1
    for idx, ln in enumerate(lines):
        score = 0
        if idx == 0:
            score += 4
        if idx == last_idx:
            score += 3
        if re.search(r"\d|%|￥|\$|年|月|日|分钟|小时|ms|MB|GB|km|℃", ln):
            score += 3
        if re.search(r"因为|所以|但是|然而|如果|前提|风险|边界|反例|结论|建议|成本|收益|限制|条件|步骤|方案|假设", ln):
            score += 2
        if len(ln) >= 22:
            score += 1
        scored.append((score, idx, ln))

    keep_idx: set[int] = {0, last_idx}
    for _, idx, _ in sorted(scored, key=lambda x: (-x[0], x[1])):
        keep_idx.add(idx)
        if len(keep_idx) >= max_lines:
            break

    picked = [lines[i] for i in sorted(keep_idx)]
    out: list[str] = []
    used = 0
    for ln in picked:
        extra = len(ln) + (1 if out else 0)
        if used + extra > max_chars:
            break
        out.append(ln)
        used += extra

    if not out:
        return _clip_text("\n".join(lines), max_chars)
    merged = "\n".join(out).strip()
    if len(merged) < len(t) and len(merged) < max_chars:
        if not merged.endswith("…"):
            merged += "\n…"
    return merged


def _strip_leading_status_noise(text: str) -> str:
    t = core.normalize_text(text)
    if not t:
        return ""
    out: list[str] = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            continue
        if _QWEN_STATUS_ONLY_PAT.match(s):
            continue
        s2 = re.sub(
            r"^\s*(?:已?完成(?:思考)?|已经完成(?:思考)?|思考|思考中|正在思考|继续思考|生成中|回答中|正在构思|构思中)\s*(?:[|｜:：\-—>»]+\s*)",
            "",
            s,
            flags=re.I,
        )
        s2 = re.sub(r"^\s*(?:首先写|先写|先答|先说|先回一句)\s*[：:]\s*", "", s2, flags=re.I)
        s2 = core.normalize_text(s2) or s
        if _QWEN_STATUS_ONLY_PAT.match(s2):
            continue
        out.append(s2)
    return core.normalize_text("\n".join(out))


def _format_msg_for_context(msg: UiMessage) -> str:
    # 优先用【转发摘要】；没有时做“细节保留型”压缩，统一三模型上下文质量。
    if msg.role == "model":
        main, summary = _split_forward_summary(msg.text)
        body_src = summary or (main or msg.text)
        body = _detail_digest(
            body_src,
            max_chars=CONTEXT_MODEL_DIGEST_MAX_CHARS,
            max_lines=CONTEXT_MODEL_DIGEST_MAX_LINES,
        )
    else:
        body = _detail_digest(
            msg.text,
            max_chars=CONTEXT_USER_DIGEST_MAX_CHARS,
            max_lines=CONTEXT_USER_DIGEST_MAX_LINES,
        )
    prefix = "系统" if msg.role == "system" else msg.speaker
    return f"{prefix}: {body}".strip()


def _looks_prompt_leak_reply(text: str) -> bool:
    t = core.normalize_text(text)
    if not t:
        return True
    if any(
        tok in t
        for tok in (
            "[[META]]",
            "[[/META]]",
            "【系统回执】",
            "ack_in=",
            "packet_hash=",
            "evidence_hook=",
            "PUBLIC_REPLY",
            "PRIVATE_REPLY",
        )
    ):
        return True
    if re.match(
        r"^\s*(?:正在阅读|阅读中|读取中|正在加载|加载中)"
        r"(?:\s*(?:正在阅读|阅读中|读取中|正在加载|加载中))*\s*$",
        t,
        re.I,
    ):
        return True
    if re.search(r"(按(?:照)?格式(?:来)?|格式已遵守|然后按格式写)\s*(?:META|PUBLIC_REPLY|PRIVATE_REPLY)?", t, re.I):
        return True
    # Task-planning / pre-action lines are not valid public chat replies.
    if re.search(
        r"^(?:查阅|查看|分析|评估|梳理|检查|定位|研究).{0,24}(?:仓库|项目).{0,24}(?:问题|定位|建议|风险|潜在问题)",
        t,
        re.I,
    ):
        return True
    if re.search(r"^(?:检测到|发现).{0,18}(?:潜在).{0,18}(?:问题|风险)[。.]?$", t):
        return True
    if re.search(r"涉嫌违反.+(?:使用规范|社区规范)|若有误判.*不喜欢", t):
        return True
    # Hard prompt/broadcast markers: if any of these appear, treat as leaked prompt text.
    if re.search(
        r"(下面是群聊广播窗口|下面是你还没处理的群聊新消息|群主最新话题：|"
        r"\[\d+\]\s*(?:群主|ChatGPT|Gemini|DeepSeek|Qwen|豆包|Doubao)\s*:)",
        t,
        re.I,
    ):
        return True
    # Gemini home/welcome surface frequently contaminates the first-turn extraction.
    if re.search(r"(需要我为你做些什么|制作图片|创作音乐|创作视频|给我的一天注入活力|随便写点什么)", t):
        if re.search(r"(?:\bPRO\b|Gemini|快速|群主最新话题|下面是群聊广播窗口)", t, re.I):
            return True
    hints = (
        _PUBLIC_WRAP_OPEN,
        _PUBLIC_WRAP_CLOSE,
        _PRIVATE_WRAP_OPEN,
        _PRIVATE_WRAP_CLOSE,
        f"格式：{_PUBLIC_WRAP_OPEN}正文{_PUBLIC_WRAP_CLOSE}",
        "你在多人群聊中发言",
        "你在多人群聊中继续讨论",
        "群主最新话题：",
        "上一位发言（",
        "最终发言要求：",
        "只输出给群里的正文",
        "只输出群里的正文",
        "请把公开发言放在",
        "如需隐藏想法，可放在",
        "如暂不发言，仅回复 [PASS]",
        "只输出 [PASS]",
        "只依据以下群聊消息回复",
        "不要复述本提示",
        "不要复述提示词",
        "[[META]]",
        "[[/META]]",
        "ack_in=",
        "packet_hash=",
        "evidence_hook=",
        "可回应对象：",
        "可点名对象：",
        "上下文边界",
        "忽略网页里更早的旧对话",
        "Gemini 应用",
        "与 Gemini 对话",
        "须遵守《Google 条款》",
        "须遵守《Google 隐私权政策》",
        "Gemini 是一款 AI 工具",
    )
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return True

    bad = 0
    good = 0
    for ln in lines:
        candidate = re.sub(
            r"^\s*(?:已?完成(?:思考)?|已经完成(?:思考)?|思考中|正在思考|生成中|回答中)\s*[|｜:：\-]*\s*",
            "",
            ln,
            flags=re.I,
        )
        candidate = core.normalize_text(candidate) or ln
        k = _line_dedupe_key(candidate)
        is_bad = False
        q_ratio = (candidate.count("?") + candidate.count("？")) / max(1, len(candidate))
        if any(h in ln for h in hints):
            is_bad = True
        if q_ratio >= 0.35 and len(k) < 12:
            is_bad = True
        if _QWEN_STATUS_ONLY_PAT.match(candidate):
            is_bad = True
        if _GROUP_HOST_CHATTER_PAT.search(candidate) and len(k) <= 56:
            is_bad = True
        if _GROUP_ORCHESTRATION_PAT.search(candidate) and len(k) <= 56:
            is_bad = True
        if _LOW_VALUE_PROCESS_PAT.match(candidate):
            is_bad = True
        if is_bad:
            if k and len(k) >= 12 and not _QWEN_STATUS_ONLY_PAT.match(candidate):
                good += 1
            bad += 1
            continue
        if k and len(k) >= 10:
            good += 1

    if good >= 1 and bad < len(lines):
        return False
    return bad >= max(1, len(lines) // 2)


def _line_dedupe_key(line: str) -> str:
    s = core.normalize_text(line).lower()
    if not s:
        return ""
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", s)


def _drop_repeated_line_blocks(lines: list[str], *, max_block: int = 10) -> list[str]:
    if len(lines) < 4:
        return lines
    keys = [_line_dedupe_key(x) for x in lines]
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        block_cap = min(max_block, i, n - i)
        skip = 0
        for block in range(block_cap, 1, -1):
            cur = keys[i : i + block]
            if not all(cur):
                continue
            if sum(len(x) for x in cur) < 16:
                continue
            seen_before = False
            for j in range(0, i - block + 1):
                if keys[j : j + block] == cur:
                    seen_before = True
                    break
            if seen_before:
                skip = block
                break
        if skip:
            i += skip
            continue
        out.append(lines[i])
        i += 1
    return out


def _dedupe_public_reply(text: str) -> str:
    t = core.normalize_text(text)
    if not t:
        return ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return t

    out: list[str] = []
    seen: set[str] = set()
    seen_long: list[str] = []
    prev_key = ""
    for ln in lines:
        k = _line_dedupe_key(ln)
        if not k:
            continue
        if k == prev_key:
            continue
        if len(k) >= 8 and k in seen:
            continue
        covered = False
        if len(k) >= 8:
            for old in seen_long[-24:]:
                if len(old) >= len(k) + 8 and k in old:
                    covered = True
                    break
        if covered:
            continue
        out.append(ln)
        if k:
            seen.add(k)
        if len(k) >= 14:
            seen_long.append(k)
        prev_key = k

    out = _drop_repeated_line_blocks(out, max_block=10)
    changed = True
    while changed and len(out) >= 4:
        changed = False
        cap = min(8, len(out) // 2)
        for block in range(cap, 0, -1):
            left = out[-2 * block : -block]
            right = out[-block:]
            if not left or not right:
                continue
            if left == right:
                out = out[:-block]
                changed = True
                break
    return core.normalize_text("\n".join(out))


def _compact_public_reply(
    text: str,
    *,
    max_chars: int = GROUP_PUBLIC_REPLY_MAX_CHARS,
    max_lines: int = GROUP_PUBLIC_REPLY_MAX_LINES,
    max_sentences: int = GROUP_PUBLIC_REPLY_MAX_SENTENCES,
) -> str:
    t = core.normalize_text(text)
    if not t:
        return ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(t) <= max_chars and len(lines) <= max_lines:
        return t

    norm_lines: list[str] = []
    for ln in lines:
        s = re.sub(r"^\s*[-*•●]+\s*", "", ln)
        s = re.sub(r"^\s*\d+\s*[).、]\s*", "", s)
        s = re.sub(r"^\s*#{1,6}\s*", "", s)
        s = core.normalize_text(s)
        if not s:
            continue
        if norm_lines and _line_dedupe_key(s) == _line_dedupe_key(norm_lines[-1]):
            continue
        norm_lines.append(s)

    if not norm_lines:
        return _clip_text(t, max_chars)

    flat = " ".join(norm_lines)
    sentences = [x.strip() for x in re.split(r"(?<=[。！？?!；;])\s*", flat) if x.strip()]
    picked_sent: list[str] = []
    used = 0
    scan_cap = max(4, int(max_sentences) * 2)
    for s in sentences[:scan_cap]:
        extra = len(s) + (0 if not picked_sent else 0)
        if used + extra > max_chars:
            break
        picked_sent.append(s)
        used += extra
        if len(picked_sent) >= max(2, int(max_sentences)):
            break
    if picked_sent:
        out = core.normalize_text("".join(picked_sent))
        if out and len(_line_dedupe_key(out)) >= max(18, min(60, max_chars // 3)):
            return out

    picked_lines: list[str] = []
    used_lines = 0
    for ln in norm_lines:
        if len(picked_lines) >= max_lines:
            break
        extra = len(ln) + (1 if picked_lines else 0)
        if used_lines + extra > max_chars:
            break
        picked_lines.append(ln)
        used_lines += extra
    out2 = core.normalize_text("\n".join(picked_lines))
    if out2:
        return out2
    return _clip_text(t, max_chars)


def _strip_reply_history_echo(text: str, previous_texts: list[str]) -> str:
    cur = core.normalize_text(text)
    if not cur:
        return ""
    prevs = [core.normalize_text(x) for x in previous_texts if core.normalize_text(x)]
    if not prevs:
        return cur

    def _split_lines(x: str) -> list[str]:
        return [ln.strip() for ln in core.normalize_text(x).splitlines() if ln.strip()]

    cur_lines = _split_lines(cur)
    if not cur_lines:
        return cur

    cur_keys = [_line_dedupe_key(ln) for ln in cur_lines]
    cur_keys = [k for k in cur_keys if k]
    if not cur_keys:
        return cur

    best = cur
    best_key_len = len(_line_dedupe_key(cur))

    for prev in prevs:
        if not prev or prev == cur:
            continue
        cand = ""
        if cur.startswith(prev) and len(cur) > len(prev):
            cand = core.normalize_text(cur[len(prev) :])

        prev_lines = _split_lines(prev)
        prev_keys = [_line_dedupe_key(ln) for ln in prev_lines]
        prev_keys = [k for k in prev_keys if k]
        if prev_keys and cur_keys:
            common = 0
            max_common = min(len(prev_keys), len(cur_keys))
            while common < max_common and prev_keys[common] == cur_keys[common]:
                common += 1
            if common >= max(2, len(prev_keys) - 1) and len(cur_lines) > common:
                tail = core.normalize_text("\n".join(cur_lines[common:]))
                if tail:
                    cand = tail if (not cand or len(_line_dedupe_key(tail)) < len(_line_dedupe_key(cand))) else cand

            max_overlap = min(len(prev_keys), len(cur_keys) - 1)
            for overlap in range(max_overlap, 1, -1):
                if prev_keys[-overlap:] == cur_keys[:overlap]:
                    tail2 = core.normalize_text("\n".join(cur_lines[overlap:]))
                    if tail2:
                        cand = (
                            tail2
                            if (not cand or len(_line_dedupe_key(tail2)) < len(_line_dedupe_key(cand)))
                            else cand
                        )
                    break

        cand = core.normalize_text(cand)
        if not cand:
            continue
        cand_key_len = len(_line_dedupe_key(cand))
        if cand_key_len >= 6 and cand_key_len < best_key_len:
            best = cand
            best_key_len = cand_key_len

    hist_keys: set[str] = set()
    for prev in prevs:
        for ln in prev.splitlines():
            k = _line_dedupe_key(ln)
            if k and len(k) >= 8:
                hist_keys.add(k)

    if hist_keys:
        best_lines = [ln.strip() for ln in best.splitlines() if ln.strip()]
        if len(best_lines) >= 3:
            kept: list[str] = []
            dropped = 0
            for ln in best_lines:
                k = _line_dedupe_key(ln)
                if k and len(k) >= 8 and k in hist_keys:
                    dropped += 1
                    continue
                kept.append(ln)
            if kept and (dropped >= 2 or (dropped * 1.0 / max(1, len(best_lines))) >= 0.45):
                tail_keep = core.normalize_text("\n".join(kept))
                if tail_keep:
                    best = tail_keep

    if hist_keys:
        best_lines2 = [ln.strip() for ln in best.splitlines() if ln.strip()]
        if len(best_lines2) >= 2:
            idx = 0
            while idx < len(best_lines2) - 1:
                k = _line_dedupe_key(best_lines2[idx])
                if not k or k not in hist_keys:
                    break
                idx += 1
            if idx > 0:
                lead_trim = core.normalize_text("\n".join(best_lines2[idx:]))
                if lead_trim:
                    best = lead_trim

    best_norm = core.normalize_text(best)
    if not best_norm:
        return cur

    cur_key_len = len(_line_dedupe_key(cur))
    best_key_len = len(_line_dedupe_key(best_norm))
    if cur_key_len >= 20 and best_key_len > 0 and best_key_len < max(10, int(cur_key_len * 0.45)):
        return cur
    if re.match(r"^[，,。.!！?？；;:：]", best_norm):
        if best_key_len < max(12, int(cur_key_len * 0.85)):
            return cur
    return best_norm


def _strip_instruction_echo(reply: str, instruction: str) -> str:
    cur = core.normalize_text(reply)
    inst = core.normalize_text(instruction)
    if not cur or not inst:
        return cur

    cur_key = _line_dedupe_key(cur)
    inst_key = _line_dedupe_key(inst)
    if cur_key and inst_key:
        if cur_key == inst_key:
            return ""
        if len(cur_key) >= 16 and cur_key in inst_key:
            return ""
        if len(inst_key) >= 24 and inst_key in cur_key and (len(cur_key) - len(inst_key)) <= 64:
            return ""

    inst_line_keys = {
        _line_dedupe_key(ln)
        for ln in inst.splitlines()
        if _line_dedupe_key(ln) and len(_line_dedupe_key(ln)) >= 10
    }
    if not inst_line_keys:
        return cur

    kept: list[str] = []
    dropped = 0
    for ln in cur.splitlines():
        s = ln.strip()
        if not s:
            continue
        k = _line_dedupe_key(s)
        if k and len(k) >= 10:
            if k in inst_line_keys:
                dropped += 1
                continue
            if any(k in ik for ik in inst_line_keys if len(ik) >= len(k) + 10):
                dropped += 1
                continue
        kept.append(s)

    out = core.normalize_text("\n".join(kept))
    if dropped <= 0:
        return cur
    if not out:
        return ""
    if dropped >= 2 and len(_line_dedupe_key(out)) < 12:
        return ""
    return out


def _is_near_duplicate_reply(text: str, previous_texts: list[str]) -> bool:
    cur = core.normalize_text(text)
    if not cur:
        return False
    cur_key = _line_dedupe_key(cur)
    if not cur_key:
        return False
    cur_lines = {
        _line_dedupe_key(ln)
        for ln in cur.splitlines()
        if _line_dedupe_key(ln) and len(_line_dedupe_key(ln)) >= 8
    }

    for prev in previous_texts:
        old = core.normalize_text(prev)
        if not old:
            continue
        old_key = _line_dedupe_key(old)
        if not old_key:
            continue
        if cur_key == old_key:
            return True
        if len(cur_key) < 10 or len(old_key) < 10:
            continue
        if len(cur_key) >= 18 and len(old_key) >= 18:
            if old_key in cur_key and (len(cur_key) - len(old_key)) <= 88:
                return True
            if cur_key in old_key and (len(old_key) - len(cur_key)) <= 88:
                return True
            ratio = SequenceMatcher(None, cur_key[-1500:], old_key[-1500:]).ratio()
            if ratio >= 0.86:
                return True
            if min(len(cur_key), len(old_key)) >= 24 and ratio >= 0.78:
                pref = 0
                for a, b in zip(cur_key, old_key):
                    if a != b:
                        break
                    pref += 1
                if (pref / max(1, min(len(cur_key), len(old_key)))) >= 0.72:
                    return True
            if ratio >= 0.70:
                intro_pat = r"(?:大家好|我是|我叫|i am|this is)"
                if re.search(intro_pat, cur, re.I) and re.search(intro_pat, old, re.I):
                    return True

        if cur_lines:
            old_lines = {
                _line_dedupe_key(ln)
                for ln in old.splitlines()
                if _line_dedupe_key(ln) and len(_line_dedupe_key(ln)) >= 8
            }
            if len(cur_lines) >= 3 and old_lines:
                overlap = len(cur_lines & old_lines) / max(1, len(cur_lines))
                if overlap >= 0.78:
                    return True
    return False


def _looks_unfinished_public_reply(text: str) -> bool:
    t = core.normalize_text(text)
    if not t:
        return True
    if _WRAP_TOKEN_PAT.search(t):
        pub, _ = _extract_wrapped_reply(t)
        if not core.normalize_text(pub):
            return True
    if _LOW_VALUE_PROCESS_PAT.match(t):
        return True
    if re.search(r"</?\s*think\s*>", t, re.I):
        return True
    if re.search(r"(?:我需要\s*[:：]|让我(?:构思|想一下|数一下|再确认)|检查字数|字数符合要求|好，就这个)", t):
        return True
    if re.match(r"^\s*(?:现在)?我需要.{0,140}(?:然后|再)(?:回应|补充|给出|讨论)", t):
        return True
    if re.search(r"(基于当前信息提出预测与假设|先给区间和假设|首先[，,]先给区间和假设)", t):
        return True
    if ("价格区间" in t and "假设" in t) and not re.search(r"\d", t):
        return True
    m_pref = re.match(r"^(?:@?[\w\u4e00-\u9fff-]{1,24})[：:]\s*(.+)$", t)
    if m_pref and _LOW_VALUE_PROCESS_PAT.match(core.normalize_text(m_pref.group(1))):
        return True
    if re.fullmatch(r"[\u4e00-\u9fff]{4,10}(?:\n@[\w\u4e00-\u9fff-]{1,24})?", t):
        # Keep idiom-like short answers, but reject generic process/title fragments.
        if re.search(
            r"(确认|寻找|评估|分析|讨论|回应|回复|计算|接龙|开始|继续|推进|聚焦|专注|启动|承接|保持|优化|理解|权衡|组织|整理|结果|条件|思考|逻辑|结构|框架|过程|步骤)",
            t,
        ):
            return True
        return False
    k = _line_dedupe_key(t)
    klen = len(k)
    if "\n" not in t and 6 <= klen <= 34:
        if re.search(r"(开始|启动|继续|承接|推进|分析|理解|权衡|优化|聚焦|专注|整理|总结|接龙|回应|流程|进程)", t):
            if not re.search(r"(我|你|他|她|我们|建议|同意|反对|认为|可以|应该|因为|所以)", t):
                return True
    if klen < 8:
        return True
    if any(h in t for h in ("正在构思", "构思中", "先想一下", "先整理一下", "组织语言")):
        return True
    if re.search(r"(?:\d{1,3}\s*字(?:左右)?|字数).{0,10}(?:符合要求|即可)|符合要求(?:即可)?", t):
        return True
    if any(h in t for h in ("不要复述提示词", "不要复述题目", "实质内容", "附一个追问", "直接发你在群里的这条回复")):
        return True
    if t.endswith(("...", "…")) and klen < 64:
        return True
    if re.search(r"[A-Za-z]$", t) and klen < 120:
        return True
    if "\n" not in t and klen < 18 and not re.search(r"[。？！?!]", t):
        return True
    return False


def _looks_protocol_placeholder_public_reply(text: str) -> bool:
    t = core.normalize_text(text)
    if not t:
        return False
    k = _line_dedupe_key(t)
    if not k:
        return False
    # Reject placeholder body copied from the protocol template.
    if re.search(
        r"(这里写公开发言|若不发言写\s*\[?\s*PASS\s*\]?|在此填写公开回复|WRITE_PUBLIC_OR_)",
        t,
        re.I,
    ):
        return True
    if re.search(r"(公开发言放在|请把公开发言放在).{0,40}(之间|PUBLIC_REPLY)", t, re.I):
        return True
    if re.search(r"^\s*(?:\[\[)?PUBLIC_REPLY(?:\]\])?\s*$", t, re.I):
        return True
    return False


def _looks_like_suggestion_chip_reply(text: str) -> bool:
    t = core.normalize_text(text)
    if not t:
        return False
    parts: list[str] = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            continue
        segs = [x.strip() for x in re.split(r"[|｜/]+", s) if x.strip()]
        if segs:
            parts.extend(segs)
        else:
            parts.append(s)
    if len(parts) < 2:
        return False

    chip_like = 0
    for p in parts:
        key_len = len(_line_dedupe_key(p))
        if key_len <= 2 or key_len > 34:
            continue
        if re.search(r"[。！？?!；;:：]", p):
            continue
        if re.search(r"^(生成|写一份|写一封|帮我|推荐|提供|整理|翻译|总结)", p):
            chip_like += 1
            continue
        if p.endswith("生成") or p.endswith("模板") or p.endswith("大纲"):
            chip_like += 1
            continue
    return chip_like >= max(2, int(len(parts) * 0.65))


def _strip_trailing_solicit_line(text: str) -> str:
    t = core.normalize_text(text)
    if not t:
        return ""
    t = re.sub(
        r"^\s*(?:ChatGPT|Gemini|DeepSeek|Qwen|Doubao|豆包|通义千问|千问)\s*说\s*(?:[/：:]\s*)?",
        "",
        t,
        flags=re.I,
    )
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return t

    while lines:
        tail = lines[-1]
        klen = len(_line_dedupe_key(tail))
        if re.match(r"^[A-Za-z][A-Za-z0-9 _-]{1,20}\s*说$", tail):
            lines.pop()
            continue
        if klen <= 56 and re.search(r"(需要我|要不要我|如果你需要|是否需要我|我可以继续|我可以帮你|还需要我)", tail):
            lines.pop()
            continue
        if klen <= 64 and re.search(r"(要我为你|要我给你|需要我给你|我来帮你整理)", tail):
            lines.pop()
            continue
        if klen <= 48 and re.search(r"(还有什么|继续聊|继续问|欢迎继续|想聊的尽管说)", tail):
            lines.pop()
            continue
        break

    out = core.normalize_text("\n".join(lines))
    if out:
        return out
    if re.search(r"(需要我|要不要我|如果你需要|是否需要我|我可以继续|我可以帮你|还需要我|要我为你|要我给你|需要我给你)", t):
        return ""
    return t


def _looks_stale_extracted_reply(reply: str, before_snapshot: str, *, before_last_reply: str = "") -> bool:
    cur = core.normalize_text(reply)
    if not cur:
        return False

    cur_key = _line_dedupe_key(cur)
    prev_last_key = _line_dedupe_key(before_last_reply)
    if cur_key and prev_last_key and len(cur_key) >= 10 and cur_key == prev_last_key:
        return True

    before_keys = {
        _line_dedupe_key(ln)
        for ln in core.normalize_text(before_snapshot).splitlines()
        if _line_dedupe_key(ln) and len(_line_dedupe_key(ln)) >= 8
    }
    if not before_keys:
        return False

    cur_keys = [
        _line_dedupe_key(ln)
        for ln in cur.splitlines()
        if _line_dedupe_key(ln) and len(_line_dedupe_key(ln)) >= 8
    ]
    if not cur_keys:
        return False

    unseen = [k for k in cur_keys if k not in before_keys]
    if not unseen:
        return True
    overlap = 1.0 - (len(unseen) / max(1, len(cur_keys)))
    if overlap >= 0.86 and len(unseen) <= 1 and len(cur_keys) >= 2:
        return True
    return False


def _pick_best_semantic_fragment(text: str) -> str:
    t = core.normalize_text(text)
    if not t:
        return ""
    segments: list[tuple[int, str]] = []
    for idx_line, ln in enumerate(t.splitlines()):
        s = ln.strip()
        if not s:
            continue
        segs = [x.strip() for x in re.split(r"[|｜]+", s) if x.strip()]
        if not segs:
            continue
        for seg in segs:
            segments.append((idx_line, seg))
    if len(segments) <= 1:
        return t

    cleaned: list[tuple[float, str]] = []
    for idx, seg0 in segments:
        seg = core.normalize_text(seg0)
        if not seg:
            continue
        m_pref = re.match(r"^(?:@?[\w\u4e00-\u9fff-]{1,24})[：:]\s*(.+)$", seg)
        if m_pref:
            tail = core.normalize_text(m_pref.group(1))
            if tail and not _looks_unfinished_public_reply(tail):
                seg = tail
        if _TRIVIAL_PUBLIC_PAT.match(seg) or _QWEN_STATUS_ONLY_PAT.match(seg):
            continue
        if _looks_like_suggestion_chip_reply(seg):
            continue
        if _looks_unfinished_public_reply(seg):
            continue
        klen = len(_line_dedupe_key(seg))
        if klen < 4:
            continue
        bad_hint = 0
        if re.search(r"(上下文边界|忽略网页里更早的旧对话|可回应对象|可点名对象|只输出\s*\[?\s*PASS|状态词)", seg, re.I):
            bad_hint += 3
        if re.search(r"请(?:直接|基于|必须|先|你).*?(?:回复|回应|观点|短消息)", seg):
            bad_hint += 2
        if bad_hint >= 2 and klen <= 84:
            continue
        score = 0.0
        score += idx * 0.15
        if re.search(r"[。！？?!]", seg):
            score += 2.0
        if 4 <= klen <= 64:
            score += 2.0
        if klen > 64:
            score += 0.8
        if "@" in seg:
            score += 0.5
        if _LOW_VALUE_PROCESS_PAT.match(seg) or re.search(
            r"(继续推进|聚焦|专注回应|忽略初始回合|专注解析|承接对话|延续语意流|权衡上下文限制)",
            seg,
        ):
            score -= 3.0
        cleaned.append((score, seg))

    if not cleaned:
        return t
    cleaned.sort(key=lambda x: (x[0], len(_line_dedupe_key(x[1]))))
    return core.normalize_text(cleaned[-1][1]) or t


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
        self._thread_lock = threading.Lock()
        self._pw = None
        self._sp = None
        self._adapters: dict[str, ModelAdapter] = {}
        # 每次后端重启后，DeepSeek 首次发言前强制新建对话，避免误用旧线程历史。
        self._deepseek_need_fresh_chat = True
        # 每次后端重启后，Qwen 首次发言前也新建对话，避免串旧上下文。
        self._qwen_need_fresh_chat = True
        # 豆包不强制首轮新对话：部分页面版本在“新对话”按钮切换后首轮提取会不稳定。
        self._doubao_need_fresh_chat = False
        # ChatGPT / Gemini 在长会话下也会明显串旧话题；群任务开始时尽量新开对话。
        self._chatgpt_need_fresh_chat = True
        self._gemini_need_fresh_chat = True
        self._turn_seq = 0
        self._last_receipt_by_model: dict[str, Receipt] = {}
        self._last_turn_ctx_by_model: dict[str, TurnContext] = {}
        self._atexit_registered = False

    def _control_plane_header_for_model(self, model_key: str) -> str:
        r = self._last_receipt_by_model.get((model_key or "").strip().lower())
        if r is None:
            return "【系统回执】无（首次发言）。\n"
        if r.status == ReceiptStatus.ACCEPT:
            return f"【系统回执】上一轮：ACCEPT（turn_id={r.turn_id}）。继续正常输出。\n"
        if r.status == ReceiptStatus.PASS_:
            return f"【系统回执】上一轮：PASS（turn_id={r.turn_id}）。\n"
        reason = r.reason.value if r.reason is not None else "UNKNOWN"
        return (
            f"【系统回执】上一轮：REJECT（turn_id={r.turn_id}, reason={reason}）。"
            "若再次被 REJECT，请缩短输出或仅给 BLOCKER/DIFF。\n"
        )

    def _finalize_receipt_for_model(
        self,
        model_key: str,
        *,
        status: ReceiptStatus,
        reason: Optional[RejectReason] = None,
        note: Optional[str] = None,
    ) -> None:
        key_norm = (model_key or "").strip().lower()
        if not key_norm:
            return
        ctx = self._last_turn_ctx_by_model.get(key_norm)
        turn_id = int(getattr(ctx, "turn_id", 0) or 0)
        rec = Receipt(turn_id=turn_id, status=status, reason=reason, note=note)
        self._last_receipt_by_model[key_norm] = rec
        _trace_turn(key_norm, "receipt_final", json.dumps(rec.to_dict(), ensure_ascii=False))

    def _build_authoritative_broadcast_packet(
        self, *, max_items: int = 10, max_chars: int = 2400
    ) -> tuple[str, str]:
        msgs = self.state.get_all_messages()
        pub = [
            m
            for m in msgs
            if m.visibility == "public"
            and m.role in {"user", "model"}
            and core.normalize_text(m.text)
        ]
        items = pub[-max(1, int(max_items)) :]
        lines = [f"【广播包 run_id={RUN_ID}】"]
        for mm in items:
            role = "U" if mm.role == "user" else "M"
            speaker = core.normalize_text(mm.speaker)
            text = core.normalize_text(mm.text).replace("\n", " ").strip()
            if len(text) > 220:
                text = text[:220] + "…"
            lines.append(f"- {role}#{mm.id}({speaker}): {text}")
        lines.append("【广播包结束】")
        packet = "\n".join(lines)
        if len(packet) > max_chars:
            packet = packet[:max_chars] + "\n【广播包截断】"
        h = hashlib.sha1(packet.encode("utf-8")).hexdigest()[:12]
        return packet, h

    def _ack_in_from_public_timeline(self) -> str:
        msgs = self.state.get_all_messages()
        pub_ids = [int(m.id) for m in msgs if m.visibility == "public"]
        return str(max(pub_ids)) if pub_ids else "0"

    def start(self) -> None:
        if not self._atexit_registered:
            try:
                atexit.register(self._cleanup_playwright_best_effort)
                self._atexit_registered = True
            except Exception:
                pass
        with self._thread_lock:
            try:
                if self._thread.is_alive():
                    return
            except Exception:
                pass
            # Thread objects cannot be restarted after exit.
            self._thread = threading.Thread(target=self._run, name="pw-worker", daemon=True)
            self._thread.start()

    def _cleanup_playwright_best_effort(self) -> None:
        try:
            for ad in list(self._adapters.values()):
                try:
                    ad.close()
                except Exception:
                    pass
            self._adapters.clear()
        except Exception:
            pass
        try:
            if self._sp is not None:
                self._sp.stop()
        except Exception:
            pass
        self._sp = None
        self._pw = None

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

    @staticmethod
    def _classify_worker_failure(exc_or_text: Any) -> str:
        t = core.normalize_text(str(exc_or_text or "")).lower()
        if not t:
            return "unknown"
        if any(
            x in t
            for x in [
                "launch_persistent_context",
                "browsertype.launch",
                "target page, context or browser has been closed",
                "user-data-dir",
                "exitcode=21",
                "process did exit: exitcode=21",
            ]
        ):
            return "profile_locked"
        if any(x in t for x in ["captcha", "验证", "人机", "cloudflare", "cf", "风控", "过盾"]):
            return "captcha"
        if any(x in t for x in ["timeout", "timed out", "超时"]):
            return "timeout"
        if any(x in t for x in ["stale", "detached", "execution context was destroyed"]):
            return "stale"
        if any(x in t for x in ["selector", "locator", "input", "对话输入框不可用"]):
            return "selector_miss"
        return "unknown"

    @staticmethod
    def _runtime_gate_decision(
        runtime_state: str,
        cooldown_until: float,
        *,
        now_ts: Optional[float] = None,
    ) -> str:
        st = (runtime_state or "INIT").strip().upper()
        now_v = float(time.time() if now_ts is None else now_ts)
        cd = float(cooldown_until or 0.0)
        if st in {"NEEDS_HUMAN", "CAPTCHA_BLOCKED"}:
            return "BLOCK_HUMAN"
        if st == "DEAD":
            return "BLOCK_DEAD"
        if st == "COOLING_DOWN":
            if cd > now_v:
                return "SKIP_COOLDOWN"
            return "HALF_OPEN"
        return "ALLOW"

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
                elif kind == "recover_model":
                    self._handle_recover_model(action)
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
            self._cleanup_playwright_best_effort()
        finally:
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
        self.state.set_model_runtime_state(key, "INITIALIZING", reason="login_open")
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

    def _repair_adapter_surface(self, key: str, ad: ModelAdapter, *, level: int = 1) -> bool:
        """
        Best-effort page recovery ladder:
        level 1 -> reload current page
        level 2 -> close tab and recreate one inside existing context
        """
        try:
            self._ensure_playwright()
        except Exception:
            return False
        try:
            page = ad.ensure_page(self._pw)
        except Exception:
            return False
        if level <= 1:
            try:
                page.reload(wait_until="domcontentloaded")
                time.sleep(0.6)
                return True
            except Exception:
                return False
        # Level 2 hard reset: rebuild tab inside current context
        try:
            ctx = getattr(ad, "context", None)
            old_page = getattr(ad, "page", None)
            if old_page is not None:
                try:
                    old_page.close()
                except Exception:
                    pass
            if ctx is not None:
                try:
                    new_page = ctx.new_page()
                    ad.page = new_page
                    try:
                        new_page.goto(ad.meta.url, wait_until="domcontentloaded")
                    except Exception:
                        pass
                    time.sleep(0.8)
                    return True
                except Exception:
                    return False
        except Exception:
            return False
        return False
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
        self.state.set_model_runtime_state(key, "INITIALIZING", reason="login_check")
        page = ad.ensure_page(self._pw)
        m = self.state.get_model(key)

        # 允许页面在“已登录但输入框晚出现”时完成初始化，降低误判。
        ok = False
        deadline = time.time() + 18.0
        while time.time() < deadline:
            if m and m.key == "deepseek":
                try:
                    self._ensure_deepseek_dialog(page)
                except Exception:
                    pass
            try:
                ok = ad.is_authenticated_now()
            except Exception:
                ok = False
            if ok:
                break
            time.sleep(0.6)

        self.state.set_authenticated(key, ok)
        if ok:
            self.state.mark_pending_enable_done(key)
            self.state.set_model_runtime_state(key, "IDLE", reason="login_check_ok", reset_fail=True)
        else:
            self.state.set_model_runtime_state(key, "NEEDS_HUMAN", reason="login_check_failed", fail_delta=1)
            if m:
                self.state.add_system(f"{m.name} 需要人工介入：请打开登录窗口完成验证/登录后点击恢复。")
        self.state.set_status("idle")
        self._safe_reply(action, {"ok": True, "authenticated": bool(ok)})

    def _handle_recover_model(self, action: dict[str, Any]) -> None:
        key = str(action.get("key") or "").strip().lower()
        ad = self._get_adapter(key)
        if ad is None:
            self._safe_reply(action, {"ok": False, "error": "unknown_or_not_integrated"})
            return
        self._ensure_playwright()
        assert self._pw is not None
        self.state.set_status(f"recover:{key}")
        self.state.set_model_runtime_state(key, "INITIALIZING", reason="manual_recover")
        page = ad.ensure_page(self._pw)
        try:
            if not self._repair_adapter_surface(key, ad, level=1):
                self._repair_adapter_surface(key, ad, level=2)
            ok = bool(ad.is_authenticated_now())
            self.state.set_authenticated(key, ok)
            if ok:
                self.state.set_model_runtime_state(key, "IDLE", reason="manual_recover_ok", reset_fail=True)
                self._safe_reply(action, {"ok": True, "recovered": True})
            else:
                self.state.set_model_runtime_state(key, "NEEDS_HUMAN", reason="manual_recover_needs_login", fail_delta=1)
                self._safe_reply(action, {"ok": True, "recovered": False, "needs_human": True})
        except Exception as exc:
            cls = self._classify_worker_failure(exc)
            if cls == "captcha":
                self.state.set_model_runtime_state(key, "NEEDS_HUMAN", reason=f"recover:{cls}", fail_delta=1)
            elif cls == "profile_locked":
                self.state.set_model_runtime_state(key, "NEEDS_HUMAN", reason=f"recover:{cls}", fail_delta=1)
            elif cls in {"timeout", "stale", "selector_miss"}:
                self.state.set_model_runtime_state(key, "COOLING_DOWN", reason=f"recover:{cls}", fail_delta=1, cooldown_s=30)
            else:
                self.state.set_model_runtime_state(key, "DEAD", reason=f"recover:{cls}", fail_delta=1)
            self._safe_reply(action, {"ok": False, "error": str(exc), "class": cls})
        finally:
            self.state.set_status("idle")

    @staticmethod
    def _click_generic_new_chat(page: Any) -> bool:
        candidates = [
            page.get_by_role("button", name=re.compile(r"新建对话|新对话|新聊天|new chat|new conversation", re.I)),
            page.get_by_role("link", name=re.compile(r"新建对话|新对话|新聊天|new chat|new conversation", re.I)),
            page.get_by_text(re.compile(r"新建对话|新对话|新聊天|new chat|new conversation", re.I)),
            page.locator("a:has-text('新建对话'),a:has-text('新对话'),a:has-text('新聊天')"),
            page.locator("button[aria-label*='New chat' i],button[title*='New chat' i]"),
            page.locator("[data-testid*='new' i],[data-test*='new' i]"),
        ]
        for loc in candidates:
            node = core.pick_visible(loc, prefer_last=False)
            if node is None:
                continue
            try:
                node.click(timeout=3000)
                time.sleep(0.8)
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def _click_deepseek_new_chat(page: Any) -> bool:
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
                time.sleep(0.8)
                return True
            except Exception:
                continue
        return False

    def _ensure_deepseek_dialog(self, page: Any, *, force_new_chat: bool = False) -> bool:
        """
        Ensure DeepSeek has an active dialog panel.
        - force_new_chat=True: 强制优先点“新对话”（用于后端重启后的首次发言）。
        """
        if force_new_chat and self._click_deepseek_new_chat(page):
            return True

        try:
            if page.locator("div.ds-message").count() > 0:
                return True
        except Exception:
            pass

        clicked = False
        # Fallback: open the latest existing thread if visible.
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

        if clicked:
            time.sleep(0.8)
            return True

        return self._click_deepseek_new_chat(page)

    @staticmethod
    def _click_qwen_new_chat(page: Any) -> bool:
        candidates = [
            page.get_by_role("button", name=re.compile(r"新建对话|新对话|new chat|new conversation", re.I)),
            page.get_by_text(re.compile(r"新建对话|新对话|new chat|new conversation", re.I)),
            page.get_by_role("link", name=re.compile(r"新建对话|新对话|new chat|new conversation", re.I)),
            page.locator("a:has-text('新建对话'),a:has-text('新对话')"),
            page.locator("[data-testid*='new' i],[data-test*='new' i]"),
        ]
        for loc in candidates:
            node = core.pick_visible(loc, prefer_last=False)
            if node is None:
                continue
            try:
                node.click(timeout=3000)
                time.sleep(0.8)
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def _click_doubao_new_chat(page: Any) -> bool:
        candidates = [
            page.get_by_role("button", name=re.compile(r"新对话|新建对话|开始新对话|new chat", re.I)),
            page.get_by_text(re.compile(r"新对话|新建对话|开始新对话|new chat", re.I)),
            page.get_by_role("link", name=re.compile(r"新对话|新建对话|开始新对话|new chat", re.I)),
            page.locator("button[aria-label*='新对话'], button[title*='新对话']"),
            page.locator("a:has-text('新对话'),a:has-text('新建对话')"),
            page.locator("[data-testid*='new' i],[data-test*='new' i]"),
        ]
        for loc in candidates:
            node = core.pick_visible(loc, prefer_last=False)
            if node is None:
                continue
            try:
                node.click(timeout=3000)
                time.sleep(0.8)
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def _reset_adapter_reply_cache(ad: ModelAdapter) -> None:
        # Fresh web thread should not be contaminated by previous-turn extraction cache.
        for attr, value in (("_last_effective_reply", ""), ("_recent_sent_line_keys", [])):
            if hasattr(ad, attr):
                try:
                    setattr(ad, attr, value)
                except Exception:
                    pass

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
            forced = bool(self._deepseek_need_fresh_chat)
            ok = self._ensure_deepseek_dialog(page, force_new_chat=forced)
            if forced and ok:
                self._deepseek_need_fresh_chat = False
                self._reset_adapter_reply_cache(ad)
        elif m.key == "doubao" and self._doubao_need_fresh_chat:
            if self._click_doubao_new_chat(page):
                self._doubao_need_fresh_chat = False
                self._reset_adapter_reply_cache(ad)
            else:
                try:
                    page.goto(ad.meta.url, wait_until="domcontentloaded")
                    time.sleep(0.6)
                except Exception:
                    pass
                if self._click_doubao_new_chat(page):
                    self._doubao_need_fresh_chat = False
                    self._reset_adapter_reply_cache(ad)
        elif m.key == "qwen" and self._qwen_need_fresh_chat:
            if self._click_qwen_new_chat(page):
                self._qwen_need_fresh_chat = False
                self._reset_adapter_reply_cache(ad)
            else:
                try:
                    page.goto(ad.meta.url, wait_until="domcontentloaded")
                    time.sleep(0.6)
                except Exception:
                    pass
                if self._click_qwen_new_chat(page):
                    self._qwen_need_fresh_chat = False
                    self._reset_adapter_reply_cache(ad)
        elif m.key in {"chatgpt", "gemini"}:
            need = self._chatgpt_need_fresh_chat if m.key == "chatgpt" else self._gemini_need_fresh_chat
            if need:
                ok_new = self._click_generic_new_chat(page)
                if not ok_new:
                    try:
                        page.goto(ad.meta.url, wait_until="domcontentloaded")
                        time.sleep(0.6)
                    except Exception:
                        pass
                    ok_new = self._click_generic_new_chat(page)
                if ok_new:
                    if m.key == "chatgpt":
                        self._chatgpt_need_fresh_chat = False
                    else:
                        self._gemini_need_fresh_chat = False
                    self._reset_adapter_reply_cache(ad)

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
            # Some pages need a short settle window after navigation/reload.
            for _ in range(6):
                time.sleep(0.5)
                if ad.find_input() is not None:
                    break

        if ad.find_input() is None:
            raise RuntimeError(f"{m.name} 对话输入框不可用（可能被弹窗遮挡或未进入会话）")
        return page

    @staticmethod
    def _compose_turn_input(instruction: str, *, hidden_reply_hint: bool) -> str:
        body = core.normalize_text(instruction)
        if not body:
            return ""
        if not hidden_reply_hint:
            return body
        hint = "你是群聊成员，这条回复会用于后续讨论。可旁听时回复 [PASS]。"
        return f"{hint}\n\n{body}".strip()

    def _run_model_turn(
        self,
        key: str,
        instruction: str,
        *,
        visibility: str,
        hidden_reply_hint: bool = False,
        record_reply: bool = True,
        timeout_cap_s: Optional[int] = None,
        authoritative_packet_text: str = "",
        authoritative_packet_hash: str = "",
        ack_in: str = "",
        reply_to_mid: str = "",
    ) -> TurnResult:
        m = self.state.get_model(key)
        adapter_name = (m.name if m else key) or key
        ctx: Optional[TurnContext] = None

        def _tr_success(public_text: str, *, raw_reply: str = "", private_text: str = "", envelope_status: str = "", meta: Optional[dict[str, str]] = None, elapsed_s: Optional[float] = None) -> TurnResult:
            metrics = {}
            if elapsed_s is not None:
                metrics["duration_s"] = float(elapsed_s)
            return TurnResult.success(
                adapter_name=adapter_name,
                model_key=key,
                parsed_content=public_text,
                raw_reply=raw_reply,
                private_content=private_text,
                turn_id=(ctx.turn_id if ctx else 0),
                mode=(ctx.mode.value if ctx else ""),
                packet_hash=authoritative_packet_hash or "",
                envelope_status=envelope_status,
                metrics=metrics,
                meta=meta or {},
            )

        def _tr_pass(*, reason: str = "", raw_reply: str = "", private_text: str = "", envelope_status: str = "", meta: Optional[dict[str, str]] = None, elapsed_s: Optional[float] = None) -> TurnResult:
            metrics = {}
            if elapsed_s is not None:
                metrics["duration_s"] = float(elapsed_s)
            return TurnResult.pass_(
                adapter_name=adapter_name,
                model_key=key,
                reason=reason,
                raw_reply=raw_reply,
                private_content=private_text,
                turn_id=(ctx.turn_id if ctx else 0),
                mode=(ctx.mode.value if ctx else ""),
                packet_hash=authoritative_packet_hash or "",
                envelope_status=envelope_status,
                metrics=metrics,
                meta=meta or {},
            )

        def _tr_error(error_type: TurnErrorType, *, msg: str = "", raw_reply: str = "", parsed: str = "", private_text: str = "", envelope_status: str = "", meta: Optional[dict[str, str]] = None, elapsed_s: Optional[float] = None) -> TurnResult:
            metrics = {}
            if elapsed_s is not None:
                metrics["duration_s"] = float(elapsed_s)
            return TurnResult.error(
                adapter_name=adapter_name,
                model_key=key,
                error_type=error_type,
                error_msg=msg,
                raw_reply=raw_reply,
                parsed_content=parsed,
                private_content=private_text,
                turn_id=(ctx.turn_id if ctx else 0),
                mode=(ctx.mode.value if ctx else ""),
                packet_hash=authoritative_packet_hash or "",
                envelope_status=envelope_status,
                metrics=metrics,
                meta=meta or {},
            )

        if not m or not m.integrated:
            self.state.add_system(f"目标模型不可用: {key}")
            return _tr_error(TurnErrorType.MODEL_UNAVAILABLE, msg="model unavailable")
        if not m.authenticated:
            self.state.add_system(f"{m.name} 未登录：请先点模型 -> 打开登录窗口 -> 重新检测")
            return _tr_error(TurnErrorType.NOT_AUTHENTICATED, msg="not authenticated")

        ad = self._get_adapter(key)
        if ad is None:
            self.state.add_system(f"适配器缺失: {m.name}")
            return _tr_error(TurnErrorType.ADAPTER_MISSING, msg="adapter missing")

        rt_box = self.state.get_model_runtime_state(key)
        rt_state = str((rt_box or {}).get("state") or "INIT")
        rt_cooldown_remaining = float((rt_box or {}).get("cooldown_remaining_s") or 0.0)
        rt_decision = self._runtime_gate_decision(
            rt_state,
            time.time() + rt_cooldown_remaining if rt_cooldown_remaining > 0 else 0.0,
        )
        if rt_decision == "BLOCK_HUMAN":
            self.state.add_system(f"{m.name} 当前需要人工介入（验证码/风控/登录），请处理后点击恢复。")
            return _tr_error(TurnErrorType.PROVIDER_ERROR, msg=f"runtime blocked: {rt_state}")
        if rt_decision == "BLOCK_DEAD":
            self.state.add_system(f"{m.name} 当前处于不可用状态（DEAD），请点击恢复或重新登录。")
            return _tr_error(TurnErrorType.PROVIDER_ERROR, msg="runtime blocked: DEAD")
        if rt_decision == "SKIP_COOLDOWN":
            return _tr_error(TurnErrorType.TIMEOUT, msg=f"runtime cooling_down:{rt_cooldown_remaining:.1f}s")
        if rt_decision == "HALF_OPEN":
            self.state.set_model_runtime_state(key, "INITIALIZING", reason="cooldown_half_open_probe")
            try:
                self._ensure_playwright()
                assert self._pw is not None
                if not self._repair_adapter_surface(key, ad, level=1):
                    self._repair_adapter_surface(key, ad, level=2)
                probe_ok = False
                try:
                    probe_ok = bool(ad.is_authenticated_now())
                except Exception:
                    probe_ok = False
                if probe_ok:
                    self.state.set_model_runtime_state(key, "IDLE", reason="cooldown_half_open_ok", reset_fail=True)
                else:
                    self.state.set_model_runtime_state(key, "NEEDS_HUMAN", reason="cooldown_half_open_auth_failed", fail_delta=1)
                    return _tr_error(TurnErrorType.PROVIDER_ERROR, msg="half_open probe requires human")
            except Exception as exc:
                cls = self._classify_worker_failure(exc)
                if cls == "captcha":
                    self.state.set_model_runtime_state(key, "NEEDS_HUMAN", reason=f"half_open:{cls}", fail_delta=1)
                elif cls == "profile_locked":
                    self.state.set_model_runtime_state(key, "NEEDS_HUMAN", reason=f"half_open:{cls}", fail_delta=1)
                else:
                    self.state.set_model_runtime_state(key, "COOLING_DOWN", reason=f"half_open:{cls}", fail_delta=1, cooldown_s=45)
                return _tr_error(
                    TurnErrorType.TIMEOUT if cls == "timeout" else TurnErrorType.PARSE_FAIL,
                    msg=f"half_open probe failed:{cls}",
                )

        try:
            turn_start = time.time()
            self.state.set_status(f"sending:{key}")
            self.state.set_model_runtime_state(key, "GENERATING", reason="turn_start")
            try:
                instruction = self._control_plane_header_for_model(key) + (instruction or "")
            except Exception:
                pass
            page = self._ensure_chat_surface(ad, m)
            strict_token = _extract_expected_short_literal(instruction)
            self._turn_seq += 1
            turn_mode = Mode.MULTI_ROUND if (visibility == "public" and not record_reply) else Mode.SINGLE_FAST
            ctx = TurnContext(turn_id=self._turn_seq, mode=turn_mode)
            self._last_turn_ctx_by_model[(key or "").strip().lower()] = ctx
            defer_final_receipt = bool(visibility == "public" and not record_reply)
            try:
                if ctx.mode != Mode.EVIDENCE:
                    msgs = self.state.get_all_messages()
                    recent = [
                        core.normalize_text(mm.text)
                        for mm in msgs
                        if mm.visibility == "public" and mm.role == "model"
                    ]
                    if should_enter_evidence(recent):
                        ctx.mode = Mode.EVIDENCE
            except Exception:
                pass
            try:
                if ctx.mode == Mode.EVIDENCE:
                    instruction = (instruction or "") + (
                        "\n【证据模式】若你要反驳/质疑，PRIVATE_REPLY 必须包含一行："
                        "evidence_hook=ref:M#123 或 evidence_hook=check:可验证条件。否则输出 [PASS]。\n"
                    )
            except Exception:
                pass
            prompt = self._compose_turn_input(instruction, hidden_reply_hint=hidden_reply_hint)
            if not prompt:
                self.state.add_system(f"{m.name} 本轮跳过：空指令")
                return _tr_error(TurnErrorType.EMPTY_INSTRUCTION, msg="empty instruction")
            if authoritative_packet_text:
                prompt = f"{authoritative_packet_text}\n\n{prompt}".strip()
                _trace_turn(
                    key,
                    "broadcast_packet",
                    (
                        f"turn_id={ctx.turn_id} mode={ctx.mode.value} packet_hash={authoritative_packet_hash} "
                        f"ack_in={ack_in or ''}"
                    ),
                )
            if not strict_token:
                prompt = (
                    prompt
                    + _protocol_wrap_instruction_suffix(
                        mode=ctx.mode.value,
                        hidden_reply_hint=hidden_reply_hint,
                        turn_id=ctx.turn_id,
                        reply_to=reply_to_mid or "",
                        ack_in=ack_in or "",
                        packet_hash=authoritative_packet_hash or "",
                    )
                ).strip()

            _trace_turn(key, "prompt", f"turn_id={ctx.turn_id} mode={ctx.mode.value}\n{prompt}")
            before = ad.snapshot_conversation()
            before_last_reply = ""
            try:
                if key == "gemini":
                    before_last_reply = core.normalize_text(core.extract_gemini_last_reply(page))
                else:
                    pick_last = getattr(ad, "_extract_last_reply_candidate", None)
                    if callable(pick_last):
                        before_last_reply = core.normalize_text(pick_last())
            except Exception:
                before_last_reply = ""
            before_full_snapshot = ""
            if key in {"doubao", "qwen"}:
                try:
                    before_full_snapshot = core.normalize_text(page.locator("main").first.inner_text())
                except Exception:
                    try:
                        before_full_snapshot = core.normalize_text(page.locator("body").first.inner_text())
                    except Exception:
                        before_full_snapshot = ""
            ad.send_user_text(prompt)
            timeout_s = int(MODEL_REPLY_TIMEOUT_OVERRIDES.get(key, MODEL_REPLY_TIMEOUT_S))
            if timeout_cap_s is not None:
                try:
                    timeout_s = max(timeout_s, max(6, int(timeout_cap_s)))
                except Exception:
                    pass
            # Group public turns must stay responsive; avoid one model blocking the whole round too long.
            if visibility == "public" and not record_reply:
                soft_cap = int(GROUP_PUBLIC_TURN_SOFT_TIMEOUT_S.get(key, GROUP_PUBLIC_TURN_SOFT_TIMEOUT_S["default"]))
                if timeout_cap_s is not None:
                    try:
                        soft_cap = max(soft_cap, int(timeout_cap_s))
                    except Exception:
                        pass
                timeout_s = min(timeout_s, max(8, soft_cap))
            if timeout_cap_s is not None:
                try:
                    cap = max(6, int(timeout_cap_s))
                    timeout_s = min(timeout_s, cap)
                except Exception:
                    pass
            reply = core.normalize_text(ad.wait_reply_and_extract(before, timeout_s=timeout_s))
            _trace_turn(
                key,
                "extract",
                f"turn_id={ctx.turn_id} mode={ctx.mode.value}\n{reply}",
                elapsed_s=(time.time() - turn_start),
            )
            if key in {"doubao", "gemini"} and _looks_prompt_leak_reply(reply):
                # Fast retry can often pick finalized assistant content
                # instead of transient prompt-echo/nav blocks.
                try:
                    time.sleep(0.8)
                    retry_timeout = 12 if key == "doubao" else 18
                    retry = core.normalize_text(ad.wait_reply_and_extract(before, timeout_s=retry_timeout))
                    if retry and not _looks_prompt_leak_reply(retry):
                        reply = retry
                except Exception:
                    pass
            if not reply:
                # Last-chance rescue: some web UIs intermittently miss one polling window.
                try:
                    raw_last = getattr(ad, "_extract_last_reply_candidate", lambda: "")()
                    raw_last = core.normalize_text(raw_last)
                    if raw_last:
                        cleaner = getattr(ad, "_clean_candidate_text", None)
                        if callable(cleaner):
                            raw_last = core.normalize_text(cleaner(raw_last))
                    if raw_last:
                        reply = raw_last
                except Exception:
                    pass
            if not reply and visibility == "public" and not record_reply and key in {"doubao", "qwen"}:
                # One extra wait-without-resend when extraction is empty:
                # improves first-turn capture without adding extra token spend.
                late_timeout = 16 if key == "doubao" else 16
                try:
                    late = core.normalize_text(ad.wait_reply_and_extract(before, timeout_s=late_timeout))
                except Exception:
                    late = ""
                if late:
                    reply = late
            if not reply and key in {"doubao", "qwen"} and before_full_snapshot:
                try:
                    after_full = core.normalize_text(page.locator("main").first.inner_text())
                except Exception:
                    try:
                        after_full = core.normalize_text(page.locator("body").first.inner_text())
                    except Exception:
                        after_full = ""
                if after_full and after_full != before_full_snapshot:
                    try:
                        diff_full = core.normalize_text(ad._diff_reply(before_full_snapshot, after_full))
                    except Exception:
                        diff_full = ""
                    if diff_full:
                        diff_full = _strip_instruction_echo(diff_full, instruction)
                        diff_full = _sanitize_forward_payload(diff_full) or diff_full
                        diff_full = _strip_leading_status_noise(diff_full) or diff_full
                        diff_full = _dedupe_public_reply(diff_full) or diff_full
                        diff_full = _pick_best_semantic_fragment(diff_full) or diff_full
                        if (
                            diff_full
                            and not _looks_prompt_leak_reply(diff_full)
                            and not _looks_unfinished_public_reply(diff_full)
                            and not _looks_like_suggestion_chip_reply(diff_full)
                        ):
                            reply = diff_full
            if not reply:
                if visibility == "public" and not record_reply:
                    # In group auto-rotation, give one short retry before treating it as silent pass.
                    retry_empty = ""
                    try:
                        retry_empty = core.normalize_text(ad.wait_reply_and_extract(before, timeout_s=10))
                    except Exception:
                        retry_empty = ""
                    if retry_empty:
                        reply = retry_empty
                    else:
                        _trace_turn(key, "pass(empty)", "")
                        return _tr_pass(reason="empty")
                reply = "（未能提取到回复，可能仍在生成中或页面结构变化）"

            env = parse_envelope(reply)
            if env.public:
                meta_packet_hash = core.normalize_text(str(env.meta.get("packet_hash") or ""))
                if authoritative_packet_hash and meta_packet_hash and meta_packet_hash != authoritative_packet_hash:
                    self.state.add_system(f"{m.name} 本轮广播包校验失败：packet_hash 不一致，已隔离该回复。")
                    _trace_turn(
                        key,
                        "ERROR(packet_hash_mismatch_meta)",
                        (
                            f"turn_id={ctx.turn_id} mode={ctx.mode.value} "
                            f"expected={authoritative_packet_hash} got={meta_packet_hash}"
                        ),
                    )
                    return _tr_pass(reason="packet_hash_mismatch_meta", raw_reply=reply, envelope_status=env.status, meta=env.meta)
                public_reply = env.public
                private_reply = env.private
                _trace_turn(
                    key,
                    "envelope",
                    (
                        f"turn_id={ctx.turn_id} mode={ctx.mode.value} status={env.status} "
                        f"meta={env.meta} ack_out={env.meta.get('ack_out') or env.meta.get('ack') or ''}"
                    ),
                )
            elif _should_pass_no_public_tagged(env.status, env.public):
                # Protocol tags exist but public block is empty/malformed: do not fallback to raw prompt-like text.
                # This avoids leaking wrapper/instruction echoes into public timeline.
                _trace_turn(
                    key,
                    "pass(no_public_tagged)",
                    f"turn_id={ctx.turn_id} mode={ctx.mode.value} status={env.status}",
                )
                return _tr_pass(reason="no_public_tagged", raw_reply=reply, envelope_status=env.status, meta=env.meta)
            else:
                public_reply, private_reply = _split_public_private_reply(reply)
            public_reply = core.normalize_text(public_reply) or reply
            private_reply = core.normalize_text(private_reply)
            public_reply = _strip_leading_status_noise(public_reply) or public_reply
            private_reply = _strip_leading_status_noise(private_reply) or private_reply
            pub_key_len = len(re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", public_reply.lower()))
            pri_key_len = len(re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", private_reply.lower()))
            if private_reply and pri_key_len >= 28 and (pub_key_len < 10 or _TRIVIAL_PUBLIC_PAT.match(public_reply)):
                # Some UIs/models occasionally put the full answer under "private" and keep a trivial public stub.
                public_reply = private_reply
                private_reply = ""
            recent_self: list[str] = []
            if key in {"doubao", "qwen"}:
                recent_self = self._recent_public_replies(key, limit=12)
                stripped = _strip_reply_history_echo(public_reply, recent_self)
                if stripped:
                    public_reply = stripped
                public_reply = _strip_instruction_echo(public_reply, instruction)
                if not public_reply and private_reply:
                    # Some providers may place valid final text in private channel while public part echoes prompt.
                    public_reply = _strip_instruction_echo(private_reply, instruction)
                    if public_reply:
                        private_reply = ""
                sanitized_public = _sanitize_forward_payload(public_reply)
                if sanitized_public:
                    public_reply = sanitized_public
                elif visibility == "public" and not record_reply:
                    retry_timeout = 20 if key == "doubao" else 12
                    # sanitize returned empty => likely prompt-echo/status-only content.
                    # Retry once with short extra wait; still empty => PASS (do not fallback to raw leaked text).
                    try:
                        retry2 = core.normalize_text(ad.wait_reply_and_extract(before, timeout_s=retry_timeout))
                    except Exception:
                        retry2 = ""
                    if retry2:
                        r2_pub, _ = _split_public_private_reply(retry2)
                        r2_pub = core.normalize_text(r2_pub) or retry2
                        r2_pub = _strip_instruction_echo(r2_pub, instruction)
                        r2_pub_s = _sanitize_forward_payload(r2_pub)
                        if r2_pub_s:
                            r2_pub = r2_pub_s
                            r2_pub = _strip_leading_status_noise(r2_pub) or r2_pub
                            r2_pub = _dedupe_public_reply(r2_pub) or r2_pub
                            r2_pub = _pick_best_semantic_fragment(r2_pub) or r2_pub
                            if (
                                r2_pub
                                and not _looks_prompt_leak_reply(r2_pub)
                                and not _looks_unfinished_public_reply(r2_pub)
                                and not _looks_like_suggestion_chip_reply(r2_pub)
                            ):
                                public_reply = r2_pub
                                sanitized_public = r2_pub
                    if not sanitized_public:
                        # Last-chance rescue from direct DOM candidate.
                        # Helps when Doubao/Qwen first expose controls, then finalized text.
                        try:
                            raw_last = core.normalize_text(getattr(ad, "_extract_last_reply_candidate", lambda: "")())
                        except Exception:
                            raw_last = ""
                        if raw_last:
                            raw_last = _strip_instruction_echo(raw_last, instruction)
                            raw_last = _sanitize_forward_payload(raw_last) or raw_last
                            raw_last = _strip_leading_status_noise(raw_last) or raw_last
                            raw_last = _dedupe_public_reply(raw_last) or raw_last
                            raw_last = _strip_trailing_solicit_line(raw_last) or raw_last
                            if key in {"qwen", "doubao"}:
                                raw_last = _pick_best_semantic_fragment(raw_last) or raw_last
                            if (
                                raw_last
                                and not _looks_prompt_leak_reply(raw_last)
                                and not _looks_unfinished_public_reply(raw_last)
                                and not _looks_like_suggestion_chip_reply(raw_last)
                            ):
                                public_reply = raw_last
                                sanitized_public = raw_last
                    if not sanitized_public and key == "doubao":
                        # Doubao may briefly return control-text first; resend once to obtain final block.
                        try:
                            ad.send_user_text(prompt)
                            retry3 = core.normalize_text(ad.wait_reply_and_extract(before, timeout_s=20))
                        except Exception:
                            retry3 = ""
                        if retry3:
                            r3_pub, _ = _split_public_private_reply(retry3)
                            r3_pub = core.normalize_text(r3_pub) or retry3
                            r3_pub = _strip_instruction_echo(r3_pub, instruction)
                            r3_pub = _sanitize_forward_payload(r3_pub) or r3_pub
                            r3_pub = _strip_leading_status_noise(r3_pub) or r3_pub
                            r3_pub = _dedupe_public_reply(r3_pub) or r3_pub
                            r3_pub = _strip_trailing_solicit_line(r3_pub) or r3_pub
                            r3_pub = _pick_best_semantic_fragment(r3_pub) or r3_pub
                            if (
                                r3_pub
                                and not _looks_prompt_leak_reply(r3_pub)
                                and not _looks_unfinished_public_reply(r3_pub)
                                and not _looks_like_suggestion_chip_reply(r3_pub)
                            ):
                                public_reply = r3_pub
                                sanitized_public = r3_pub
                    if not sanitized_public:
                        if strict_token and re.search(
                            rf"(?i)(?:^|[^A-Za-z0-9_-]){re.escape(strict_token)}(?:[^A-Za-z0-9_-]|$)",
                            core.normalize_text(public_reply or ""),
                        ):
                            public_reply = strict_token
                            sanitized_public = strict_token
                    if not sanitized_public:
                        _trace_turn(key, "pass(prompt_echo)", public_reply)
                        return _tr_pass(
                            reason="prompt_echo",
                            raw_reply=reply,
                            private_text=private_reply,
                            envelope_status=(env.status if "env" in locals() else ""),
                            meta=(env.meta if "env" in locals() else None),
                        )
            if key == "gemini":
                # Gemini occasionally appends nav/suggestion tails in one block.
                # Apply the same sanitizer pass before final checks.
                sanitized_g = _sanitize_forward_payload(public_reply)
                if sanitized_g:
                    public_reply = sanitized_g
                elif visibility == "public" and not record_reply:
                    # sanitize-empty means we likely extracted only UI/prompt echo.
                    # Force retry path below instead of keeping the polluted raw block.
                    public_reply = ""
                if visibility == "public" and not record_reply and _looks_prompt_leak_reply(public_reply):
                    try:
                        retry_g = core.normalize_text(ad.wait_reply_and_extract(before, timeout_s=18))
                    except Exception:
                        retry_g = ""
                    if retry_g:
                        rg_pub, rg_pri = _split_public_private_reply(retry_g)
                        rg_pub = core.normalize_text(rg_pub) or retry_g
                        rg_pub = _strip_instruction_echo(rg_pub, instruction)
                        rg_pub = _sanitize_forward_payload(rg_pub) or rg_pub
                        rg_pub = _strip_leading_status_noise(rg_pub) or rg_pub
                        rg_pub = _dedupe_public_reply(rg_pub) or rg_pub
                        rg_pub = _strip_trailing_solicit_line(rg_pub) or rg_pub
                        if rg_pub and not _looks_prompt_leak_reply(rg_pub):
                            public_reply = rg_pub
                            if rg_pri:
                                private_reply = _strip_leading_status_noise(core.normalize_text(rg_pri))
                    if _looks_prompt_leak_reply(public_reply):
                        _trace_turn(key, "pass(gemini_prompt_echo)", public_reply)
                        return _tr_pass(
                            reason="gemini_prompt_echo",
                            raw_reply=reply,
                            private_text=private_reply,
                            envelope_status=(env.status if "env" in locals() else ""),
                            meta=(env.meta if "env" in locals() else None),
                        )
            public_reply = _strip_group_chatter_boilerplate(public_reply) or public_reply
            public_reply = _strip_leading_status_noise(public_reply) or public_reply
            public_reply = _dedupe_public_reply(public_reply) or public_reply
            public_reply = _strip_trailing_solicit_line(public_reply) or public_reply
            if key in {"qwen", "doubao"}:
                # Aggressive fragment picking is mainly needed for noisy web UIs.
                # For Gemini/ChatGPT/DeepSeek, keep full public semantic blocks.
                public_reply = _pick_best_semantic_fragment(public_reply) or public_reply
            public_reply = _strip_trailing_solicit_line(public_reply) or public_reply
            if _looks_provider_error_reply(public_reply):
                self.state.add_system(f"{m.name} 本轮失败：网页端返回错误提示")
                _trace_turn(key, "provider_error", public_reply)
                return _tr_error(
                    TurnErrorType.PROVIDER_ERROR,
                    msg="provider error reply",
                    raw_reply=reply,
                    parsed=public_reply,
                    private_text=private_reply,
                    envelope_status=(env.status if "env" in locals() else ""),
                    meta=(env.meta if "env" in locals() else None),
                )
            if key == "qwen" and _QWEN_STATUS_ONLY_PAT.match(public_reply or ""):
                # Qwen occasionally exposes "已完成思考/已经完成" status pills as text.
                # Retry once for final answer block before giving up.
                retry_timeout = 22 if (visibility == "public" and not record_reply) else 24
                try:
                    retry_q = core.normalize_text(ad.wait_reply_and_extract(before, timeout_s=retry_timeout))
                except Exception:
                    retry_q = ""
                if retry_q:
                    retry_pub, retry_pri = _split_public_private_reply(retry_q)
                    retry_pub = core.normalize_text(retry_pub) or core.normalize_text(retry_q)
                    retry_pub = _strip_leading_status_noise(retry_pub) or retry_pub
                    retry_pub = _strip_instruction_echo(retry_pub, instruction)
                    retry_pub = _sanitize_forward_payload(retry_pub) or retry_pub
                    retry_pub = _strip_leading_status_noise(retry_pub) or retry_pub
                    retry_pub = _dedupe_public_reply(retry_pub) or retry_pub
                    retry_pub = _pick_best_semantic_fragment(retry_pub) or retry_pub
                    if retry_pub and not _QWEN_STATUS_ONLY_PAT.match(retry_pub):
                        public_reply = retry_pub
                        if retry_pri:
                            private_reply = _strip_leading_status_noise(core.normalize_text(retry_pri))
                if key == "qwen" and _QWEN_STATUS_ONLY_PAT.match(public_reply or ""):
                    if visibility == "public" and not record_reply:
                        _trace_turn(key, "warn(qwen_status_only)", public_reply)
                        # Do not pass immediately: allow the next low-value/unfinished
                        # retry branch to fetch the finalized text once more.
                        public_reply = ""
                    else:
                        public_reply = "（未能提取到有效回复）"
            if key in {"qwen", "doubao"} and visibility == "public" and not record_reply:
                if _LOW_VALUE_PROCESS_PAT.match(public_reply or "") or _looks_unfinished_public_reply(public_reply):
                    retry_timeout2 = 18 if key == "qwen" else 16
                    try:
                        retry_lv = core.normalize_text(ad.wait_reply_and_extract(before, timeout_s=retry_timeout2))
                    except Exception:
                        retry_lv = ""
                    if retry_lv:
                        r_pub, r_pri = _split_public_private_reply(retry_lv)
                        r_pub = core.normalize_text(r_pub) or core.normalize_text(retry_lv)
                        r_pub = _strip_leading_status_noise(r_pub) or r_pub
                        r_pub = _strip_instruction_echo(r_pub, instruction)
                        r_pub = _sanitize_forward_payload(r_pub) or r_pub
                        r_pub = _dedupe_public_reply(r_pub) or r_pub
                        r_pub = _strip_trailing_solicit_line(r_pub) or r_pub
                        if key in {"qwen", "doubao"}:
                            r_pub = _pick_best_semantic_fragment(r_pub) or r_pub
                        if (
                            r_pub
                            and not _LOW_VALUE_PROCESS_PAT.match(r_pub)
                            and not _looks_unfinished_public_reply(r_pub)
                            and not _looks_prompt_leak_reply(r_pub)
                        ):
                            public_reply = r_pub
                            if r_pri:
                                private_reply = _strip_leading_status_noise(core.normalize_text(r_pri))
            if not core.normalize_text(public_reply):
                if visibility == "public" and not record_reply:
                    _trace_turn(key, "pass(empty_public)", "")
                    return _tr_pass(
                        reason="empty_public",
                        raw_reply=reply,
                        private_text=private_reply,
                        envelope_status=(env.status if "env" in locals() else ""),
                        meta=(env.meta if "env" in locals() else None),
                    )
                public_reply = "（未能提取到有效回复）"
            if key in {"qwen", "gemini"} and visibility == "public" and not record_reply:
                if _looks_stale_extracted_reply(public_reply, before, before_last_reply=before_last_reply):
                    if key == "gemini":
                        # Gemini sometimes yields a stale card first; allow one extra short fetch before skipping.
                        retry_stale = ""
                        try:
                            retry_stale = core.normalize_text(ad.wait_reply_and_extract(before, timeout_s=14))
                        except Exception:
                            retry_stale = ""
                        if retry_stale:
                            rs_pub, rs_pri = _split_public_private_reply(retry_stale)
                            rs_pub = core.normalize_text(rs_pub) or retry_stale
                            rs_pub = _strip_instruction_echo(rs_pub, instruction)
                            rs_pub = _sanitize_forward_payload(rs_pub) or rs_pub
                            rs_pub = _strip_leading_status_noise(rs_pub) or rs_pub
                            rs_pub = _dedupe_public_reply(rs_pub) or rs_pub
                            rs_pub = _strip_trailing_solicit_line(rs_pub) or rs_pub
                            if rs_pub and not _looks_stale_extracted_reply(
                                rs_pub, before, before_last_reply=before_last_reply
                            ):
                                public_reply = rs_pub
                                if rs_pri:
                                    private_reply = _strip_leading_status_noise(core.normalize_text(rs_pri))
                        if _looks_stale_extracted_reply(public_reply, before, before_last_reply=before_last_reply):
                            # One resend fallback for Gemini stale snapshots.
                            # Trade a bit more token usage for much better first-turn reliability.
                            try:
                                ad.send_user_text(prompt)
                                resend_g = core.normalize_text(ad.wait_reply_and_extract(before, timeout_s=22))
                            except Exception:
                                resend_g = ""
                            if resend_g:
                                g_pub, g_pri = _split_public_private_reply(resend_g)
                                g_pub = core.normalize_text(g_pub) or resend_g
                                g_pub = _strip_instruction_echo(g_pub, instruction)
                                g_pub = _sanitize_forward_payload(g_pub) or g_pub
                                g_pub = _strip_leading_status_noise(g_pub) or g_pub
                                g_pub = _dedupe_public_reply(g_pub) or g_pub
                                g_pub = _strip_trailing_solicit_line(g_pub) or g_pub
                                if g_pub and not _looks_stale_extracted_reply(
                                    g_pub, before, before_last_reply=before_last_reply
                                ):
                                    public_reply = g_pub
                                    if g_pri:
                                        private_reply = _strip_leading_status_noise(core.normalize_text(g_pri))
                    if _looks_stale_extracted_reply(public_reply, before, before_last_reply=before_last_reply):
                        _trace_turn(key, "pass(stale_snapshot)", public_reply)
                        return _tr_pass(
                            reason="stale_snapshot",
                            raw_reply=reply,
                            private_text=private_reply,
                            envelope_status=(env.status if "env" in locals() else ""),
                            meta=(env.meta if "env" in locals() else None),
                        )
            if key in {"doubao", "qwen"} and visibility == "public" and not record_reply:
                if _looks_unfinished_public_reply(public_reply):
                    _trace_turn(key, "pass(unfinished)", public_reply)
                    return _tr_pass(reason="unfinished", raw_reply=reply, private_text=private_reply)
                if _TRIVIAL_PUBLIC_PAT.match(public_reply):
                    _trace_turn(key, "pass(trivial)", public_reply)
                    return _tr_pass(reason="trivial", raw_reply=reply, private_text=private_reply)
                if _looks_like_suggestion_chip_reply(public_reply):
                    _trace_turn(key, "pass(chip_reply)", public_reply)
                    return _tr_pass(reason="chip_reply", raw_reply=reply, private_text=private_reply)
                compact_len = len(_line_dedupe_key(public_reply))
                if compact_len < 4:
                    _trace_turn(key, "pass(short)", public_reply)
                    return _tr_pass(reason="short", raw_reply=reply, private_text=private_reply)
                if key in {"doubao", "qwen"}:
                    min_klen = 12 if key == "doubao" else 10
                    if compact_len >= min_klen and _is_near_duplicate_reply(public_reply, recent_self):
                        _trace_turn(key, "pass(near_duplicate)", public_reply)
                        return _tr_pass(reason="near_duplicate", raw_reply=reply, private_text=private_reply)
                cur_key = _line_dedupe_key(public_reply)
                if cur_key and len(cur_key) >= 14:
                    for old in recent_self:
                        if cur_key == _line_dedupe_key(old):
                            # In group mode, duplicated old text is usually stale extraction; skip this turn.
                            _trace_turn(key, "pass(history_duplicate)", public_reply)
                            return _tr_pass(reason="history_duplicate", raw_reply=reply, private_text=private_reply)
            if visibility == "public" and not record_reply and _looks_protocol_placeholder_public_reply(public_reply):
                _trace_turn(key, "pass(protocol_placeholder)", public_reply)
                return _tr_pass(reason="protocol_placeholder", raw_reply=reply, private_text=private_reply)
            if key == "qwen" and re.search(
                r"(internal server error|连接到[^\\n]{0,40}出现问题|网络错误|请求失败|暂时不可用)",
                public_reply,
                re.I,
            ):
                self.state.add_system(f"{m.name} 本轮失败：{_clip_text(public_reply, 140)}")
                return _tr_error(
                    TurnErrorType.PROVIDER_ERROR,
                    msg="qwen provider error",
                    raw_reply=reply,
                    parsed=public_reply,
                    private_text=private_reply,
                    envelope_status=(env.status if "env" in locals() else ""),
                    meta=(env.meta if "env" in locals() else None),
                )
            if visibility == "public":
                public_reply = _compact_public_reply(
                    public_reply,
                    max_chars=GROUP_PUBLIC_REPLY_MAX_CHARS,
                    max_lines=GROUP_PUBLIC_REPLY_MAX_LINES,
                    max_sentences=GROUP_PUBLIC_REPLY_MAX_SENTENCES,
                )
            try:
                if ctx.mode == Mode.EVIDENCE:
                    if self._looks_like_disagreement(public_reply) and not has_valid_evidence_hook(private_reply or ""):
                        _trace_turn(
                            key,
                            "pass(no_valid_evidence_hook)",
                            f"turn_id={ctx.turn_id} mode={ctx.mode.value}\n{public_reply}",
                        )
                        public_reply = "[PASS]"
            except Exception:
                pass

            if record_reply:
                self.state.add_message(
                    "model",
                    m.name,
                    public_reply,
                    visibility=visibility,
                    model_key=key,
                )
            if private_reply and private_reply != public_reply:
                self.state.add_message(
                    "model",
                    f"{m.name}·内心",
                    _clip_text(private_reply, 1200),
                    visibility="shadow",
                    model_key=key,
                    shadow_scope="local",
                )
            try:
                tid = getattr(ctx, "turn_id", 0) if "ctx" in locals() else 0
                if defer_final_receipt:
                    status_hint = (
                        "PASS" if self._is_pass_reply(public_reply)
                        else ("HAS_PUBLIC" if (public_reply and public_reply.strip()) else "NO_PUBLIC")
                    )
                    _trace_turn(
                        key,
                        "receipt_provisional",
                        json.dumps({"turn_id": tid, "status_hint": status_hint}, ensure_ascii=False),
                    )
                else:
                    if self._is_pass_reply(public_reply):
                        rec = Receipt(turn_id=tid, status=ReceiptStatus.PASS_)
                    elif public_reply and public_reply.strip():
                        rec = Receipt(turn_id=tid, status=ReceiptStatus.ACCEPT)
                    else:
                        rec = Receipt(turn_id=tid, status=ReceiptStatus.REJECT, reason=RejectReason.NO_PUBLIC_TAG)
                    self._last_receipt_by_model[(key or "").strip().lower()] = rec
                    _trace_turn(key, "receipt_final", json.dumps(rec.to_dict(), ensure_ascii=False))
            except Exception:
                pass
            _trace_turn(
                key,
                "final",
                f"turn_id={ctx.turn_id} mode={ctx.mode.value}\n{public_reply}",
                elapsed_s=(time.time() - turn_start),
            )
            self.state.set_model_runtime_state(key, "IDLE", reason="turn_ok", reset_fail=True, turn_id=(ctx.turn_id if ctx else 0))
            return _tr_success(
                public_reply,
                raw_reply=reply,
                private_text=private_reply,
                envelope_status=(env.status if "env" in locals() else ""),
                meta=(env.meta if "env" in locals() else None),
                elapsed_s=(time.time() - turn_start),
            )
        except Exception as exc:
            self.state.add_system(f"{m.name} 本轮失败：{exc}")
            cls = self._classify_worker_failure(exc)
            if cls == "captcha":
                self.state.set_model_runtime_state(key, "NEEDS_HUMAN", reason=f"turn:{cls}", fail_delta=1, turn_id=(ctx.turn_id if ctx else 0))
                self.state.add_system(f"{m.name} 触发风控/验证码：请在浏览器中手动处理后点击恢复。")
            elif cls == "profile_locked":
                self.state.set_model_runtime_state(key, "NEEDS_HUMAN", reason=f"turn:{cls}", fail_delta=1, turn_id=(ctx.turn_id if ctx else 0))
                self.state.add_system(f"{m.name} 浏览器配置目录可能被占用。请关闭该模型相关浏览器窗口后点击恢复。")
            elif cls in {"timeout", "stale", "selector_miss"}:
                self.state.set_model_runtime_state(
                    key,
                    "COOLING_DOWN",
                    reason=f"turn:{cls}",
                    fail_delta=1,
                    cooldown_s=30 if cls != "timeout" else 20,
                    turn_id=(ctx.turn_id if ctx else 0),
                )
            else:
                self.state.set_model_runtime_state(key, "STALE", reason=f"turn:{cls}", fail_delta=1, turn_id=(ctx.turn_id if ctx else 0))
            try:
                if "ctx" in locals():
                    tid = getattr(ctx, "turn_id", 0)
                    if not defer_final_receipt:
                        rec = Receipt(turn_id=tid, status=ReceiptStatus.REJECT, reason=RejectReason.PARSE_FAIL)
                        self._last_receipt_by_model[(key or "").strip().lower()] = rec
                        _trace_turn(key, "receipt_final", json.dumps(rec.to_dict(), ensure_ascii=False))
                    else:
                        _trace_turn(
                            key,
                            "receipt_provisional",
                            json.dumps({"turn_id": tid, "status_hint": "ERROR"}, ensure_ascii=False),
                        )
            except Exception:
                pass
            et = TurnErrorType.TIMEOUT if "timeout" in str(exc).lower() else TurnErrorType.PARSE_FAIL
            return _tr_error(et, msg=str(exc))

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

    def _mention_candidates_line(self, keys: list[str], *, exclude_key: Optional[str] = None) -> str:
        tags: list[str] = []
        for key in keys:
            if exclude_key and key == exclude_key:
                continue
            m = self.state.get_model(key)
            name = core.normalize_text(m.name if m else key)
            if not name:
                continue
            tags.append(name)
        if not tags:
            return ""
        return "可回应对象：" + " / ".join(tags) + "（可直接写名字，不必使用@）。"

    @staticmethod
    def _should_sync_shadow_to_peers(shadow_scope: str) -> bool:
        return (shadow_scope or "shared").strip().lower() == "shared"

    @staticmethod
    def _check_round_packet_hash_consistency(seen: dict[str, str], model_key: str, packet_hash: str) -> tuple[bool, str]:
        h = (packet_hash or "").strip()
        if not h:
            return True, ""
        if not seen:
            seen[(model_key or "").strip().lower()] = h
            return True, h
        expected = next(iter(seen.values()))
        if h != expected:
            return False, expected
        seen[(model_key or "").strip().lower()] = h
        return True, expected

    def _reject_reason_from_turn_result(self, result: TurnResult) -> RejectReason:
        et = result.error_type
        rr = map_turn_error_to_reject_reason(et)
        if et in {TurnErrorType.UNKNOWN, None}:
            _trace_turn(
                result.model_key or "unknown",
                "WARN(reject_reason_unmapped)",
                f"error_type={et.value if isinstance(et, TurnErrorType) else et} error_msg={result.error_msg or ''}",
            )
        return rr

    def _apply_outer_turn_guards(
        self,
        *,
        key: str,
        result: TurnResult,
        round_no: int,
        round_packet_hash_seen: dict[str, str],
        authoritative_packet_hash: str,
        enable_packet_hash: bool = True,
        enable_leak: bool = True,
    ) -> TurnResult:
        def _guard_meta_tokens(r: TurnResult) -> TurnResult:
            meta = r.meta or {}
            pkt = core.normalize_text(str(meta.get("packet_hash") or ""))
            ack = core.normalize_text(str(meta.get("ack") or meta.get("ack_out") or ""))
            ack_in = core.normalize_text(str(meta.get("ack_in") or ""))
            if not is_valid_packet_hash_token(pkt):
                _trace_turn(key, "ERROR(protocol_meta_invalid)", f"bad packet_hash={pkt!r} meta={meta}")
                return error_like(r, error_type=TurnErrorType.PARSE_FAIL, error_msg="invalid_meta_packet_hash")
            if not is_valid_ack_token(ack):
                _trace_turn(key, "ERROR(protocol_meta_invalid)", f"bad ack/ack_out={ack!r} meta={meta}")
                return error_like(r, error_type=TurnErrorType.PARSE_FAIL, error_msg="invalid_meta_ack")
            if not is_valid_ack_token(ack_in):
                _trace_turn(key, "ERROR(protocol_meta_invalid)", f"bad ack_in={ack_in!r} meta={meta}")
                return error_like(r, error_type=TurnErrorType.PARSE_FAIL, error_msg="invalid_meta_ack_in")
            return r

        def _guard_packet_hash(r: TurnResult) -> TurnResult:
            pkt_ok, pkt_expected = self._check_round_packet_hash_consistency(
                round_packet_hash_seen, key, r.packet_hash or authoritative_packet_hash
            )
            if pkt_ok:
                return r
            self.state.add_system(f"广播包不一致告警：{key} 收到异常 packet_hash，已隔离本轮输出。")
            _trace_turn(
                key,
                "ERROR(packet_hash_mismatch)",
                (
                    f"round_no={round_no} turn_id_hint={r.turn_id} "
                    f"expected={pkt_expected} got={r.packet_hash or authoritative_packet_hash}"
                ),
            )
            return error_like(r, error_type=TurnErrorType.HASH_MISMATCH, error_msg="packet_hash_mismatch")

        def _guard_leak(r: TurnResult) -> TurnResult:
            if _looks_prompt_leak_reply(r.public_text):
                _trace_turn(key, "ERROR(loop_prompt_leak_guard)", r.public_text)
                return error_like(r, error_type=TurnErrorType.LEAK, error_msg="prompt leak after outer clean")
            return r

        guard_fns = []
        guard_fns.append(_guard_meta_tokens)
        if enable_packet_hash:
            guard_fns.append(_guard_packet_hash)
        if enable_leak:
            guard_fns.append(_guard_leak)
        return apply_turn_guards(result, *guard_fns)

    @staticmethod
    def _looks_like_disagreement(text: str) -> bool:
        raw = core.normalize_text(text).lower()
        if not raw:
            return False
        strong = [
            "不同意",
            "不认同",
            "不成立",
            "站不住脚",
            "我反对",
            "反驳",
            "你忽略",
            "你高估",
            "你低估",
            "i disagree",
            "not true",
            "incorrect",
            "counterpoint",
        ]
        weak = [
            "但是",
            "然而",
            "不过",
            "相反",
            "but",
            "however",
            "yet",
        ]
        strong_hits = sum(1 for x in strong if x in raw)
        weak_hits = sum(1 for x in weak if x in raw)
        return strong_hits >= 1 or weak_hits >= 2

    @staticmethod
    def _is_pass_reply(text: str) -> bool:
        t = core.normalize_text(text).lower()
        if not t:
            return True
        compact = re.sub(r"\s+", "", t)
        return compact in {
            "pass",
            "[pass]",
            "skip",
            "[skip]",
            "旁听",
            "暂不加入",
            "不加入",
            "已完成",
            "已经完成",
            "done",
            "ok",
            "好的",
            "收到",
        }

    @staticmethod
    def _infer_group_style(text: str) -> str:
        # Disable topic-specific hard mode in the core group engine.
        # Group chat should stay general-purpose by default.
        return ""

    @staticmethod
    def _is_continue_signal(text: str) -> bool:
        t = core.normalize_text(text).lower()
        if not t:
            return False
        compact = re.sub(r"\s+", "", t)
        return bool(
            re.fullmatch(
                r"(继续|继续吧|继续一下|继续进行|继续聊|继续讨论|接着|接着来|接下去|接下去吧)",
                compact,
            )
        )

    @staticmethod
    def _looks_like_idiom_payload(text: str) -> bool:
        t = core.normalize_text(text)
        if not t:
            return False
        if t == "[PASS]":
            return True
        return bool(re.match(r"^[\u4e00-\u9fff]{4}\n@[\w\u4e00-\u9fff-]{1,24}$", t))

    def _infer_group_style_from_recent(self) -> str:
        # Disable sticky topic-mode inference for now.
        return ""

    def _coerce_idiom_reply(
        self,
        text: str,
        *,
        speaker_key: str,
        all_keys: list[str],
        preferred_key: Optional[str] = None,
    ) -> str:
        raw = core.normalize_text(text)
        if not raw:
            return ""

        # Strict idiom extraction:
        # - only accept a standalone 4-char Chinese line (or "xxxx@name" line),
        # - do not slice arbitrary 4-char fragments from long sentences.
        banned = {"成语接龙", "群主插话", "等待接龙", "回应群主", "继续接龙", "接龙规则", "指出违规", "继续成语"}
        idiom = ""
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        for ln in reversed(lines):
            s = re.sub(r"\s+", "", ln)
            if not s:
                continue
            m0 = re.match(r"^([\u4e00-\u9fff]{4})$", s)
            if m0:
                cand = m0.group(1)
                if cand not in banned:
                    idiom = cand
                    break
            m1 = re.match(r"^([\u4e00-\u9fff]{4})@[\w\u4e00-\u9fff-]{1,24}$", s)
            if m1:
                cand = m1.group(1)
                if cand not in banned:
                    idiom = cand
                    break
            m2 = re.match(r"^(?:接成语|接龙|我接|接)[:：]?([\u4e00-\u9fff]{4})$", s)
            if m2:
                cand = m2.group(1)
                if cand not in banned:
                    idiom = cand
                    break
        if not idiom:
            return "[PASS]"

        peers: list[tuple[str, str]] = []
        for key in all_keys:
            if key == speaker_key:
                continue
            m = self.state.get_model(key)
            name = core.normalize_text(m.name if m else key)
            if name:
                peers.append((key, name))
        if not peers:
            return idiom

        mention = ""
        for token in re.findall(r"@([A-Za-z0-9_\-\u4e00-\u9fff]{1,24})", raw):
            t = core.normalize_text(token).lower()
            if not t:
                continue
            for key, name in peers:
                aliases = {
                    key.lower(),
                    core.normalize_text(name).lower(),
                    re.sub(r"[（(].*?[)）]", "", core.normalize_text(name)).strip().lower(),
                }
                if t in aliases:
                    mention = name
                    break
            if mention:
                break
        if not mention and preferred_key:
            for key, name in peers:
                if key == preferred_key:
                    mention = name
                    break
        if not mention:
            mention = peers[0][1]

        return f"{idiom}\n@{mention}".strip()

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

    def _latest_public_user_message(self) -> Optional[UiMessage]:
        msgs = self.state.get_all_messages()
        for msg in reversed(msgs):
            if msg.role != "user":
                continue
            if msg.visibility != "public":
                continue
            return msg
        return None

    def _host_interjected_since(self, before_user_mid: int) -> bool:
        latest = self._latest_public_user_message()
        latest_id = int(latest.id) if latest else 0
        return latest_id > int(before_user_mid or 0)

    def _recent_public_replies(self, key: str, limit: int = 3) -> list[str]:
        out: list[str] = []
        if not key or limit <= 0:
            return out
        msgs = self.state.get_all_messages()
        for msg in reversed(msgs):
            if msg.role != "model":
                continue
            if msg.visibility != "public":
                continue
            if msg.model_key != key:
                continue
            txt = _strip_private_thoughts(msg.text) or core.normalize_text(msg.text)
            txt = _sanitize_forward_payload(txt) or txt
            txt = _dedupe_public_reply(txt) or txt
            txt = core.normalize_text(txt)
            if not txt:
                continue
            out.append(txt)
            if len(out) >= limit:
                break
        return out

    def _estimate_recent_public_context_chars(self, *, limit_msgs: Optional[int] = None) -> int:
        """
        估算近期公开群聊上下文体量（字符数），用于动态放宽单轮等待时长。
        只看 user/model 的 public 消息，避免系统提示干扰。
        """
        max_msgs = int(limit_msgs if limit_msgs is not None else GROUP_TIMEOUT_CONTEXT_WINDOW_MSGS)
        max_msgs = max(1, max_msgs)
        per_msg_cap = max(40, int(GROUP_TIMEOUT_CONTEXT_PER_MSG_CHAR_CAP))
        total = 0
        used = 0
        msgs = self.state.get_all_messages()
        for msg in reversed(msgs):
            if msg.visibility != "public":
                continue
            if msg.role not in {"user", "model"}:
                continue
            t = _strip_private_thoughts(msg.text) or core.normalize_text(msg.text)
            t = core.normalize_text(t)
            if not t:
                continue
            total += min(per_msg_cap, len(t))
            used += 1
            if used >= max_msgs:
                break
        return total

    def _build_group_unseen_packet(
        self,
        *,
        after_id: int,
        floor_id: int = 0,
        max_msgs: int = GROUP_BROADCAST_MAX_MSGS,
        max_chars: int = GROUP_BROADCAST_MAX_CHARS,
    ) -> tuple[str, int, int]:
        """
        为单个模型构建“未读群消息增量包”。
        - 只包含 public 且 role in {user, model} 的消息。
        - 从 after_id 之后开始，按时间顺序发送。
        - 受 max_msgs/max_chars 限制；若超出，remaining>0 表示还有未同步消息。
        返回: (packet_text, delivered_upto_id, remaining_count)
        """
        start_after = max(int(after_id), int(floor_id) - 1)
        msgs = self.state.get_all_messages()
        unseen = [
            m
            for m in msgs
            if m.visibility == "public" and m.role in {"user", "model"} and int(m.id) > start_after
        ]
        if not unseen:
            return "", start_after, 0

        per_msg_cap = max(80, int(GROUP_BROADCAST_PER_MSG_CHAR_CAP))
        rows: list[str] = []
        used = 0
        delivered_upto = start_after
        delivered_count = 0

        for msg in unseen:
            speaker = "群主" if msg.role == "user" else core.normalize_text(msg.speaker)
            body = _pick_forward_payload(msg.text) or (_strip_private_thoughts(msg.text) or "")
            body = core.normalize_text(body)
            if not body:
                continue
            body = _clip_text(body.replace("\n", " "), per_msg_cap)
            row = f"[{msg.id}] {speaker}: {body}"
            cost = len(row) + 1
            # 至少保证送达 1 条，避免首条超长时被整包丢弃。
            if rows and (delivered_count >= max_msgs or used + cost > max_chars):
                break
            rows.append(row)
            used += cost
            delivered_count += 1
            delivered_upto = int(msg.id)
            if delivered_count >= max_msgs or used >= max_chars:
                break

        if not rows:
            first = unseen[0]
            speaker = "群主" if first.role == "user" else core.normalize_text(first.speaker)
            body = _pick_forward_payload(first.text) or (_strip_private_thoughts(first.text) or "")
            body = _clip_text(core.normalize_text(body).replace("\n", " "), per_msg_cap)
            rows = [f"[{first.id}] {speaker}: {body}"]
            delivered_upto = int(first.id)
            delivered_count = 1

        remaining = max(0, len(unseen) - delivered_count)
        return "\n".join(rows), delivered_upto, remaining

    def _seed_group_pending_ids(self, *, floor_id: int, target_key: str) -> list[int]:
        """
        为某个模型初始化“待广播消息队列”。
        - 包含本次群任务起点以来的 public user/model 消息；
        - 排除该模型自己发出的消息（避免自回显）。
        """
        out: list[int] = []
        floor = max(1, int(floor_id))
        key_norm = (target_key or "").strip().lower()
        for msg in self.state.get_all_messages():
            if msg.visibility != "public" or msg.role not in {"user", "model"}:
                continue
            mid = int(msg.id)
            if mid < floor:
                continue
            if msg.role == "model" and (msg.model_key or "").strip().lower() == key_norm:
                continue
            out.append(mid)
        return out

    def _build_group_pending_packet(
        self,
        *,
        model_key: str,
        pending_ids: list[int],
        max_msgs: int = GROUP_BROADCAST_MAX_MSGS,
        max_chars: int = GROUP_BROADCAST_MAX_CHARS,
    ) -> tuple[str, list[int], int]:
        """
        从“待广播消息队列”构造本轮分发包。
        返回: (packet_text, delivered_ids, remaining_count)
        - delivered_ids: 本轮已成功分发（或可安全丢弃）的消息 id。
        """
        if not pending_ids:
            return "", [], 0

        key_norm = (model_key or "").strip().lower()
        msg_map: dict[int, UiMessage] = {}
        for msg in self.state.get_all_messages():
            if msg.visibility != "public" or msg.role not in {"user", "model"}:
                continue
            msg_map[int(msg.id)] = msg

        per_msg_cap = max(80, int(GROUP_BROADCAST_PER_MSG_CHAR_CAP))
        rows: list[str] = []
        delivered_ids: list[int] = []
        used = 0

        for mid_raw in pending_ids:
            mid = int(mid_raw)
            msg = msg_map.get(mid)
            if msg is None:
                # 已不存在的消息视作可清理，避免队列阻塞。
                delivered_ids.append(mid)
                continue
            if msg.role == "model" and (msg.model_key or "").strip().lower() == key_norm:
                # 自己发过的消息无需回传给自己。
                delivered_ids.append(mid)
                continue

            speaker = "群主" if msg.role == "user" else core.normalize_text(msg.speaker)
            body = _pick_forward_payload(msg.text) or (_strip_private_thoughts(msg.text) or "")
            body = core.normalize_text(body)
            if not body:
                # 清洗后空内容（噪声/提示词泄漏）直接跳过并出队。
                delivered_ids.append(mid)
                continue
            body = _clip_text(body.replace("\n", " "), per_msg_cap)
            row = f"[{mid}] {speaker}: {body}"
            cost = len(row) + 1
            # 至少发送 1 条，避免首条超长导致整包丢弃。
            if rows and (len(rows) >= max_msgs or used + cost > max_chars):
                break
            rows.append(row)
            used += cost
            delivered_ids.append(mid)
            if len(rows) >= max_msgs or used >= max_chars:
                break

        remaining = max(0, len(pending_ids) - len(delivered_ids))
        return "\n".join(rows), delivered_ids, remaining

    def _build_group_recent_packet(
        self,
        *,
        floor_id: int,
        exclude_key: Optional[str],
        max_msgs: int = GROUP_BROADCAST_RECENT_MAX_MSGS,
        max_chars: int = GROUP_BROADCAST_RECENT_MAX_CHARS,
    ) -> str:
        """
        构建最近公开消息窗口（广播回放）。
        用于兜底：即使某轮未读包较小，模型也能看到最近群聊全貌。
        """
        floor = max(1, int(floor_id))
        ex = (exclude_key or "").strip().lower()
        candidates: list[UiMessage] = []
        for msg in self.state.get_all_messages():
            if msg.visibility != "public" or msg.role not in {"user", "model"}:
                continue
            if int(msg.id) < floor:
                continue
            if ex and msg.role == "model" and (msg.model_key or "").strip().lower() == ex:
                continue
            candidates.append(msg)
        if not candidates:
            return ""

        per_msg_cap = max(80, int(GROUP_BROADCAST_PER_MSG_CHAR_CAP))
        rows_rev: list[str] = []
        used = 0
        for msg in reversed(candidates):
            speaker = "群主" if msg.role == "user" else core.normalize_text(msg.speaker)
            body = _pick_forward_payload(msg.text) or (_strip_private_thoughts(msg.text) or "")
            body = core.normalize_text(body)
            if not body:
                continue
            body = _clip_text(body.replace("\n", " "), per_msg_cap)
            row = f"[{msg.id}] {speaker}: {body}"
            cost = len(row) + 1
            if rows_rev and (len(rows_rev) >= max_msgs or used + cost > max_chars):
                break
            rows_rev.append(row)
            used += cost
            if len(rows_rev) >= max_msgs or used >= max_chars:
                break
        if not rows_rev:
            return ""
        return "\n".join(reversed(rows_rev))

    def _drain_group_interjections(self, max_items: int = 20) -> list[str]:
        """
        在群聊自动轮转期间，抽取队列里的 send(group) 作为插话。
        其他动作先暂存再放回队列，保持“可插话”而不破坏现有串行架构。
        """
        picked: list[str] = []
        stash: list[dict[str, Any]] = []
        for _ in range(max_items):
            try:
                item = self.state.inbox.get_nowait()
            except queue.Empty:
                break
            kind = str(item.get("kind") or "")
            if kind == "send":
                tgt = str(item.get("target") or "").strip().lower()
                txt = core.normalize_text(str(item.get("text") or ""))
                if tgt == "group" and txt:
                    picked.append(txt)
                    continue
            stash.append(item)

        for item in stash:
            self.state.inbox.put(item)
        return picked

    def _purge_stale_group_sends(self, max_items: int = 80) -> None:
        """
        新群任务启动时清理遗留的 send(group) 动作，避免旧话题覆盖本轮用户消息。
        """
        stash: list[dict[str, Any]] = []
        for _ in range(max_items):
            try:
                item = self.state.inbox.get_nowait()
            except queue.Empty:
                break
            kind = str(item.get("kind") or "")
            if kind == "send":
                tgt = str(item.get("target") or "").strip().lower()
                if tgt == "group":
                    continue
            stash.append(item)
        for item in stash:
            self.state.inbox.put(item)

    @staticmethod
    def _build_shadow_sync_instruction(source_name: str, user_text: str, source_reply: str) -> str:
        user_part = _clip_text(user_text, 2000)
        source_public, _ = _split_public_private_reply(source_reply)
        reply_part = _clip_text(source_public or _strip_private_thoughts(source_reply), 2400)
        return (
            "你在同一个群聊中，下面是刚发生的一轮消息：\n\n"
            f"群主：{user_part}\n"
            f"{source_name}：{reply_part}\n\n"
            "请像正常群聊一样直接回复：先给结论，再补1-2条依据（可含数据/假设），信息充分但不跑题。\n"
            "若你这轮不想发言，只输出 [PASS]。\n"
            "可选：使用【对外】...【内心】...（内心不会公开）。"
        ).strip()

    @staticmethod
    def _build_observer_probe_instruction(source_name: str, source_reply: str, mention_line: str = "") -> str:
        source_public, _ = _split_public_private_reply(source_reply)
        reply_part = _clip_text(source_public or _strip_private_thoughts(source_reply), 2200)
        mention_part = (mention_line.strip() + "\n") if mention_line.strip() else ""
        return (
            "你在同一个群聊中，下面是刚发生的一条新消息：\n\n"
            f"{source_name}：{reply_part}\n\n"
            f"{mention_part}"
            "如果你想加入讨论，直接发一条短消息；如果暂时旁听，只输出 [PASS]。\n"
            "不要解释规则，不要输出思考过程。"
        ).strip()

    def _handle_send(self, action: dict[str, Any]) -> None:
        target = str(action.get("target") or "").strip().lower()
        text = core.normalize_text(str(action.get("text") or ""))
        if not text:
            self._safe_reply(action, {"ok": False, "error": "empty_text"})
            return

        selected_snapshot = self.state.selected_keys()
        # 用户点击“停止”后再次发送新消息，应恢复 worker 主循环继续工作。
        self.state.clear_stop()
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
            # 用户经常会“已勾选但未登录”，这里单独提示，避免误以为程序坏了。
            authed_selected = []
            for k in keys:
                m0 = self.state.get_model(k)
                if m0 and bool(getattr(m0, "authenticated", False)):
                    authed_selected.append(k)
            if not authed_selected:
                # If startup/login recheck is still running, wait briefly for the fresh auth snapshot.
                # Avoid false "all unauthenticated" right after restart.
                wait_deadline = time.time() + 12.0
                while time.time() < wait_deadline and not authed_selected:
                    waiting = False
                    for k in keys:
                        m0 = self.state.get_model(k)
                        if not m0:
                            continue
                        if _should_wait_for_login_recheck(
                            str(getattr(m0, "runtime_state", "")),
                            str(getattr(m0, "runtime_reason", "")),
                        ):
                            waiting = True
                            break
                    if not waiting:
                        break
                    time.sleep(1.0)
                    authed_selected = []
                    for k in keys:
                        m0 = self.state.get_model(k)
                        if m0 and bool(getattr(m0, "authenticated", False)):
                            authed_selected.append(k)
            if not authed_selected:
                selected_names = []
                for k in keys:
                    try:
                        mm = MODEL_METAS.get(k)
                        selected_names.append(mm.name if mm else k)
                    except Exception:
                        selected_names.append(k)
                joined = " / ".join(selected_names[:5]) if selected_names else "（未知）"
                self.state.add_system(
                    f"已启用但未检测到登录成功：{joined}。请先在左侧打开登录窗口并完成登录，再点【我已登录，重新检测】或 ↻ 恢复。"
                )
                self._safe_reply(action, {"ok": False, "error": "selected_but_not_authenticated"})
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
            action_shadow_scope = (str(action.get("shadow_scope") or "")).strip().lower()
            if action_shadow_scope not in {"shared", "local"}:
                action_shadow_scope = SHADOW_SCOPE_DEFAULT

        if visibility == "shadow":
            user_mid = self.state.add_message(
                "user",
                "用户",
                text,
                visibility=visibility,
                model_key=thread_key,
                shadow_scope=action_shadow_scope,
            )
        else:
            user_mid = self.state.add_message("user", "用户", text, visibility=visibility, model_key=thread_key)

        self._ensure_playwright()
        assert self._pw is not None

        # Start each new group task with a fresh Qwen web thread.
        # Reason: long in-page history can increase latency and leak stale topics.
        if target == "group":
            self._purge_stale_group_sends()
            self._deepseek_need_fresh_chat = True
            self._qwen_need_fresh_chat = True
            self._doubao_need_fresh_chat = True
            self._chatgpt_need_fresh_chat = True
            self._gemini_need_fresh_chat = True

        if target != "group":
            main_res = self._run_model_turn(keys[0], text, visibility=visibility, hidden_reply_hint=False)
            main_res = self._apply_outer_turn_guards(
                key=keys[0],
                result=main_res,
                round_no=0,
                round_packet_hash_seen={},
                authoritative_packet_hash="",
                enable_packet_hash=False,
                enable_leak=True,
            )
            if main_res.ok and main_res.public_text:
                # Freeze peer targets at send-time snapshot to avoid mid-turn toggle races causing wrong sync targets.
                peers = [k for k in selected_snapshot if k != target]
                if visibility == "shadow" and not self._should_sync_shadow_to_peers(action_shadow_scope):
                    peers = []
                if peers:
                    sm = self.state.get_model(target)
                    source_name = sm.name if sm else target
                    payload = _pick_forward_payload(main_res.public_text) or main_res.public_text
                    for peer in peers:
                        # 让每个旁听模型也有同一条“用户消息”，便于后续切换到该模型时上下文完整。
                        self.state.add_message(
                            "user",
                            "用户",
                            text,
                            visibility="shadow",
                            model_key=peer,
                            shadow_scope="shared",
                        )
                        instruction = self._build_shadow_sync_instruction(source_name, text, payload)
                        peer_res = self._run_model_turn(peer, instruction, visibility="shadow", hidden_reply_hint=True)
                        _ = self._apply_outer_turn_guards(
                            key=peer,
                            result=peer_res,
                            round_no=0,
                            round_packet_hash_seen={},
                            authoritative_packet_hash="",
                            enable_packet_hash=False,
                            enable_leak=True,
                        )
            self.state.set_status("idle")
            self._safe_reply(action, {"ok": True})
            return

        focus_keys: Optional[set[str]] = None
        focus_idle_rounds = 0
        silent_rounds = 0
        round_no = 0
        group_floor_id = max(1, int(user_mid or 1))
        group_pending_ids: dict[str, list[int]] = {
            k: self._seed_group_pending_ids(floor_id=group_floor_id, target_key=k) for k in selected_snapshot
        }

        def _enqueue_broadcast(mid: Optional[int], targets: list[str], *, exclude_key: Optional[str] = None) -> None:
            if not mid:
                return
            mid_i = int(mid)
            ex = (exclude_key or "").strip().lower()
            for tk in targets:
                tkn = (tk or "").strip().lower()
                if not tkn:
                    continue
                if ex and tkn == ex:
                    continue
                q = group_pending_ids.setdefault(tkn, [])
                if mid_i not in q:
                    q.append(mid_i)

        def _rebase_pending_for_topic(targets: list[str], *, latest_user_mid: Optional[int]) -> None:
            """
            群主插话后，裁剪待广播积压，优先让模型先处理新话题。
            避免“旧 backlog 太大导致看起来没听到群主插话”。
            """
            keep_n = max(2, int(TOPIC_INTERJECT_PENDING_KEEP))
            lm = int(latest_user_mid or 0)
            for tk in targets:
                # 从“最新群主插话”开始重建，硬切断旧话题 backlog。
                floor_local = lm if lm > 0 else group_floor_id
                seed = self._seed_group_pending_ids(floor_id=floor_local, target_key=tk)
                if lm > 0:
                    seed = [mid for mid in seed if int(mid) >= lm]
                if len(seed) > keep_n:
                    seed = seed[-keep_n:]
                if lm > 0 and lm not in seed:
                    seed.append(lm)
                group_pending_ids[tk] = seed

        active_user_instruction: Optional[str] = text
        active_user_hard_literal = _extract_expected_short_literal(active_user_instruction or "")
        if not active_user_hard_literal:
            active_user_hard_literal = _extract_expected_numeric_literal_if_requested(active_user_instruction or "")
        active_topic_floor_id = max(group_floor_id, int(user_mid or group_floor_id))
        active_user_force_rounds = _topic_lock_rounds(len(selected_snapshot))
        active_user_pending_models: set[str] = set(selected_snapshot)
        active_user_pending_attempts: dict[str, int] = {}
        active_user_strict_ok_models: set[str] = set()
        while True:
            if self.state.should_round_stop():
                self.state.add_system("已停止自动轮聊。")
                break

            queued_user_msgs = self._drain_group_interjections()
            if queued_user_msgs:
                now_selected = self.state.selected_keys()
                latest_mid = 0
                for extra in queued_user_msgs:
                    extra_mid = self.state.add_message("user", "用户", extra, visibility="public", model_key=None)
                    _enqueue_broadcast(extra_mid, now_selected)
                    latest_mid = int(extra_mid or latest_mid)
                _rebase_pending_for_topic(now_selected, latest_user_mid=latest_mid)
                active_user_instruction = queued_user_msgs[-1]
                active_user_hard_literal = _extract_expected_short_literal(active_user_instruction or "")
                if not active_user_hard_literal:
                    active_user_hard_literal = _extract_expected_numeric_literal_if_requested(active_user_instruction or "")
                active_topic_floor_id = max(group_floor_id, int(latest_mid or active_topic_floor_id))
                active_user_force_rounds = _topic_lock_rounds(len(now_selected))
                active_user_pending_models = set(now_selected)
                active_user_pending_attempts = {}
                active_user_strict_ok_models = set()
                focus_keys = None
                focus_idle_rounds = 0
                self.state.add_system("群主插话已加入当前讨论。")

            current_keys = self.state.selected_keys()
            if not current_keys:
                self.state.add_system("轮聊结束：当前没有已启用模型。")
                break
            for kk in current_keys:
                if kk not in group_pending_ids:
                    # 新加入群聊的模型从本次群任务起点开始补同步。
                    group_pending_ids[kk] = self._seed_group_pending_ids(floor_id=group_floor_id, target_key=kk)
            for kk in [x for x in list(group_pending_ids.keys()) if x not in current_keys]:
                group_pending_ids.pop(kk, None)
            if active_user_pending_models:
                active_user_pending_models = {x for x in active_user_pending_models if x in current_keys}
                for stale in [x for x in list(active_user_pending_attempts.keys()) if x not in active_user_pending_models]:
                    active_user_pending_attempts.pop(stale, None)
            if round_no >= 1 and len(current_keys) < 2:
                self.state.add_system("轮聊结束：当前仅 1 个模型，无法继续轮流发言。")
                break

            if focus_keys:
                focus_keys = {k for k in focus_keys if k in current_keys}
                if len(focus_keys) < 2:
                    focus_keys = None

            if focus_keys:
                pri = [k for k in current_keys if k in focus_keys]
                rest = [k for k in current_keys if k not in focus_keys]
                talk_keys = pri + rest
            else:
                talk_keys = list(current_keys)
            if len(current_keys) == 2 and talk_keys:
                # Two-model fairness: alternate start speaker based on latest public model message.
                # Avoid one side repeatedly speaking first when the other side is slow.
                last_pub = self._latest_public_model_message()
                if last_pub and last_pub.model_key in talk_keys:
                    lead = [x for x in talk_keys if x != last_pub.model_key]
                    follow = [x for x in talk_keys if x == last_pub.model_key]
                    if lead and follow:
                        talk_keys = lead + follow
            if len(current_keys) >= 2 and talk_keys:
                # Generic fairness: avoid same speaker starting consecutive rounds in larger groups.
                last_pub2 = self._latest_public_model_message()
                if last_pub2 and talk_keys[0] == (last_pub2.model_key or "") and len(talk_keys) > 1:
                    talk_keys = talk_keys[1:] + talk_keys[:1]

            round_no += 1
            any_turn_ok = False
            round_visible_replies = 0
            mention_switched = False
            round_interrupted_by_host = False
            round_context_chars = self._estimate_recent_public_context_chars()
            authoritative_packet_text, authoritative_packet_hash = self._build_authoritative_broadcast_packet()
            authoritative_ack_in = self._ack_in_from_public_timeline()
            round_packet_hash_seen: dict[str, str] = {}
            for k in talk_keys:
                if self.state.should_round_stop():
                    break

                mid_round_msgs = self._drain_group_interjections(max_items=8)
                if mid_round_msgs:
                    latest_mid = 0
                    for extra in mid_round_msgs:
                        extra_mid = self.state.add_message("user", "用户", extra, visibility="public", model_key=None)
                        _enqueue_broadcast(extra_mid, current_keys)
                        latest_mid = int(extra_mid or latest_mid)
                    _rebase_pending_for_topic(current_keys, latest_user_mid=latest_mid)
                    active_user_instruction = mid_round_msgs[-1]
                    active_user_hard_literal = _extract_expected_short_literal(active_user_instruction or "")
                    if not active_user_hard_literal:
                        active_user_hard_literal = _extract_expected_numeric_literal_if_requested(active_user_instruction or "")
                    active_topic_floor_id = max(group_floor_id, int(latest_mid or active_topic_floor_id))
                    active_user_force_rounds = _topic_lock_rounds(len(current_keys))
                    active_user_pending_models = set(current_keys)
                    active_user_pending_attempts = {}
                    active_user_strict_ok_models = set()
                    focus_keys = None
                    focus_idle_rounds = 0
                    self.state.add_system("群主插话已加入当前回合。")
                    round_interrupted_by_host = True
                    # 抢占：下一个完整回合优先处理群主新话题，避免旧话题继续刷屏。
                    break

                last_msg = self._latest_public_model_message(exclude_key=k)
                counterpart_key: Optional[str] = None
                if last_msg and last_msg.model_key and last_msg.model_key != k:
                    counterpart_key = last_msg.model_key
                strict_topic_phase = bool(
                    active_user_instruction
                    and (bool(active_user_hard_literal) or active_user_force_rounds > 0 or bool(active_user_pending_models))
                )
                pending_for_k = list(group_pending_ids.get(k, []))
                if strict_topic_phase and active_topic_floor_id > 0:
                    pending_for_k = [mid for mid in pending_for_k if int(mid) >= int(active_topic_floor_id)]
                    group_pending_ids[k] = list(pending_for_k)
                unseen_packet, delivered_ids, unseen_remaining = self._build_group_pending_packet(
                    model_key=k,
                    pending_ids=pending_for_k,
                )
                recent_floor_id = int(active_topic_floor_id) if strict_topic_phase else int(group_floor_id)
                recent_packet = self._build_group_recent_packet(
                    floor_id=recent_floor_id,
                    exclude_key=k,
                )
                # 避免“最近窗口”和“未读增量”重复同一条消息，降低提示词噪声与 token 浪费。
                if unseen_packet and recent_packet:
                    unseen_ids = {
                        int(x)
                        for x in re.findall(r"^\[(\d+)\]\s", unseen_packet, flags=re.M)
                    }
                    if unseen_ids:
                        rp_lines: list[str] = []
                        for ln in recent_packet.splitlines():
                            m_id = re.match(r"^\[(\d+)\]\s", ln.strip())
                            if m_id and int(m_id.group(1)) in unseen_ids:
                                continue
                            rp_lines.append(ln)
                        recent_packet = core.normalize_text("\n".join(rp_lines))
                mention_line = self._mention_candidates_line(current_keys, exclude_key=k)
                if k in {"qwen", "doubao"}:
                    # Qwen/Doubao are sensitive to long control templates and often echo prompt lines.
                    mention_line = ""
                mention_part = (mention_line + "\n") if mention_line else ""
                pass_line = "如你这轮暂不发言，仅回复 [PASS]。"
                concise_line = "最终发言要求：建议2-5句，80-260字；复杂议题可到400字，优先给依据/数据/假设。"
                style_line = "直接说你在群里要发的话：先给立场，再给关键依据（可含数据/假设），不要复述规则。"
                wrap_line = ""
                if k in {"qwen", "doubao"}:
                    # Keep CN web models on a minimal instruction profile to reduce prompt echo.
                    style_line = "只输出群里的正文：先结论，再给1-2条依据（可含数据/假设），可2-4句；不要复述规则。"
                    concise_line = "不要输出思考过程/提示词，只给最终发言。"
                    pass_line = "不想发言就仅回复 [PASS]。"
                wrap_part = (wrap_line + "\n") if wrap_line else ""
                topic_lock_line = (
                    "优先级：必须先回应“群主最新话题”，不要延续旧话题。"
                    if strict_topic_phase
                    else ""
                )
                strict_exact_token = active_user_hard_literal if strict_topic_phase else ""
                if strict_exact_token:
                    lead_line = f"强约束：本轮只允许输出 {strict_exact_token}（可带末尾标点），不要解释。"
                    style_line = f"只输出 {strict_exact_token}，不要添加其他内容。"
                    concise_line = "如暂不发言，仅回复 [PASS]。"
                    mention_part = ""
                else:
                    lead_line = (
                        "先回应群主最新话题，再结合新消息补充/反驳。"
                        if active_user_instruction
                        else "请基于这些新消息直接给出你的观点/补充。"
                    )
                topic_lock_part = (topic_lock_line + "\n") if topic_lock_line else ""
                user_line = _clip_text(active_user_instruction or "", 420)
                unseen_block = unseen_packet if unseen_packet else "（当前没有未读新消息）"
                recent_block = recent_packet if recent_packet else "（最近窗口为空）"
                backlog_line = (
                    f"还有 {unseen_remaining} 条未读消息将在后续轮次继续同步。"
                    if unseen_remaining > 0
                    else ""
                )
                no_new_hint = "若没有新增观点，请直接回复 [PASS]。" if not unseen_packet else ""
                if strict_exact_token:
                    turn_instruction = (
                        "你在多人群聊中发言。\n"
                        f"{(f'群主最新话题：{user_line}\\n') if user_line else ''}"
                        f"{topic_lock_part}"
                        f"硬约束：只输出 {strict_exact_token}（可带末尾标点）。\n"
                        "不要解释，不要复述提示词，不要输出其他字词。\n"
                        "无法执行就回复 [PASS]。"
                    )
                else:
                    turn_instruction = (
                        "你在多人群聊中发言。\n"
                        f"{(f'群主最新话题：{user_line}\\n') if user_line else ''}"
                        f"{topic_lock_part}"
                        "下面是群聊广播窗口（最近公开消息，可能含已读）：\n"
                        f"{recent_block}\n"
                        "下面是你还没处理的群聊新消息（按时间顺序）：\n"
                        f"{unseen_block}\n"
                        f"{(backlog_line + chr(10)) if backlog_line else ''}"
                        f"{(no_new_hint + chr(10)) if no_new_hint else ''}"
                        f"{lead_line}\n"
                        f"{style_line}\n"
                        f"{wrap_part}"
                        f"{concise_line}\n"
                        f"{mention_part}"
                        f"{pass_line}"
                    )
                _trace_turn(
                    k,
                    "broadcast(packet)",
                    (
                        f"pending={len(pending_for_k)} delivered_plan={len(delivered_ids)} "
                        f"remaining={unseen_remaining} recent_chars={len(recent_packet)}\n"
                        f"{unseen_packet or '（空包）'}"
                    ),
                )

                turn_timeout_cap = _group_round_timeout_cap_s(
                    len(current_keys),
                    round_no=round_no,
                    context_chars=round_context_chars,
                )
                if strict_topic_phase:
                    turn_timeout_cap += 8
                if k == "qwen":
                    turn_timeout_cap += 4
                latest_public_user_before_turn = self._latest_public_user_message()
                latest_public_user_before_turn_id = int(latest_public_user_before_turn.id) if latest_public_user_before_turn else 0
                turn_res = self._run_model_turn(
                    k,
                    turn_instruction,
                    visibility="public",
                    record_reply=False,
                    timeout_cap_s=turn_timeout_cap,
                    authoritative_packet_text=authoritative_packet_text,
                    authoritative_packet_hash=authoritative_packet_hash,
                    ack_in=authoritative_ack_in,
                )
                turn_res = self._apply_outer_turn_guards(
                    key=k,
                    result=turn_res,
                    round_no=round_no,
                    round_packet_hash_seen=round_packet_hash_seen,
                    authoritative_packet_hash=authoritative_packet_hash,
                )
                ok = turn_res.ok
                reply_text = turn_res.public_text
                def _finalize_turn_receipt_local(
                    status: ReceiptStatus,
                    reason: Optional[RejectReason] = None,
                    note: Optional[str] = None,
                ) -> None:
                    try:
                        self._finalize_receipt_for_model(k, status=status, reason=reason, note=note)
                    except Exception:
                        pass
                def _mark_delivered(ids: list[int]) -> None:
                    if not ids:
                        return
                    delivered_set = {int(x) for x in ids}
                    group_pending_ids[k] = [
                        mid for mid in group_pending_ids.get(k, []) if int(mid) not in delivered_set
                    ]
                turn_has_visible = bool(turn_res.ok and core.normalize_text(turn_res.public_text) and (not self._is_pass_reply(turn_res.public_text)))
                any_turn_ok = any_turn_ok or turn_has_visible
                if turn_res.ok and turn_res.public_text:
                    # If the host inserted a newer public message while this model was generating,
                    # discard the stale reply and let the next round process the newer topic first.
                    if self._host_interjected_since(latest_public_user_before_turn_id):
                        _finalize_turn_receipt_local(
                            ReceiptStatus.REJECT,
                            RejectReason.OFF_TOPIC,
                            "stale_after_host_interject",
                        )
                        _trace_turn(k, "pass(loop_stale_after_host_interject)", turn_res.public_text)
                        continue
                    clean_reply = _strip_private_thoughts(reply_text) or core.normalize_text(reply_text)
                    clean_reply = _strip_group_chatter_boilerplate(clean_reply) or clean_reply
                    clean_reply = _strip_instruction_echo_lines(clean_reply) or clean_reply
                    clean_reply = _sanitize_forward_payload(clean_reply) or clean_reply
                    clean_reply = _dedupe_public_reply(clean_reply) or clean_reply
                    clean_reply = _strip_trailing_solicit_line(clean_reply) or clean_reply
                    if k in {"qwen", "doubao"} or _looks_prompt_leak_reply(clean_reply):
                        clean_reply = _pick_best_semantic_fragment(clean_reply) or clean_reply
                    clean_reply = _strip_trailing_solicit_line(clean_reply) or clean_reply
                    clean_reply = _compact_public_reply(
                        clean_reply,
                        max_chars=GROUP_PUBLIC_REPLY_MAX_CHARS,
                        max_lines=GROUP_PUBLIC_REPLY_MAX_LINES,
                        max_sentences=GROUP_PUBLIC_REPLY_MAX_SENTENCES,
                    )
                    def _mark_topic_miss_if_needed() -> None:
                        if not (strict_topic_phase and active_user_instruction and k in active_user_pending_models):
                            return
                        tries_local = int(active_user_pending_attempts.get(k, 0)) + 1
                        active_user_pending_attempts[k] = tries_local
                        if tries_local >= TOPIC_LOCK_MAX_ATTEMPTS_PER_MODEL:
                            active_user_pending_models.discard(k)
                            active_user_pending_attempts.pop(k, None)

                    if self._is_pass_reply(clean_reply):
                        _finalize_turn_receipt_local(ReceiptStatus.PASS_)
                        _mark_topic_miss_if_needed()
                        _mark_delivered(delivered_ids)
                        continue
                    if _looks_prompt_leak_reply(clean_reply):
                        _finalize_turn_receipt_local(ReceiptStatus.REJECT, RejectReason.LOW_VALUE, "prompt_leak")
                        _mark_topic_miss_if_needed()
                        _trace_turn(k, "pass(loop_prompt_leak)", clean_reply)
                        continue
                    prev_same_model = [
                        mm.text
                        for mm in self.state.get_all_messages()[-24:]
                        if mm.visibility == "public" and mm.role == "model" and (mm.model_key or "") == k
                    ]
                    if _looks_redundant_model_repeat(clean_reply, prev_same_model):
                        _finalize_turn_receipt_local(ReceiptStatus.REJECT, RejectReason.LOW_VALUE, "repeat_self")
                        _mark_topic_miss_if_needed()
                        _trace_turn(k, "pass(loop_repeat_self)", clean_reply)
                        continue
                    if k in {"qwen", "doubao"} and _looks_unfinished_public_reply(clean_reply):
                        _finalize_turn_receipt_local(ReceiptStatus.REJECT, RejectReason.LOW_VALUE, "unfinished")
                        _mark_topic_miss_if_needed()
                        _trace_turn(k, "pass(loop_unfinished)", clean_reply)
                        continue
                    if k in {"qwen", "doubao"} and _looks_like_suggestion_chip_reply(clean_reply):
                        _finalize_turn_receipt_local(ReceiptStatus.REJECT, RejectReason.LOW_VALUE, "chip_reply")
                        _mark_topic_miss_if_needed()
                        _trace_turn(k, "pass(loop_chip)", clean_reply)
                        continue
                    if _LOW_VALUE_PROCESS_PAT.match(clean_reply) or _QWEN_STATUS_ONLY_PAT.match(clean_reply):
                        _finalize_turn_receipt_local(ReceiptStatus.REJECT, RejectReason.LOW_VALUE, "process_or_status")
                        _mark_topic_miss_if_needed()
                        _trace_turn(k, "pass(loop_process_or_status)", clean_reply)
                        continue
                    if strict_topic_phase and active_user_instruction:
                        if not _is_reply_aligned_with_user_topic(clean_reply, active_user_instruction, strict=True):
                            if k in active_user_pending_models:
                                tries = int(active_user_pending_attempts.get(k, 0)) + 1
                                active_user_pending_attempts[k] = tries
                                if tries >= TOPIC_LOCK_MAX_ATTEMPTS_PER_MODEL:
                                    active_user_pending_models.discard(k)
                                    active_user_pending_attempts.pop(k, None)
                                    mk = self.state.get_model(k)
                                    nm = mk.name if mk else k
                                    self.state.add_system(f"{nm} 连续{tries}次未对齐新话题，已暂时跳过。")
                            # Topic-lock phase: drop off-topic content to avoid stale-thread pollution.
                            _finalize_turn_receipt_local(ReceiptStatus.REJECT, RejectReason.OFF_TOPIC)
                            _trace_turn(k, "pass(loop_strict_topic_misaligned)", clean_reply)
                            continue
                    mcur = self.state.get_model(k)
                    speaker_name = mcur.name if mcur else k
                    reply_mid = self.state.add_message(
                        "model",
                        speaker_name,
                        clean_reply,
                        visibility="public",
                        model_key=k,
                    )
                    if reply_mid:
                        _finalize_turn_receipt_local(ReceiptStatus.ACCEPT)
                        _mark_delivered(delivered_ids)
                        # 广播到其他模型的待分发队列（发送者本人不回传）。
                        _enqueue_broadcast(reply_mid, current_keys, exclude_key=k)
                    round_visible_replies += 1
                    if k in active_user_pending_models:
                        active_user_pending_models.discard(k)
                        active_user_pending_attempts.pop(k, None)
                    if strict_exact_token:
                        active_user_strict_ok_models.add(k)
                        need_consensus = _strict_topic_consensus_threshold(len(current_keys))
                        if len(active_user_strict_ok_models) >= need_consensus:
                            self.state.add_system(
                                f"硬约束题已收敛（{len(active_user_strict_ok_models)}个模型给出 {strict_exact_token}），解除锁定。"
                            )
                            active_user_instruction = None
                            active_user_hard_literal = ""
                            active_user_force_rounds = 0
                            active_user_pending_models.clear()
                            active_user_pending_attempts.clear()
                            active_user_strict_ok_models = set()
                    targets = self._extract_target_keys_from_text(
                        clean_reply,
                        [x for x in current_keys if x != k],
                    )
                    new_focus: Optional[set[str]] = None
                    if targets:
                        # 仅锁定 2 人焦点，避免很快退化为固定全员轮转。
                        new_focus = {k, targets[0]}
                    elif counterpart_key and self._looks_like_disagreement(clean_reply):
                        new_focus = {k, counterpart_key}

                    if new_focus and len(new_focus) >= 2:
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

                    # 焦点讨论期间：旁听模型也收到增量消息，并可选择加入（PASS=继续旁听）。
                    if False and focus_keys:
                        observers = [x for x in current_keys if x not in focus_keys]
                        if observers:
                            sm = self.state.get_model(k)
                            source_name = sm.name if sm else k
                            payload = _pick_forward_payload(clean_reply) or clean_reply
                            for observer in observers:
                                if self.state.should_round_stop():
                                    break
                                om = self.state.get_model(observer)
                                observer_mentions = self._mention_candidates_line(current_keys, exclude_key=observer)
                                probe_instruction = self._build_observer_probe_instruction(
                                    source_name,
                                    payload,
                                    observer_mentions,
                                )
                                obs_res = self._run_model_turn(
                                    observer,
                                    probe_instruction,
                                    visibility="shadow",
                                    hidden_reply_hint=True,
                                    record_reply=False,
                                )
                                obs_res = self._apply_outer_turn_guards(
                                    key=observer,
                                    result=obs_res,
                                    round_no=round_no,
                                    round_packet_hash_seen={},
                                    authoritative_packet_hash="",
                                    enable_packet_hash=False,
                                    enable_leak=True,
                                )
                                observer_clean = _strip_private_thoughts(obs_res.public_text)
                                if not obs_res.ok or self._is_pass_reply(observer_clean):
                                    continue

                                speaker = om.name if om else observer
                                self.state.add_message(
                                    "model",
                                    speaker,
                                    observer_clean,
                                    visibility="public",
                                    model_key=observer,
                                )
                                round_visible_replies += 1
                                join_targets = self._extract_target_keys_from_text(
                                    observer_clean,
                                    [x for x in current_keys if x != observer],
                                )
                                join_peer = join_targets[0] if join_targets else k
                                observer_focus = {observer, join_peer}
                                if focus_keys != observer_focus:
                                    ordered = [x for x in current_keys if x in observer_focus]
                                    names = []
                                    for kk in ordered:
                                        mm = self.state.get_model(kk)
                                        names.append(mm.name if mm else kk)
                                    self.state.add_system("旁听模型加入焦点讨论：" + " ↔ ".join(names))
                                focus_keys = observer_focus
                                focus_idle_rounds = 0
                                mention_switched = True
                                # 继续探测其他旁听者：允许同一回合多个模型依次加入。
                                continue
                else:
                    _finalize_turn_receipt_local(
                        ReceiptStatus.REJECT,
                        self._reject_reason_from_turn_result(turn_res),
                        note=(turn_res.error_type.value if turn_res.error_type else (turn_res.error_msg or "")),
                    )
                core.jitter("模型轮转间隔")

            if round_interrupted_by_host:
                if round_no > 0:
                    round_no -= 1
                self.state.add_system("检测到群主新话题，正在切换到新一轮讨论。")
                continue

            # 群主插话至少影响后续 2 个完整回合，降低“说了但像没说”的体验。
            if active_user_instruction:
                if not active_user_hard_literal:
                    if round_visible_replies > 0:
                        active_user_force_rounds -= 1
                    if active_user_force_rounds <= 0 and not active_user_pending_models:
                        active_user_instruction = None
                        active_user_hard_literal = ""
                        active_user_force_rounds = 0
                else:
                    # “只输出某短词”属于硬约束题，保持锁定直到群主下一次插话切题。
                    active_user_force_rounds = max(1, active_user_force_rounds)

            if self.state.should_round_stop():
                self.state.add_system("已停止自动轮聊。")
                break
            if round_visible_replies <= 0:
                silent_rounds += 1
                if any_turn_ok and silent_rounds < 2:
                    self.state.add_system("本轮暂无可见回复，正在自动重试下一轮。")
                    continue
                if any_turn_ok:
                    self.state.add_system("轮聊结束：连续多轮没有可见回复。")
                else:
                    self.state.add_system("轮聊结束：本轮没有模型成功回复。")
                break
            silent_rounds = 0
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

        instruction = "请基于当前对话提出观点/反驳/补充，抓重点，像人聊天，尽量短，并保留关键细节。"
        nudge_res = self._run_model_turn(key, instruction, visibility="public")
        _ = self._apply_outer_turn_guards(
            key=key,
            result=nudge_res,
            round_no=0,
            round_packet_hash_seen={},
            authoritative_packet_hash="",
            enable_packet_hash=False,
            enable_leak=True,
        )
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
    state.add_system(f"运行日志文件：{TURN_LOG_FILE.resolve()}")
    state.add_system(f"群聊消息日志：{MESSAGE_LOG_FILE.resolve()}")
    state.add_system("请在左侧选择要启用的模型（未登录会弹出登录提示）。")
    state.add_system("底部请选择发送目标：请选择 / 群聊 / 单聊（仅已启用模型）。")
    refreshed = _enqueue_startup_login_refresh(state)
    if refreshed:
        state.add_system(f"启动后正在后台自动重检 {refreshed} 个已选模型的登录态。")

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


def _sanitize_forward_payload(text: str) -> str:
    t = _strip_group_chatter_boilerplate(text)
    t = _strip_instruction_echo_lines(t) or t
    t = core.normalize_text(t)
    if not t:
        return ""
    wrapped_public, _wrapped_private = _extract_wrapped_reply(t)
    if wrapped_public:
        # If model returned explicit wrapper blocks, keep only public payload.
        t = core.normalize_text(wrapped_public)
        if not t:
            return ""

    prompt_hints = (
        _PUBLIC_WRAP_OPEN,
        _PUBLIC_WRAP_CLOSE,
        _PRIVATE_WRAP_OPEN,
        _PRIVATE_WRAP_CLOSE,
        f"格式：{_PUBLIC_WRAP_OPEN}正文{_PUBLIC_WRAP_CLOSE}",
        "请把公开发言放在",
        "如需隐藏想法，可放在",
        "你在多人群聊中发言",
        "你在多人群聊中继续讨论",
        "群主最新话题：",
        "上一位发言（",
        "优先级：必须先回应“群主最新话题”",
        "最终发言要求：",
        "只输出给群里的正文",
        "只输出群里的正文",
        "如暂不发言，仅回复 [PASS]",
        "只输出 [PASS]",
        "可回应对象：",
        "可点名对象：",
        "只依据以下群聊消息回复",
        "不要复述本提示",
        "不要复述提示词",
        "上下文边界",
        "忽略网页里更早的旧对话",
        "Gemini 应用",
        "与 Gemini 对话",
        "须遵守《Google 条款》",
        "须遵守《Google 隐私权政策》",
        "Gemini 是一款 AI 工具",
    )
    chip_hints = (
        "造个句",
        "有哪些技巧",
        "有哪些规则",
        "推荐一些",
        "提供一些",
        "成语接龙",
        "视频生成",
        "图片生成",
        "图像生成",
        "PPT生成",
        "PPT 生成",
        "帮我写作",
        "写一份",
        "生成",
        "超能模式",
        "免费",
    )
    gemini_home_ui_hints = (
        "需要我为你做些什么",
        "制作图片",
        "创作音乐",
        "创作视频",
        "给我的一天注入活力",
        "随便写点什么",
    )
    out: list[str] = []
    seen: set[str] = set()
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            continue
        segs = [s]
        if ("|" in s or "｜" in s) and len(s) <= 180:
            split_segs = [x.strip() for x in re.split(r"[|｜]+", s) if x.strip()]
            if split_segs:
                segs = split_segs
        for seg in segs:
            if _WRAP_TOKEN_PAT.search(seg):
                continue
            if re.match(r"^(?:PRO|Pro)\b", seg) and len(_line_dedupe_key(seg)) <= 24:
                continue
            if any(h in seg for h in prompt_hints):
                continue
            if any(h in seg for h in gemini_home_ui_hints) and len(_line_dedupe_key(seg)) <= 48:
                continue
            if re.match(r"^\s*(?:显示思路|显示思考|显示推理)\s*$", seg, re.I):
                continue
            if re.match(r"^\s*(?:Gemini|ChatGPT|DeepSeek|Qwen|豆包|Doubao)\s*说\s*$", seg, re.I):
                continue
            if _GROUP_HOST_CHATTER_PAT.search(seg) and len(_line_dedupe_key(seg)) <= 56:
                continue
            if _GROUP_ORCHESTRATION_PAT.search(seg) and len(_line_dedupe_key(seg)) <= 56:
                continue
            if _TRIVIAL_PUBLIC_PAT.match(seg):
                continue
            if seg.startswith("【群聊") or seg.startswith("【用户】"):
                continue
            if re.match(r"^[（(][^）)]{1,16}[)）]\s*[:：]?\s*$", seg):
                continue
            if re.match(r"^【[^】]{1,12}】$", seg):
                continue
            if "→" in seg and len(seg) <= 42:
                continue
            if len(seg) <= 34 and any(h in seg for h in chip_hints):
                continue
            if len(seg) <= 26 and re.match(r"^(?:写一份|生成|推荐|提供|帮我)", seg):
                continue
            if len(seg) <= 22 and re.search(r"(?:视频|图片|图像|海报|插画|头像|PPT|文案)?生成$", seg):
                continue
            q_ratio = (seg.count("?") + seg.count("？")) / max(1, len(seg))
            if q_ratio >= 0.35 and len(re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", seg.lower())) < 12:
                continue
            if re.match(r"^(?:@?[\w\u4e00-\u9fff-]{1,20}\s*)?(?:接|你接|请接|来接)\s*[~～!！。\.]*$", seg):
                continue
            if _LOW_VALUE_PROCESS_PAT.match(seg):
                continue
            if re.match(r"^\s*(?:现在)?我需要.{0,140}(?:然后|再)(?:回应|补充|给出|讨论)", seg):
                continue
            m_pref = re.match(r"^(?:@?[\w\u4e00-\u9fff-]{1,24})[：:]\s*(.+)$", seg)
            if m_pref:
                tail = core.normalize_text(m_pref.group(1))
                if tail and _LOW_VALUE_PROCESS_PAT.match(tail):
                    continue

            key = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", seg.lower())
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            out.append(seg)

    return core.normalize_text("\n".join(out))


def _pick_forward_payload(full_reply: str) -> str:
    """
    只转发本轮原始发言，不做摘要提取。
    仅做长度安全截断，避免网页输入框卡死。
    """
    text = _strip_private_thoughts(full_reply) or core.normalize_text(full_reply)
    text = _sanitize_forward_payload(text) or text
    if not text:
        return ""
    return _clip_text(text, FORWARD_MAX_CHARS)


def _strip_forward_summary(full_reply: str) -> str:
    """
    不再强制摘要，展示原始回复。
    """
    text = core.normalize_text(full_reply)
    return text


def _compose_shadow_sync_message(source_site: str, user_text: str, source_reply: str) -> str:
    """
    把“当前主对话模型”的这一轮（用户 + 主模型回复）同步给另一个模型，让它在后台生成自己的回复。

    重点：
    - 用户当前不一定会立刻看到这个模型的回复（UI 会隐藏/延后显示），因此提示模型不要假设用户已读。
    - 同时要求回复“有结论也有依据”，避免空泛短句。
    """
    user_text = core.normalize_text(user_text)
    source_reply = core.normalize_text(source_reply)

    # 这段提示词是“用户消息”的一部分（网页端无法注入 system prompt），保持可控且可执行。
    sys_hint = "\n".join(
        [
            "【系统提示（旁听同步）】",
            "- 你现在处于“隐藏回复”模式：你的回复不会立刻展示给用户，用户稍后才会查看。",
            "- 用户可能尚未阅读你之前的回复：不要用“如我上面所说”等依赖已读的指代；如需引用，请简要重述关键点。",
            "- 像正常人聊天：自然、有温度，可结构化表达（短段落或少量要点）。",
            "- 允许联网：仅在需要最新事实/数据时联网；只补充事实，不要复述网页当结论；不要贴 URL。",
            "- 抓重点：先给立场，再给1-2条关键依据（可含数字、区间、假设、边界条件）。",
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

