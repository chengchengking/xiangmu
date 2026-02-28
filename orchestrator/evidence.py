from __future__ import annotations

import re
from typing import Iterable


_NEG = re.compile(
    r"(不同意|不认同|不成立|错误|你错了|胡扯|证据|依据|反例|矛盾|"
    r"not true|incorrect|i disagree|counterpoint|evidence)",
    re.I,
)

_HOOK_RE = re.compile(
    r"(?im)^\s*evidence_hook\s*=\s*(ref:M#\d+|check:[^\r\n]{3,200})\s*$"
)


def _iter_recent_texts(recent_public: Iterable[str], *, window: int) -> list[str]:
    items = [t for t in recent_public if t]
    return items[-max(1, int(window)) :]


def should_enter_evidence(
    recent_public: list[str],
    *,
    threshold: int = 3,
    window: int = 8,
    recent_guard: int = 3,
) -> bool:
    """
    Enter evidence mode only when:
    - enough challenge signals in a wider window, AND
    - at least one challenge appears in the most recent `recent_guard` messages
      (prevents "stuck in evidence forever" due to stale conflicts).
    """
    if not recent_public:
        return False
    window_items = _iter_recent_texts(recent_public, window=window)
    if not window_items:
        return False
    hits = sum(1 for t in window_items if _NEG.search(t))
    if hits < max(1, int(threshold)):
        return False
    tail = window_items[-max(1, int(recent_guard)) :]
    return any(_NEG.search(t) for t in tail)


def has_valid_evidence_hook(private_reply: str) -> bool:
    raw = private_reply or ""
    return bool(_HOOK_RE.search(raw))

