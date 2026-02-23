from __future__ import annotations

import re
from typing import List

_NEG = re.compile(
    r"(不同意|不认同|不成立|错误|你错了|胡扯|证据|依据|反例|矛盾|not true|incorrect|i disagree)",
    re.I,
)


def should_enter_evidence(recent_public: List[str], *, threshold: int = 3, window: int = 8) -> bool:
    if not recent_public:
        return False
    hits = 0
    for t in recent_public[-max(1, int(window)) :]:
        if not t:
            continue
        if _NEG.search(t):
            hits += 1
    return hits >= max(1, int(threshold))
