from __future__ import annotations

from enum import Enum


class Mode(str, Enum):
    SINGLE_FAST = "SINGLE_FAST"
    MULTI_ROUND = "MULTI_ROUND"
    EVIDENCE = "EVIDENCE"
