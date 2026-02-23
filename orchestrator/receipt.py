from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ReceiptStatus(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    PASS_ = "PASS"


class RejectReason(str, Enum):
    TOO_SLOW = "TOO_SLOW"
    LOW_VALUE = "LOW_VALUE"
    OFF_TOPIC = "OFF_TOPIC"
    NO_PUBLIC_TAG = "NO_PUBLIC_TAG"
    TIMEOUT = "TIMEOUT"
    PARSE_FAIL = "PARSE_FAIL"


@dataclass
class Receipt:
    turn_id: int
    status: ReceiptStatus
    reason: Optional[RejectReason] = None
    winner_mid: Optional[str] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        if self.reason is not None:
            d["reason"] = self.reason.value
        return d

