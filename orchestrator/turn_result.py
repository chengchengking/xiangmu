from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class TurnResultStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PASS_ = "PASS"
    ERROR = "ERROR"


class TurnErrorType(str, Enum):
    LEAK = "LEAK"
    HASH_MISMATCH = "HASH_MISMATCH"
    TIMEOUT = "TIMEOUT"
    PARSE_FAIL = "PARSE_FAIL"
    EVIDENCE_INVALID = "EVIDENCE_INVALID"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    ADAPTER_MISSING = "ADAPTER_MISSING"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    NOT_AUTHENTICATED = "NOT_AUTHENTICATED"
    EMPTY_INSTRUCTION = "EMPTY_INSTRUCTION"
    UNKNOWN = "UNKNOWN"


@dataclass
class TurnResult:
    status: TurnResultStatus
    adapter_name: str
    model_key: str
    raw_reply: str = ""
    parsed_content: str = ""
    private_content: str = ""
    turn_id: int = 0
    mode: str = ""
    packet_hash: str = ""
    envelope_status: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    error_type: Optional[TurnErrorType] = None
    error_msg: Optional[str] = None
    meta: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {TurnResultStatus.SUCCESS, TurnResultStatus.PASS_}

    @property
    def public_text(self) -> str:
        if self.status == TurnResultStatus.PASS_:
            return "[PASS]"
        return self.parsed_content or ""

    def as_legacy_tuple(self) -> tuple[bool, str]:
        return self.ok, self.public_text

    def __iter__(self):
        ok, text = self.as_legacy_tuple()
        yield ok
        yield text

    @classmethod
    def success(
        cls,
        *,
        adapter_name: str,
        model_key: str,
        parsed_content: str,
        raw_reply: str = "",
        private_content: str = "",
        turn_id: int = 0,
        mode: str = "",
        packet_hash: str = "",
        envelope_status: str = "",
        metrics: Optional[Dict[str, float]] = None,
        meta: Optional[Dict[str, str]] = None,
    ) -> "TurnResult":
        return cls(
            status=TurnResultStatus.SUCCESS,
            adapter_name=adapter_name,
            model_key=model_key,
            raw_reply=raw_reply,
            parsed_content=parsed_content,
            private_content=private_content,
            turn_id=turn_id,
            mode=mode,
            packet_hash=packet_hash,
            envelope_status=envelope_status,
            metrics=metrics or {},
            meta=meta or {},
        )

    @classmethod
    def pass_(
        cls,
        *,
        adapter_name: str,
        model_key: str,
        reason: str = "",
        turn_id: int = 0,
        mode: str = "",
        raw_reply: str = "",
        private_content: str = "",
        packet_hash: str = "",
        envelope_status: str = "",
        metrics: Optional[Dict[str, float]] = None,
        meta: Optional[Dict[str, str]] = None,
    ) -> "TurnResult":
        return cls(
            status=TurnResultStatus.PASS_,
            adapter_name=adapter_name,
            model_key=model_key,
            raw_reply=raw_reply,
            parsed_content="[PASS]",
            private_content=private_content,
            turn_id=turn_id,
            mode=mode,
            packet_hash=packet_hash,
            envelope_status=envelope_status,
            metrics=metrics or {},
            error_msg=reason or None,
            meta=meta or {},
        )

    @classmethod
    def error(
        cls,
        *,
        adapter_name: str,
        model_key: str,
        error_type: TurnErrorType,
        error_msg: str = "",
        raw_reply: str = "",
        parsed_content: str = "",
        private_content: str = "",
        turn_id: int = 0,
        mode: str = "",
        packet_hash: str = "",
        envelope_status: str = "",
        metrics: Optional[Dict[str, float]] = None,
        meta: Optional[Dict[str, str]] = None,
    ) -> "TurnResult":
        return cls(
            status=TurnResultStatus.ERROR,
            adapter_name=adapter_name,
            model_key=model_key,
            raw_reply=raw_reply,
            parsed_content=parsed_content,
            private_content=private_content,
            turn_id=turn_id,
            mode=mode,
            packet_hash=packet_hash,
            envelope_status=envelope_status,
            metrics=metrics or {},
            error_type=error_type,
            error_msg=error_msg or None,
            meta=meta or {},
        )
