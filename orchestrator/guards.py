from __future__ import annotations

from dataclasses import replace
from typing import Callable

from .receipt import RejectReason
from .turn_result import TurnErrorType, TurnResult, TurnResultStatus


GuardFn = Callable[[TurnResult], TurnResult]


def apply_turn_guards(result: TurnResult, *guards: GuardFn) -> TurnResult:
    out = result
    for guard in guards:
        if out.status != TurnResultStatus.SUCCESS:
            break
        out = guard(out)
    return out


def error_like(result: TurnResult, *, error_type: TurnErrorType, error_msg: str = "") -> TurnResult:
    return replace(
        result,
        status=TurnResultStatus.ERROR,
        error_type=error_type,
        error_msg=error_msg or result.error_msg,
    )


def pass_like(result: TurnResult, *, reason: str = "") -> TurnResult:
    return replace(
        result,
        status=TurnResultStatus.PASS_,
        parsed_content="[PASS]",
        error_type=None,
        error_msg=reason or result.error_msg,
    )


def map_turn_error_to_reject_reason(error_type: TurnErrorType | None) -> RejectReason:
    match error_type:
        case TurnErrorType.TIMEOUT:
            return RejectReason.TIMEOUT
        case TurnErrorType.LEAK | TurnErrorType.EVIDENCE_INVALID | TurnErrorType.EMPTY_INSTRUCTION:
            return RejectReason.LOW_VALUE
        case (
            TurnErrorType.HASH_MISMATCH
            | TurnErrorType.PROVIDER_ERROR
            | TurnErrorType.PARSE_FAIL
            | TurnErrorType.MODEL_UNAVAILABLE
            | TurnErrorType.NOT_AUTHENTICATED
            | TurnErrorType.ADAPTER_MISSING
            | TurnErrorType.UNKNOWN
            | None
        ):
            return RejectReason.PARSE_FAIL
        case _:
            return RejectReason.PARSE_FAIL
