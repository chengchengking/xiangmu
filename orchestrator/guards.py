from __future__ import annotations

from dataclasses import replace
from typing import Callable

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

