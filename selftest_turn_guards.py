import unittest

from orchestrator.guards import (
    apply_turn_guards,
    error_like,
    map_turn_error_to_reject_reason,
    pass_like,
)
from orchestrator.receipt import RejectReason
from orchestrator.turn_result import TurnErrorType, TurnResult, TurnResultStatus


class TurnGuardTests(unittest.TestCase):
    def test_pipeline_short_circuits_after_error(self):
        seen = []

        def g1(r):
            seen.append("g1")
            return error_like(r, error_type=TurnErrorType.LEAK, error_msg="leak")

        def g2(r):
            seen.append("g2")
            return r

        r0 = TurnResult.success(adapter_name="a", model_key="k", parsed_content="hello")
        r1 = apply_turn_guards(r0, g1, g2)
        self.assertEqual(r1.status, TurnResultStatus.ERROR)
        self.assertEqual(r1.error_type, TurnErrorType.LEAK)
        self.assertEqual(seen, ["g1"])

    def test_pass_like_keeps_structure(self):
        r0 = TurnResult.success(adapter_name="a", model_key="k", parsed_content="x")
        r1 = pass_like(r0, reason="filtered")
        self.assertEqual(r1.status, TurnResultStatus.PASS_)
        self.assertEqual(r1.public_text, "[PASS]")
        self.assertEqual(r1.error_msg, "filtered")

    def test_error_mapping_covers_common_types(self):
        self.assertEqual(map_turn_error_to_reject_reason(TurnErrorType.TIMEOUT), RejectReason.TIMEOUT)
        self.assertEqual(map_turn_error_to_reject_reason(TurnErrorType.LEAK), RejectReason.LOW_VALUE)
        self.assertEqual(map_turn_error_to_reject_reason(TurnErrorType.EVIDENCE_INVALID), RejectReason.LOW_VALUE)
        self.assertEqual(map_turn_error_to_reject_reason(TurnErrorType.PARSE_FAIL), RejectReason.PARSE_FAIL)
        self.assertEqual(map_turn_error_to_reject_reason(None), RejectReason.PARSE_FAIL)


if __name__ == "__main__":
    unittest.main()
