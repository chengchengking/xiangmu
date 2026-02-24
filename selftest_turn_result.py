import unittest

from orchestrator.turn_result import TurnErrorType, TurnResult, TurnResultStatus


class TurnResultTests(unittest.TestCase):
    def test_success_legacy_tuple(self):
        r = TurnResult.success(adapter_name="Gemini", model_key="gemini", parsed_content="hello", turn_id=3, mode="MULTI_ROUND")
        self.assertEqual(r.status, TurnResultStatus.SUCCESS)
        self.assertTrue(r.ok)
        self.assertEqual(r.public_text, "hello")
        self.assertEqual(r.as_legacy_tuple(), (True, "hello"))

    def test_pass_result(self):
        r = TurnResult.pass_(adapter_name="Qwen", model_key="qwen", reason="no_evidence_hook")
        self.assertEqual(r.status, TurnResultStatus.PASS_)
        self.assertTrue(r.ok)
        self.assertEqual(r.public_text, "[PASS]")
        self.assertEqual(r.error_msg, "no_evidence_hook")

    def test_error_result(self):
        r = TurnResult.error(
            adapter_name="ChatGPT",
            model_key="chatgpt",
            error_type=TurnErrorType.TIMEOUT,
            error_msg="timed out waiting for reply",
        )
        self.assertEqual(r.status, TurnResultStatus.ERROR)
        self.assertFalse(r.ok)
        self.assertEqual(r.error_type, TurnErrorType.TIMEOUT)
        self.assertEqual(r.as_legacy_tuple(), (False, ""))


if __name__ == "__main__":
    unittest.main()

