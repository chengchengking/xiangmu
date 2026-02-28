import unittest

from ai_duel_webui import _strip_protocol_suffix_from_instruction
from orchestrator.evidence import has_valid_evidence_hook, should_enter_evidence


class ProtocolFallbackRetryTests(unittest.TestCase):
    def test_strip_protocol_suffix(self) -> None:
        src = (
            "核心任务：讨论仓库问题。\n"
            "请按以下格式输出（不要复述本提示）：\n"
            "[[META]]\nturn_id=1\n[[/META]]\n[[PUBLIC_REPLY]]\n[[/PUBLIC_REPLY]]\n"
        )
        out = _strip_protocol_suffix_from_instruction(src)
        self.assertEqual(out, "核心任务：讨论仓库问题。")

    def test_has_valid_evidence_hook(self) -> None:
        self.assertTrue(has_valid_evidence_hook("evidence_hook=ref:M#123"))
        self.assertTrue(has_valid_evidence_hook("x\nevidence_hook=check:引用运行日志字段\nz"))
        self.assertFalse(has_valid_evidence_hook("evidence_hook=ok"))

    def test_should_enter_evidence_signal(self) -> None:
        recent = [
            "我不同意这个结论",
            "请给证据，不然不成立",
            "i disagree with this",
            "继续讨论",
        ]
        self.assertTrue(should_enter_evidence(recent, threshold=2, window=4, recent_guard=3))


if __name__ == "__main__":
    unittest.main()

