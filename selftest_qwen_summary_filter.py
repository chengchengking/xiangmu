from __future__ import annotations

import unittest

from model_adapters import ModelMeta, QwenAdapter


def _meta() -> ModelMeta:
    return ModelMeta(
        slot=5,
        key="qwen",
        name="Qwen",
        url="https://chat.qwen.ai/",
        color="#55aaee",
        integrated=True,
        avatar_url="",
        login_help="",
    )


class TestQwenSummaryFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.ad = QwenAdapter(_meta())

    def test_reject_summary_title_like_sentence(self) -> None:
        # Typical wrong extraction from Qwen thought/sidebar summary.
        bad = "构建多层级容错机制以增强对话系统的鲁棒性。"
        self.assertEqual(self.ad._clean_candidate_text(bad), "")

    def test_keep_real_argument_sentence(self) -> None:
        good = "我不同意只做重试；应先加 packet_hash 一致性校验，再做超时降级。"
        self.assertEqual(self.ad._clean_candidate_text(good), good)


if __name__ == "__main__":
    unittest.main()

