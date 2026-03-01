from __future__ import annotations

import unittest

from model_adapters import ModelMeta, QwenAdapter


def _meta() -> ModelMeta:
    return ModelMeta(
        slot=5,
        key='qwen',
        name='Qwen',
        url='https://chat.qwen.ai/',
        color='#55aaee',
        integrated=True,
        avatar_url='',
        login_help='',
    )


class TestQwenSummaryFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.ad = QwenAdapter(_meta())

    def test_reject_focus_confirmation_sentence(self) -> None:
        bad = '确认当前讨论焦点是千问标题提取异常问题。'
        self.assertEqual(self.ad._clean_candidate_text(bad), '')

    def test_reject_repo_action_sentence(self) -> None:
        bad = '查阅项目仓库以明确产品定位与潜在问题。'
        self.assertEqual(self.ad._clean_candidate_text(bad), '')

    def test_reject_generic_assessment_sentence(self) -> None:
        bad = '审查项目架构并评估技术可行性。'
        self.assertEqual(self.ad._clean_candidate_text(bad), '')

    def test_keep_actionable_one_line(self) -> None:
        good = '构建多层级容错机制以增强对话系统鲁棒性，并通过日志阈值回退降低误判。'
        self.assertEqual(self.ad._clean_candidate_text(good), good)

    def test_keep_real_argument_sentence(self) -> None:
        good = '我不同意只做重试；应先加 packet_hash 一致性校验，再做超时降级。'
        self.assertEqual(self.ad._clean_candidate_text(good), good)

    def test_drops_packet_history_line_and_keeps_new_line(self) -> None:
        good_line = '我不同意只做重试；应先加 packet_hash 一致性校验，再做超时降级。'
        mixed = (
            '[18] DeepSeek: 建议先做统一包校验\n'
            + good_line
        )
        out = self.ad._clean_candidate_text(mixed)
        self.assertNotIn('[18] DeepSeek', out)
        self.assertEqual(out, good_line)


if __name__ == '__main__':
    unittest.main()
