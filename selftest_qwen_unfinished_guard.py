from __future__ import annotations

import unittest

from ai_duel_webui import _looks_unfinished_public_reply


class QwenUnfinishedGuardTest(unittest.TestCase):
    def test_rejects_summary_title_like_sentence(self) -> None:
        txt = "构建基于动态反馈的解析校验闭环"
        self.assertTrue(_looks_unfinished_public_reply(txt))

    def test_rejects_short_mechanism_problem_title(self) -> None:
        txt = "重构状态同步机制以应对分布式场景下的冲突问题。"
        self.assertTrue(_looks_unfinished_public_reply(txt))

    def test_rejects_routing_instruction_sentence(self) -> None:
        txt = "立即响应最新议题并整合广播窗口中的新信息。"
        self.assertTrue(_looks_unfinished_public_reply(txt))

    def test_rejects_vague_commitment_line(self) -> None:
        txt = "针对当前群聊话题，我将基于仓库架构提出独立评审意见。"
        self.assertTrue(_looks_unfinished_public_reply(txt))

    def test_rejects_generic_assistant_help_line(self) -> None:
        txt = "你能给我提供哪些方面的帮助？"
        self.assertTrue(_looks_unfinished_public_reply(txt))

    def test_keeps_real_argument(self) -> None:
        txt = "我建议引入消息去重与幂等校验，并把失败包隔离到 shadow，先止血再重试。"
        self.assertFalse(_looks_unfinished_public_reply(txt))


if __name__ == "__main__":
    unittest.main()
