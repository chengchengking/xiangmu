from __future__ import annotations

import unittest

from ai_duel_webui import _looks_unfinished_public_reply


class QwenUnfinishedGuardTest(unittest.TestCase):
    def test_rejects_summary_process_stub(self) -> None:
        txt = '识别并处理输出中的结构化标记与无法解析干扰。'
        self.assertTrue(_looks_unfinished_public_reply(txt))

    def test_rejects_repo_action_stub(self) -> None:
        txt = '查阅项目仓库以明确产品定位与潜在问题。'
        self.assertTrue(_looks_unfinished_public_reply(txt))

    def test_rejects_trailing_conjunction_fragment(self) -> None:
        txt = '在 extractor 入口加入 MutationObserver 监听目标区域，并'
        self.assertTrue(_looks_unfinished_public_reply(txt))

    def test_rejects_trying_to_fetch_more_docs_stub(self) -> None:
        txt = '正在尝试获取更详细的架构文档以解析运行时问题。'
        self.assertTrue(_looks_unfinished_public_reply(txt))

    def test_rejects_push_topic_forward_stub(self) -> None:
        txt = '推动群聊议题向前发展'
        self.assertTrue(_looks_unfinished_public_reply(txt))

    def test_keeps_actionable_one_line(self) -> None:
        txt = '构建多层级容错机制以增强对话系统鲁棒性，并通过日志阈值回退降低误判。'
        self.assertFalse(_looks_unfinished_public_reply(txt))

    def test_keeps_real_argument(self) -> None:
        txt = '我建议引入消息去重与幂等校验，并把失败包隔离到 shadow，先止血再重试。'
        self.assertFalse(_looks_unfinished_public_reply(txt))


if __name__ == '__main__':
    unittest.main()
