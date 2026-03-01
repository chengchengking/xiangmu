from __future__ import annotations

import unittest

from ai_duel_webui import _looks_low_independence_reply


class QwenIndependenceGuardTest(unittest.TestCase):
    def test_rejects_plain_agreement(self) -> None:
        txt = '我同意上一位的观点。'
        self.assertTrue(_looks_low_independence_reply(txt))

    def test_keeps_agreement_with_new_technical_point(self) -> None:
        txt = '我同意上一位结论，但新增风险是阈值过低会误杀短回复，建议把长度阈值从24调到32并记录日志。'
        self.assertFalse(_looks_low_independence_reply(txt))

    def test_non_agreement_line_not_blocked_by_independence_guard(self) -> None:
        txt = '建议先做 packet_hash 一致性校验，再把 low-value 规则移到外层守卫。'
        self.assertFalse(_looks_low_independence_reply(txt))


if __name__ == '__main__':
    unittest.main()
