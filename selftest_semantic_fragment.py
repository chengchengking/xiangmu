from __future__ import annotations

import unittest

from ai_duel_webui import _pick_best_semantic_fragment


class SemanticFragmentPickTest(unittest.TestCase):
    def test_prefers_full_coherent_block_over_short_heading(self) -> None:
        text = (
            '构建多层级容错机制以增强对话系统鲁棒性。\n'
            '我建议先在解析层做 packet_hash 与 envelope 完整性校验，再做 timeout 降级，'
            '并记录 reject reason 分布用于阈值调参。'
        )
        out = _pick_best_semantic_fragment(text)
        self.assertIn('packet_hash', out)
        self.assertIn('reject reason', out)


if __name__ == '__main__':
    unittest.main()
