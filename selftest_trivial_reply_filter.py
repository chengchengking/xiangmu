import unittest

from ai_duel_webui import _TRIVIAL_PUBLIC_PAT


class TrivialReplyFilterTest(unittest.TestCase):
    def test_standalone_rengran_is_trivial(self):
        self.assertTrue(bool(_TRIVIAL_PUBLIC_PAT.match("仍然")))

    def test_normal_sentence_is_not_trivial(self):
        self.assertFalse(bool(_TRIVIAL_PUBLIC_PAT.match("仍然建议优先修复广播一致性与回执闭环。")))


if __name__ == "__main__":
    unittest.main()
