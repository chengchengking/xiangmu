import unittest

from ai_duel_webui import _looks_prompt_leak_reply, _looks_transient_progress_only


class TransientProgressFilterTest(unittest.TestCase):
    def test_transient_progress_detected(self):
        self.assertTrue(_looks_transient_progress_only("正在阅读正在阅读"))
        self.assertTrue(_looks_prompt_leak_reply("正在阅读"))

    def test_normal_public_not_progress(self):
        txt = "我同意这个修复方向，先做广播一致性，再做ACK闭环。"
        self.assertFalse(_looks_transient_progress_only(txt))
        self.assertFalse(_looks_prompt_leak_reply(txt))


if __name__ == "__main__":
    unittest.main()

