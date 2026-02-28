import unittest

from ai_duel_webui import _LOW_VALUE_PROCESS_PAT


class LowValueFilterTest(unittest.TestCase):
    def test_qwen_source_read_complete_is_low_value(self):
        self.assertTrue(bool(_LOW_VALUE_PROCESS_PAT.match("读取来源已完成")))

    def test_real_reply_not_low_value(self):
        txt = "结论：先修消息顺序和topic隔离，再加解析兜底，最后做Worker恢复。"
        self.assertFalse(bool(_LOW_VALUE_PROCESS_PAT.match(txt)))


if __name__ == "__main__":
    unittest.main()

