import unittest

from ai_duel_webui import _looks_prompt_leak_reply, _strip_instruction_echo_lines


class PromptEchoGuardTest(unittest.TestCase):
    def test_chinese_format_echo_is_leak(self):
        txt = "然后按照格式来，META 部分按要求，PUBLIC_REPLY 写 concise 的内容。"
        self.assertTrue(_looks_prompt_leak_reply(txt))

    def test_strip_format_echo_line(self):
        txt = "然后按照格式来，META 部分按要求，PUBLIC_REPLY 写 concise 的内容。\n结论：先修复广播一致性与回执闭环。"
        out = _strip_instruction_echo_lines(txt)
        self.assertIn("结论：先修复广播一致性与回执闭环。", out)
        self.assertNotIn("META 部分按要求", out)


if __name__ == "__main__":
    unittest.main()
