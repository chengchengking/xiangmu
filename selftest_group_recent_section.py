import unittest

import ai_duel_webui as w


class GroupRecentSectionTest(unittest.TestCase):
    def test_authoritative_packet_hides_recent_body(self):
        recent = "[17] ChatGPT: alpha\n[18] DeepSeek: beta"
        section = w._format_group_recent_section(recent, use_authoritative_packet=True)
        self.assertIn("广播包", section)
        self.assertNotIn("[17]", section)
        self.assertNotIn("[18]", section)

    def test_no_authoritative_packet_keeps_recent_body(self):
        recent = "[17] ChatGPT: alpha"
        section = w._format_group_recent_section(recent, use_authoritative_packet=False)
        self.assertIn("群聊广播窗口", section)
        self.assertIn("[17]", section)


if __name__ == "__main__":
    unittest.main()
