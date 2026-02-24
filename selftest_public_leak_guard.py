import unittest

import ai_duel_webui as webui


class PublicLeakGuardTests(unittest.TestCase):
    def test_meta_echo_is_detected_as_prompt_leak(self):
        sample = (
            "[[META]]\nturn_id=12\nack_in=88\npacket_hash=abc123\n[[/META]]\n"
            "[[PUBLIC_REPLY]]hello[[/PUBLIC_REPLY]]"
        )
        self.assertTrue(webui._looks_prompt_leak_reply(sample))

    def test_normal_public_reply_not_flagged(self):
        sample = "I think the answer is 444 because 37*12 = 444, and the arithmetic is direct."
        self.assertFalse(webui._looks_prompt_leak_reply(sample))

    def test_format_parrot_is_detected_as_prompt_leak(self):
        sample = "然后按格式写 META，PUBLIC_REPLY。"
        self.assertTrue(webui._looks_prompt_leak_reply(sample))

    def test_machine_placeholder_token_is_rejected(self):
        sample = "<<WRITE_PUBLIC_OR_[PASS]>>"
        self.assertTrue(webui._looks_protocol_placeholder_public_reply(sample))

    def test_task_planning_line_is_detected_as_prompt_leak(self):
        sample = "查阅项目仓库以明确产品定位与潜在问题"
        self.assertTrue(webui._looks_prompt_leak_reply(sample))

    def test_reading_status_line_is_detected_as_prompt_leak(self):
        sample = "正在阅读正在阅读"
        self.assertTrue(webui._looks_prompt_leak_reply(sample))


if __name__ == "__main__":
    unittest.main()
