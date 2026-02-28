import unittest

import ai_duel_webui as webui


class TopicLockStrictTests(unittest.TestCase):
    def test_repo_topic_rejects_off_topic_question(self):
        topic = "Repo: https://github.com/chengchengking/xiangmu. Diagnose parser/broadcast bugs and propose fixes."
        reply = "为什么薄荷糖中间有一个洞？"
        self.assertFalse(webui._is_reply_aligned_with_user_topic(reply, topic, strict=True))

    def test_repo_topic_accepts_repo_fix_reply(self):
        topic = "Repo: https://github.com/chengchengking/xiangmu. Diagnose parser/broadcast bugs and propose fixes."
        reply = "主要问题是广播包和topic切换未对齐，建议在orchestrator加入packet_hash一致性告警与回归测试。"
        self.assertTrue(webui._is_reply_aligned_with_user_topic(reply, topic, strict=True))

    def test_authoritative_packet_respects_floor_id(self):
        state = webui.SharedState()
        worker = webui.Worker(state)
        # old topic
        state.add_message("user", "用户", "old topic", visibility="public", model_key=None)
        state.add_message("model", "Qwen", "old reply", visibility="public", model_key="qwen")
        # new topic
        floor = state.add_message("user", "用户", "new topic", visibility="public", model_key=None)
        state.add_message("model", "ChatGPT", "new reply", visibility="public", model_key="chatgpt")

        packet, _ = worker._build_authoritative_broadcast_packet(floor_id=floor)
        self.assertIn("new topic", packet)
        self.assertIn("new reply", packet)
        self.assertNotIn("old topic", packet)
        self.assertNotIn("old reply", packet)


if __name__ == "__main__":
    unittest.main()
