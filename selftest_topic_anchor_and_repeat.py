import unittest

import ai_duel_webui as webui


class TopicAnchorAndRepeatTests(unittest.TestCase):
    def test_required_topic_anchors_extracted_for_repo_debug_topic(self):
        t = "Discuss parser extraction mistakes, broadcast consistency, and topic-switching in our repo WebUI."
        anchors = webui._required_topic_anchors(t)
        self.assertTrue(any(x in anchors for x in ["parser", "parse"]))
        self.assertTrue(any(x in anchors for x in ["broadcast", "广播"]))
        self.assertTrue(any(x in anchors for x in ["topic-switch", "切题", "interject"]))

    def test_strict_alignment_rejects_reply_without_required_anchor(self):
        topic = "Repo issue: parser extraction, broadcast consistency, topic-switch interrupt behavior."
        bad = "The core issue is TCP packet fragmentation and NTP drift across distributed nodes."
        self.assertFalse(webui._is_reply_aligned_with_user_topic(bad, topic, strict=True))

    def test_redundant_model_repeat_detected(self):
        prev = [
            "当前核心问题是 parser 提取误抓和 broadcast 一致性，需要先修 topic-switching。",
            "建议先修 parser 提取误抓和 broadcast 一致性，再处理 topic-switching。"
        ]
        cur = "先修 parser 提取误抓和 broadcast 一致性，再处理 topic-switching。"
        self.assertTrue(webui._looks_redundant_model_repeat(cur, prev))


if __name__ == "__main__":
    unittest.main()

