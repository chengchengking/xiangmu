import unittest

from ai_duel_webui import _is_reply_aligned_with_user_topic


class RepoTopicAlignmentTests(unittest.TestCase):
    def test_repo_fix_reply_should_align_in_strict_mode(self):
        topic = (
            "Repo: https://github.com/chengchengking/xiangmu . "
            "Please discuss parser extraction, broadcast consistency, topic interrupt behavior, "
            "timeout/captcha recovery, and low-risk fixes for our WebUI groupchat product."
        )
        reply = (
            "先修 parser 提取与广播一致性：把 packet_hash 校验失败直接隔离，并在日志里记录。"
            "同时收紧 topic-switch 守卫，避免插话后旧回复落进 public。再补 timeout/captcha recover 流程。"
        )
        self.assertTrue(_is_reply_aligned_with_user_topic(reply, topic, strict=True))

    def test_clear_generic_planning_should_still_fail(self):
        topic = (
            "Repo: https://github.com/chengchengking/xiangmu . "
            "Please discuss concrete fixes for parser and broadcast issues."
        )
        reply = "查阅项目仓库以明确产品定位与潜在问题。"
        self.assertFalse(_is_reply_aligned_with_user_topic(reply, topic, strict=True))

    def test_repo_topic_with_digits_should_not_require_reply_digits(self):
        topic = (
            "Repo: https://github.com/chengchengking/xiangmu . "
            "Discuss fairness among 5 models, parser extraction, broadcast consistency, "
            "topic interrupt switching, timeout/captcha recovery."
        )
        reply = (
            "先修 parser 提取和广播一致性，再处理切题打断与 timeout/captcha 恢复。"
            "另外要限制单模型连续刷屏，提升群聊公平性。"
        )
        self.assertTrue(_is_reply_aligned_with_user_topic(reply, topic, strict=True))


if __name__ == "__main__":
    unittest.main()
