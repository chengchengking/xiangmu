import unittest

import ai_duel_webui as webui


class GroupTopicSwitchGuardTests(unittest.TestCase):
    def test_host_interjected_since_detects_new_public_user_message(self):
        st = webui.SharedState()
        w = webui.Worker(st)
        m1 = st.add_message("user", "用户", "old topic", visibility="public", model_key=None)
        self.assertFalse(w._host_interjected_since(int(m1)))
        st.add_message("model", "Qwen", "reply", visibility="public", model_key="qwen")
        self.assertFalse(w._host_interjected_since(int(m1)))
        st.add_message("user", "用户", "new topic", visibility="public", model_key=None)
        self.assertTrue(w._host_interjected_since(int(m1)))


if __name__ == "__main__":
    unittest.main()
