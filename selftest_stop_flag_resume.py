import unittest

import ai_duel_webui as webui


class StopFlagResumeTests(unittest.TestCase):
    def test_shared_state_stop_flag_can_be_cleared(self):
        st = webui.SharedState()
        self.assertFalse(st.should_stop())
        st.request_stop()
        self.assertTrue(st.should_stop())
        st.clear_stop()
        self.assertFalse(st.should_stop())


if __name__ == "__main__":
    unittest.main()

