import unittest

import ai_duel_webui as webui


class GroupAuthWaitTests(unittest.TestCase):
    def test_should_wait_for_login_recheck_states(self):
        self.assertTrue(webui._should_wait_for_login_recheck("INITIALIZING", "login_check"))
        self.assertTrue(webui._should_wait_for_login_recheck("AUTH_CHECKING", ""))
        self.assertTrue(webui._should_wait_for_login_recheck("IDLE", "login_check_ok"))
        self.assertFalse(webui._should_wait_for_login_recheck("NEEDS_HUMAN", "login_check_failed"))
        self.assertFalse(webui._should_wait_for_login_recheck("COOLING_DOWN", "turn:timeout"))


if __name__ == "__main__":
    unittest.main()

