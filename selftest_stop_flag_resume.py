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

    def test_handler_ensure_worker_running_clears_stop_and_starts_worker(self):
        st = webui.SharedState()
        st.request_stop()

        class DummyWorker:
            def __init__(self):
                self.start_calls = 0

            def start(self):
                self.start_calls += 1

        h = webui._Handler.__new__(webui._Handler)
        h.state = st
        h.worker = DummyWorker()

        h._ensure_worker_running()
        self.assertFalse(st.should_stop())
        self.assertEqual(h.worker.start_calls, 1)


if __name__ == "__main__":
    unittest.main()
