import json
import tempfile
import unittest
from pathlib import Path

import ai_duel_webui as webui


class WorkerStateTransitionTests(unittest.TestCase):
    def test_transition_updates_model_runtime_and_logs(self):
        state = webui.SharedState()
        key = "qwen"
        m = state.get_model(key)
        self.assertIsNotNone(m)

        with tempfile.TemporaryDirectory() as td:
            old = webui.WORKER_STATE_LOG_FILE
            try:
                webui.WORKER_STATE_LOG_FILE = Path(td) / "worker_state.jsonl"
                state.set_model_runtime_state(key, "INITIALIZING", reason="test_init")
                state.set_model_runtime_state(key, "COOLING_DOWN", reason="test_timeout", fail_delta=1, cooldown_s=5)
                box = state.get_model_runtime_state(key)
                self.assertTrue(box["ok"])
                self.assertEqual(box["state"], "COOLING_DOWN")
                self.assertEqual(box["fail_count"], 1)
                self.assertGreaterEqual(box["cooldown_remaining_s"], 0.0)
                lines = webui.WORKER_STATE_LOG_FILE.read_text(encoding="utf-8").splitlines()
                self.assertGreaterEqual(len(lines), 2)
                last = json.loads(lines[-1])
                self.assertEqual(last["model_key"], key)
                self.assertEqual(last["old_state"], "INITIALIZING")
                self.assertEqual(last["new_state"], "COOLING_DOWN")
            finally:
                webui.WORKER_STATE_LOG_FILE = old

    def test_worker_failure_classifier(self):
        self.assertEqual(webui.Worker._classify_worker_failure("captcha required"), "captcha")
        self.assertEqual(webui.Worker._classify_worker_failure("timed out waiting"), "timeout")
        self.assertEqual(webui.Worker._classify_worker_failure("locator not found"), "selector_miss")


if __name__ == "__main__":
    unittest.main()

