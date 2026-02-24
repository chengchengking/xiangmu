import threading
import unittest

import ai_duel_webui as app


class SendAuthGuardTests(unittest.TestCase):
    def test_group_send_with_selected_but_not_authenticated_returns_clear_error(self):
        state = app.SharedState()
        w = app.Worker(state)
        # Simulate user selected models but has not completed login detection yet.
        state.toggle_selected("qwen")
        state.set_authenticated("qwen", False)

        box = {}
        ev = threading.Event()
        action = {
            "kind": "send",
            "target": "group",
            "text": "hello",
            "_reply": box,
            "_ev": ev,
        }
        w._handle_send(action)
        self.assertTrue(ev.is_set())
        self.assertEqual(box.get("error"), "selected_but_not_authenticated")
        msgs = state.get_all_messages()
        self.assertTrue(
            any("已启用但未检测到登录成功" in (m.text or "") for m in msgs),
            "expected clearer auth guidance in system message",
        )


if __name__ == "__main__":
    unittest.main()
