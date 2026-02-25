import unittest

import ai_duel_webui as webui


class StartupLoginRefreshTests(unittest.TestCase):
    def test_enqueue_only_selected_integrated_models(self):
        st = webui.SharedState()
        for key in [m["key"] for m in st.get_models()["models"]]:
            mm = st.get_model(key)
            if mm is not None:
                mm.selected = False
        # Simulate persisted selected models.
        for key in ["chatgpt", "gemini", "qwen"]:
            m = st.get_model(key)
            self.assertIsNotNone(m)
            m.selected = True
        # Non-integrated slot should be ignored even if selected.
        m9 = st.get_model("slot9")
        self.assertIsNotNone(m9)
        m9.selected = True

        queued = webui._enqueue_startup_login_refresh(st)
        self.assertEqual(queued, 3)

        seen = []
        for _ in range(queued):
            item = st.inbox.get_nowait()
            seen.append((item.get("kind"), item.get("key")))
        self.assertEqual(seen, [("login_check", "chatgpt"), ("login_check", "gemini"), ("login_check", "qwen")])


if __name__ == "__main__":
    unittest.main()
