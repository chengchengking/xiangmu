import unittest

import ai_duel_webui as webui


class ShadowScopeTests(unittest.TestCase):
    def test_should_sync_shadow_to_peers(self):
        self.assertTrue(webui.Worker._should_sync_shadow_to_peers("shared"))
        self.assertTrue(webui.Worker._should_sync_shadow_to_peers("SHARED"))
        self.assertFalse(webui.Worker._should_sync_shadow_to_peers("local"))
        self.assertFalse(webui.Worker._should_sync_shadow_to_peers("LOCAL"))

    def test_shared_state_records_shadow_scope(self):
        st = webui.SharedState()
        mid = st.add_message(
            "user",
            "用户",
            "secret",
            visibility="shadow",
            model_key="qwen",
            shadow_scope="local",
        )
        msgs = st.get_all_messages()
        self.assertTrue(mid > 0)
        self.assertEqual(msgs[-1].visibility, "shadow")
        self.assertEqual(msgs[-1].shadow_scope, "local")


if __name__ == "__main__":
    unittest.main()

