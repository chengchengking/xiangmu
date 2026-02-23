from __future__ import annotations

import unittest

import ai_duel_webui as webui


class _DummyWorker(webui.Worker):
    def __init__(self, state: webui.SharedState) -> None:
        super().__init__(state)
        self.calls: list[tuple[str, str, str]] = []
        self._pw = object()

    def _ensure_playwright(self) -> None:  # type: ignore[override]
        self._pw = object()

    def _safe_reply(self, action, payload):  # type: ignore[override]
        return None

    def _run_model_turn(self, key: str, instruction: str, **kwargs):  # type: ignore[override]
        vis = kwargs.get("visibility", "")
        self.calls.append((key, vis, instruction))
        if vis == "shadow" and kwargs.get("hidden_reply_hint"):
            return True, "[PASS]"
        return True, "主模型回复"


class ShadowSyncIsolationTests(unittest.TestCase):
    def setUp(self):
        self.state = webui.SharedState()
        # Select two integrated models for deterministic single-chat sync behavior.
        for key in ("chatgpt", "gemini"):
            m = self.state.get_model(key)
            assert m is not None
            m.selected = True
            m.integrated = True
        self.worker = _DummyWorker(self.state)

    def test_shadow_local_not_synced_to_peers(self):
        self.worker._handle_send({"target": "chatgpt", "text": "secret", "shadow_scope": "local"})
        peer_shadow_user = [
            m
            for m in self.state.get_all_messages()
            if m.role == "user" and m.visibility == "shadow" and m.model_key == "gemini"
        ]
        self.assertEqual(peer_shadow_user, [])
        self.assertEqual([c[0] for c in self.worker.calls], ["chatgpt"])

    def test_shadow_shared_syncs_to_peers(self):
        self.worker._handle_send({"target": "chatgpt", "text": "share me", "shadow_scope": "shared"})
        peer_shadow_user = [
            m
            for m in self.state.get_all_messages()
            if m.role == "user" and m.visibility == "shadow" and m.model_key == "gemini"
        ]
        self.assertTrue(peer_shadow_user)
        self.assertEqual(peer_shadow_user[-1].shadow_scope, "shared")
        self.assertEqual([c[0] for c in self.worker.calls], ["chatgpt", "gemini"])


if __name__ == "__main__":
    unittest.main()

