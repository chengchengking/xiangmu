from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import ai_duel_webui as webui


class SelectedModelsPersistenceTests(unittest.TestCase):
    def test_load_and_save_selected_models(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "selected_models.json"
            old = webui.SELECTED_MODELS_FILE
            try:
                webui.SELECTED_MODELS_FILE = f
                f.write_text(json.dumps({"selected_keys": ["chatgpt", "qwen"]}), encoding="utf-8")
                st = webui.SharedState()
                self.assertEqual(set(st.selected_keys()), {"chatgpt", "qwen"})

                g = st.get_model("gemini")
                assert g is not None
                g.integrated = True
                g.authenticated = True
                res = st.toggle_selected("gemini")
                self.assertTrue(res.get("ok"))
                data = json.loads(f.read_text(encoding="utf-8"))
                self.assertIn("gemini", set(data.get("selected_keys") or []))
            finally:
                webui.SELECTED_MODELS_FILE = old


if __name__ == "__main__":
    unittest.main()

