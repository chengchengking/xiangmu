import unittest

import ai_duel_webui as webui
from model_adapters import DoubaoAdapter, ModelMeta


class ReplyFilterGuardsTests(unittest.TestCase):
    def test_should_pass_no_public_tagged(self):
        self.assertTrue(webui._should_pass_no_public_tagged("NO_PUBLIC", ""))
        self.assertTrue(webui._should_pass_no_public_tagged("MALFORMED", ""))
        self.assertFalse(webui._should_pass_no_public_tagged("NO_TAGS", ""))
        self.assertFalse(webui._should_pass_no_public_tagged("OK", "real answer"))

    def test_doubao_clean_candidate_filters_format_prompt_line(self):
        meta = ModelMeta(
            slot=4,
            key="doubao",
            name="豆包",
            url="https://www.doubao.com/chat/",
            color="#a855f7",
            integrated=True,
            avatar_url="",
            login_help="",
        )
        ad = DoubaoAdapter(meta)
        raw = (
            "请按以下格式输出（不要复述本提示）：\n"
            "[[META]]\nturn_id=1\n[[/META]]\n"
            "[[PUBLIC_REPLY]]\n\n[[/PUBLIC_REPLY]]\n"
            "[[PRIVATE_REPLY]]可选[[/PRIVATE_REPLY]]"
        )
        self.assertEqual(ad._clean_candidate_text(raw), "")


if __name__ == "__main__":
    unittest.main()

