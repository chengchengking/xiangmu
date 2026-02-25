import unittest

from ai_duel_webui import _PUBLIC_PLACEHOLDER_TOKEN, _protocol_wrap_instruction_suffix


class ProtocolSuffixPlaceholderTests(unittest.TestCase):
    def test_suffix_no_longer_embeds_public_placeholder_token(self):
        s = _protocol_wrap_instruction_suffix(mode="MULTI_ROUND", hidden_reply_hint=True, turn_id=1)
        self.assertNotIn(_PUBLIC_PLACEHOLDER_TOKEN, s)
        self.assertNotIn("在公开回复块中直接写正文", s)
        self.assertIn("[[PUBLIC_REPLY]]", s)
        self.assertIn("[[/PUBLIC_REPLY]]", s)


if __name__ == "__main__":
    unittest.main()
