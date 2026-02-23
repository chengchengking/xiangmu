import unittest

import ai_duel_webui as webui


class PacketHashGuardTests(unittest.TestCase):
    def test_consistent_hashes_pass(self):
        seen = {}
        ok1, exp1 = webui.Worker._check_round_packet_hash_consistency(seen, "chatgpt", "abc123")
        ok2, exp2 = webui.Worker._check_round_packet_hash_consistency(seen, "gemini", "abc123")
        self.assertTrue(ok1)
        self.assertTrue(ok2)
        self.assertEqual(exp1, "abc123")
        self.assertEqual(exp2, "abc123")

    def test_mismatch_is_detected(self):
        seen = {}
        webui.Worker._check_round_packet_hash_consistency(seen, "chatgpt", "aaa111")
        ok, expected = webui.Worker._check_round_packet_hash_consistency(seen, "gemini", "bbb222")
        self.assertFalse(ok)
        self.assertEqual(expected, "aaa111")


if __name__ == "__main__":
    unittest.main()

