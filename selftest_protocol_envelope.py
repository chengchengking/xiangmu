from __future__ import annotations

import unittest

from protocol.envelope import parse_envelope


class EnvelopeParserTests(unittest.TestCase):
    def test_full_blocks(self) -> None:
        raw = """
[[META]]
turn_id=12
mode=SINGLE_FAST
reply_to=31
ack=watermark:31
[[/META]]

[[PUBLIC_REPLY]]
hello public
[[/PUBLIC_REPLY]]

[[PRIVATE_REPLY]]
hello private
[[/PRIVATE_REPLY]]
"""
        env = parse_envelope(raw)
        self.assertEqual(env.meta.get("turn_id"), "12")
        self.assertEqual(env.meta.get("mode"), "SINGLE_FAST")
        self.assertEqual(env.public, "hello public")
        self.assertEqual(env.private, "hello private")
        self.assertTrue(env.status.startswith("OK"))

    def test_public_only(self) -> None:
        raw = "[[PUBLIC_REPLY]]only public[[/PUBLIC_REPLY]]"
        env = parse_envelope(raw)
        self.assertEqual(env.meta, {})
        self.assertEqual(env.public, "only public")
        self.assertEqual(env.private, "")
        self.assertEqual(env.status, "OK")

    def test_private_only(self) -> None:
        raw = "[[PRIVATE_REPLY]]private text[[/PRIVATE_REPLY]]"
        env = parse_envelope(raw)
        self.assertEqual(env.public, "")
        self.assertEqual(env.private, "private text")
        self.assertTrue(env.status.startswith("NO_PUBLIC"))

    def test_multiple_public_uses_last(self) -> None:
        raw = """
[[PUBLIC_REPLY]]first[[/PUBLIC_REPLY]]
noise
[[PUBLIC_REPLY]]second[[/PUBLIC_REPLY]]
"""
        env = parse_envelope(raw)
        self.assertEqual(env.public, "second")
        self.assertIn("MULTI_PUBLIC", env.status)

    def test_meta_ignores_bad_lines(self) -> None:
        raw = """
[[META]]
turn_id=9
bad line
=empty_key
mode=MULTI_ROUND
[[/META]]
[[PUBLIC_REPLY]]ok[[/PUBLIC_REPLY]]
"""
        env = parse_envelope(raw)
        self.assertEqual(env.meta, {"turn_id": "9", "mode": "MULTI_ROUND"})
        self.assertEqual(env.public, "ok")
        self.assertEqual(env.status, "OK")

    def test_no_tags(self) -> None:
        env = parse_envelope("plain text without any envelope")
        self.assertEqual(env.meta, {})
        self.assertEqual(env.public, "")
        self.assertEqual(env.private, "")
        self.assertEqual(env.status, "NO_TAGS")

    def test_salvage_public_before_meta_when_public_close_missing(self) -> None:
        raw = (
            "[[PUBLIC_REPLY]]有效公开建议：先修 parser 抓取，再做广播一致性。"
            "[[META]]\nturn_id=8\nmode=MULTI_ROUND\n[[/META]]"
        )
        env = parse_envelope(raw)
        self.assertIn("MALFORMED", env.status)
        self.assertIn("有效公开建议", env.public)


if __name__ == "__main__":
    unittest.main()
