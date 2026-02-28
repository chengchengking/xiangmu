import unittest

from ai_duel_webui import _should_pass_no_public_tagged


class ProtocolWrapPolicyTest(unittest.TestCase):
    def test_no_public_tagged_only_when_wrap_expected(self):
        self.assertTrue(_should_pass_no_public_tagged("NO_PUBLIC", "", expect_protocol_wrap=True))
        self.assertFalse(_should_pass_no_public_tagged("NO_PUBLIC", "", expect_protocol_wrap=False))

    def test_no_tags_never_forced_pass(self):
        self.assertFalse(_should_pass_no_public_tagged("NO_TAGS", "", expect_protocol_wrap=True))
        self.assertFalse(_should_pass_no_public_tagged("NO_TAGS", "", expect_protocol_wrap=False))


if __name__ == "__main__":
    unittest.main()

