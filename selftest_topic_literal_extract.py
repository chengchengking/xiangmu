import unittest

import ai_duel_webui as webui


class TopicLiteralExtractTests(unittest.TestCase):
    def test_english_only_this_is_not_treated_as_literal_answer(self):
        q = "NEW TOPIC: ignore previous topic. Answer only this: what is 17*19? give the number only."
        self.assertEqual(webui._extract_expected_short_literal(q), "")

    def test_explicit_literal_still_works(self):
        self.assertEqual(webui._extract_expected_short_literal("answer only BLUE"), "BLUE")


if __name__ == "__main__":
    unittest.main()
