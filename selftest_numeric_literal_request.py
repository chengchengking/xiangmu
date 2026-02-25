import unittest

import ai_duel_webui as webui


class NumericLiteralRequestTests(unittest.TestCase):
    def test_number_only_math_promotes_strict_numeric_literal(self):
        q = "NEW TOPIC: what is 17*19? give the number only."
        self.assertEqual(webui._extract_expected_numeric_literal_if_requested(q), "323")

    def test_normal_math_without_number_only_does_not_force_literal(self):
        q = "what is 17*19? explain briefly"
        self.assertEqual(webui._extract_expected_numeric_literal_if_requested(q), "")


if __name__ == "__main__":
    unittest.main()
