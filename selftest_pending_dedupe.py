from __future__ import annotations

import unittest

import ai_duel_webui as w


class PendingDedupeTest(unittest.TestCase):
    def test_preserves_order_and_removes_duplicates(self) -> None:
        ids = [18, 19, 18, 20, 19, 21]
        self.assertEqual(w._dedupe_int_ids_preserve_order(ids), [18, 19, 20, 21])

    def test_skips_non_int_values(self) -> None:
        ids = [1, "2", "x", None, 3]
        self.assertEqual(w._dedupe_int_ids_preserve_order(ids), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()

