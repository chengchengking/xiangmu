import unittest

from ai_duel_webui import _looks_stale_extracted_reply


class StaleSnapshotGuardTest(unittest.TestCase):
    def test_long_unseen_fragment_is_not_stale(self):
        before = "A old line\nB old line\nC old line\nD old line"
        reply = "A old line\nB old line\nC old line\nD old line\n这是新增的长结论行，包含足够多的新信息用于判断不应视为旧快照复读。"
        self.assertFalse(_looks_stale_extracted_reply(reply, before))


if __name__ == "__main__":
    unittest.main()
