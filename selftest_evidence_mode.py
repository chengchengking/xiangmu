import unittest

from orchestrator.evidence import has_valid_evidence_hook, should_enter_evidence


class EvidenceModeTests(unittest.TestCase):
    def test_valid_evidence_hook_formats(self):
        self.assertTrue(has_valid_evidence_hook("evidence_hook=ref:M#123"))
        self.assertTrue(has_valid_evidence_hook("foo\n evidence_hook=check:verify against rule 3 \nbar"))
        self.assertFalse(has_valid_evidence_hook("evidence_hook=lol"))
        self.assertFalse(has_valid_evidence_hook("evidence_hook=ref:xyz"))

    def test_enter_evidence_requires_recent_conflict(self):
        old_conflicts = [
            "I disagree with that claim",
            "counterpoint: weak evidence",
            "not true under this assumption",
            "let us summarize and proceed",  # recent tail is calm
            "thanks",
            "continue",
        ]
        self.assertFalse(should_enter_evidence(old_conflicts, threshold=3, window=8, recent_guard=3))
        live_conflicts = old_conflicts[:-2] + ["不同意，这里有反例", "证据不足"]
        self.assertTrue(should_enter_evidence(live_conflicts, threshold=3, window=8, recent_guard=3))


if __name__ == "__main__":
    unittest.main()

