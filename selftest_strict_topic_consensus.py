import unittest

import ai_duel_webui as webui


class StrictTopicConsensusTests(unittest.TestCase):
    def test_threshold_two_models(self):
        self.assertEqual(webui._strict_topic_consensus_threshold(2), 2)

    def test_threshold_three_models(self):
        self.assertEqual(webui._strict_topic_consensus_threshold(3), 3)

    def test_threshold_many_models_caps_at_three(self):
        self.assertEqual(webui._strict_topic_consensus_threshold(5), 3)
        self.assertEqual(webui._strict_topic_consensus_threshold(10), 3)


if __name__ == "__main__":
    unittest.main()

