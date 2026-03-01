from __future__ import annotations

import unittest

import ai_duel_webui as w


class PromptResendPolicyTest(unittest.TestCase):
    def test_group_public_turn_disables_same_turn_resend(self) -> None:
        self.assertFalse(
            w._allow_prompt_resend_same_turn(
                visibility="public",
                record_reply=False,
            )
        )

    def test_shadow_or_record_turn_allows_resend(self) -> None:
        self.assertTrue(
            w._allow_prompt_resend_same_turn(
                visibility="shadow",
                record_reply=True,
            )
        )
        self.assertTrue(
            w._allow_prompt_resend_same_turn(
                visibility="public",
                record_reply=True,
            )
        )


if __name__ == "__main__":
    unittest.main()

