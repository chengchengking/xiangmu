from __future__ import annotations

import unittest

from ai_duel_webui import _is_unfinished_public_reply_for_model


class DoubaoUnfinishedLeniencyTest(unittest.TestCase):
    def test_keeps_short_but_complete_technical_patch_line(self) -> None:
        txt = "补丁：在独立执行环境外层设置超时与内存上限，超阈值直接终止并标记失败，杜绝脏结果扩散。"
        self.assertFalse(_is_unfinished_public_reply_for_model("doubao", txt))

    def test_still_rejects_process_stub(self) -> None:
        txt = "识别并处理输出中的结构化标记与无法解析干扰。"
        self.assertTrue(_is_unfinished_public_reply_for_model("doubao", txt))


if __name__ == "__main__":
    unittest.main()

