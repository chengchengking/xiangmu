import unittest

import ai_duel_webui as webui


class _FakePage:
    def __init__(self, fail_reload=False):
        self.fail_reload = fail_reload
        self.reloaded = 0
        self.closed = 0
        self.gotos = 0

    def reload(self, **kwargs):
        self.reloaded += 1
        if self.fail_reload:
            raise RuntimeError("reload failed")

    def close(self):
        self.closed += 1

    def goto(self, *args, **kwargs):
        self.gotos += 1


class _FakeContext:
    def __init__(self):
        self.new_pages = 0
        self.created = []

    def new_page(self):
        self.new_pages += 1
        p = _FakePage()
        self.created.append(p)
        return p


class _FakeAdapter:
    def __init__(self, page):
        self.meta = type("Meta", (), {"url": "https://example.com"})()
        self.page = page
        self.context = _FakeContext()

    def ensure_page(self, p):
        return self.page


class WorkerRecoveryLadderTests(unittest.TestCase):
    def test_level1_reload_success(self):
        w = webui.Worker(webui.SharedState())
        w._pw = object()  # bypass ensure playwight path
        w._sp = object()
        p = _FakePage(fail_reload=False)
        ad = _FakeAdapter(p)
        self.assertTrue(w._repair_adapter_surface("qwen", ad, level=1))
        self.assertEqual(p.reloaded, 1)

    def test_level2_rebuild_tab_when_reload_fails(self):
        w = webui.Worker(webui.SharedState())
        w._pw = object()
        w._sp = object()
        p = _FakePage(fail_reload=True)
        ad = _FakeAdapter(p)
        self.assertTrue(w._repair_adapter_surface("qwen", ad, level=2))
        self.assertEqual(p.closed, 1)
        self.assertEqual(ad.context.new_pages, 1)
        self.assertIsNotNone(ad.page)


if __name__ == "__main__":
    unittest.main()

