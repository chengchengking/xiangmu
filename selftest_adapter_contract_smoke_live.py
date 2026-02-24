from __future__ import annotations

import tools.adapter_contract_smoke as smoke


def test_pick_metas_filters_known_models() -> None:
    metas = smoke._pick_metas(["chatgpt", "gemini"])
    keys = [m.key for m in metas]
    assert "chatgpt" in keys and "gemini" in keys


def test_live_probe_no_models() -> None:
    out = smoke.live_probe(model_keys=["__no_such_model__"], timeout_s=0.1)
    assert out["ok"] is False
    assert out["mode"] == "live"
    assert out["error"] == "no_models_selected"


def main() -> int:
    test_pick_metas_filters_known_models()
    test_live_probe_no_models()
    print("OK: adapter_contract_smoke live helpers passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

