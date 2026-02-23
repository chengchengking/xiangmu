# RELEASE CHECKLIST

## 1) Environment / startup
- [ ] `python -m pip install -r requirements.txt`
- [ ] `python -m playwright install chromium` (or use `install_playwright_chromium_fast.ps1`)
- [ ] `python -m py_compile ai_duel.py ai_duel_webui.py model_adapters.py`
- [ ] `python ai_duel_webui.py` starts and `/api/state` returns `ok=true`
- [ ] `start_webui.bat` launches backend and opens UI window
- [ ] `stop_webui.bat` stops backend cleanly

## 2) Formatting / repo hygiene
- [ ] `python -m black --check protocol orchestrator selftest_protocol_envelope.py ai_duel.py model_adapters.py`
- [ ] CI `basic-checks` passes

## 3) Protocol / leakage safety
- [ ] `python selftest_protocol_envelope.py`
- [ ] `python selftest_public_leak_guard.py`
- [ ] `python tools\leak_scan.py --since-unix <test_start_ts>`
- [ ] Public timeline contains no control-plane markers:
  - `【系统回执】`
  - `[[META]]`
  - `[[PUBLIC_REPLY]]`
  - `[[PRIVATE_REPLY]]`
  - `evidence_hook=`
  - `【广播包 ...】`

## 4) WebUI API / basic messaging
- [ ] `python selftest_webui_api_smoke.py`
- [ ] `/api/send` accepts `shadow_scope=shared|local`

## 5) Shadow semantics
- [ ] `python selftest_shadow_scope.py`
- [ ] `python selftest_shadow_sync_isolation.py`
- [ ] `python selftest_selected_models_persistence.py`
- [ ] Verify `.tmp/selected_models.json` updates after toggling model selection
- [ ] Restart WebUI and confirm selected model state restores

## 6) Orchestrator correctness
- [ ] `python selftest_packet_hash_guard.py`
- [ ] `python selftest_evidence_mode.py`
- [ ] Check turn log has `receipt_final` (not only provisional) in group auto-rotation path
- [ ] No case of `receipt_final=ACCEPT` followed by outer `pass(...)` for same model turn

## 7) Broadcast consistency (authoritative packet)
- [ ] Run group regression with at least 2 selected models
- [ ] `packet_hash_mismatch_turns=[]` in metrics report
- [ ] If mismatch occurs:
  - [ ] system alert visible in UI
  - [ ] turn log contains `ERROR(packet_hash_mismatch)` / `ERROR(packet_hash_mismatch_meta)`
  - [ ] offending output isolated (`PASS`/`REJECT`)

## 8) Adapter contract sanity
- [ ] `python tools\adapter_contract_smoke.py`
- [ ] (Optional live) run adapter-specific smoke in logged-in environment and classify failures:
  - `selector_miss`
  - `timeout`
  - `captcha`
  - `stale`

## 9) 20-round regression (live)
- [ ] `python tools\group_regression_20r.py --port 8765 --rounds 20 --wait-seconds 360`
- [ ] Inspect report:
  - [ ] `public_leak_hits == 0`
  - [ ] `packet_hash_mismatch_turns == []`
  - [ ] `receipt_status` distributions look reasonable
  - [ ] `reject_reason` distribution not dominated by unexpected categories

## 10) Privacy / logging
- [ ] Confirm `.tmp/` logs are not committed
- [ ] Confirm `user_data/` (browser profiles) is not committed
- [ ] If sharing logs, redact sensitive content / cookies / identifiers

