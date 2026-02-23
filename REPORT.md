# REPORT

## Scope

This round focused on making the WebUI group-chat system more maintainable and verifiable without rewriting the Playwright + WebUI architecture.

## What Was Fixed / Added

### 1) Maintainability and repo guardrails
- Added `pyproject.toml` with Black configuration.
- Added CI formatting check in `.github/workflows/basic-checks.yml`.
- Formatted `protocol/`, `orchestrator/`, and key scripts with Black.

### 2) Shadow semantics (shared/local) alignment
- Fixed single-chat peer sync so `shadow_local` is **not** synchronized to peer models.
- Kept `shadow_shared` synchronization behavior for compatibility.
- Added `/api/send` forwarding of `shadow_scope` so HTTP/UI path no longer drops the field.
- Updated README to document `shadow_shared` vs `shadow_local`.

### 3) Protocol leak prevention (public timeline)
- Hardened `_looks_prompt_leak_reply()` to hard-reject control-plane markers:
  - `[[META]]`, `ack_in=`, `packet_hash=`, `evidence_hook=`, etc.
- Added regression test for control-plane leak samples.
- This directly addresses prompt-echo/control-plane leakage into public timeline.

### 4) Broadcast consistency hardening
- Added packet hash consistency guard (`_check_round_packet_hash_consistency`) and isolation path.
- Added META `packet_hash` mismatch isolation in `_run_model_turn()` (if model echoes mismatched hash).
- Added system alert + turn trace error stages for mismatch cases.

### 5) Evidence mode productization (partial but functional)
- Rewrote `orchestrator/evidence.py` into readable/maintainable form.
- Added `has_valid_evidence_hook()` with minimum accepted formats:
  - `evidence_hook=ref:M#123`
  - `evidence_hook=check:<...>`
- Tightened EVIDENCE-mode gate to PASS disagreements without valid evidence hook.
- Added recent-conflict guard to reduce “stuck in EVIDENCE forever” behavior.

### 6) Selected model state persistence
- Added `.tmp/selected_models.json` persistence for selected model slots.
- Restores selected state on restart (does not persist auth cookies here; browser profile still handles auth).

### 7) Regression / selftest tooling
- `selftest_protocol_envelope.py` (existing; retained)
- `selftest_webui_api_smoke.py` (new)
- `tools/leak_scan.py` (new)
- `tools/group_regression_20r.py` (new)
- `tools/adapter_contract_smoke.py` (new)
- Plus focused unit/selftests for:
  - `shadow_scope`
  - packet hash guard
  - evidence mode
  - public leak guard
  - selected model persistence

## Validation Performed

### Basic compile / format
- `python -m black --check protocol orchestrator selftest_protocol_envelope.py ai_duel.py model_adapters.py`
- `python -m py_compile ai_duel.py ai_duel_webui.py model_adapters.py`
- `python -m py_compile protocol\envelope.py orchestrator\receipt.py orchestrator\evidence.py`

### Selftests
- `python selftest_shadow_scope.py`
- `python selftest_shadow_sync_isolation.py`
- `python selftest_packet_hash_guard.py`
- `python selftest_evidence_mode.py`
- `python selftest_public_leak_guard.py`
- `python selftest_selected_models_persistence.py`

### WebUI/API smoke
- `python selftest_webui_api_smoke.py`
  - starts WebUI
  - checks `/api/state`
  - POST `/api/send`
  - validates system message path and stops server

### Real group-chat regression (live, 5 selected models already logged in)
- Ran `tools/group_regression_20r.py` against active `127.0.0.1:8765`
- Confirmed:
  - `packet_hash_mismatch_turns=[]` (broadcast consistency held in tested runs)
  - after leak-guard fix, follow-up short regression returned `public_leak_hits=0`

## Remaining Risks / Known Gaps

1. **Packet hash round guard is mostly defensive**
- Current implementation computes one authoritative packet per round, so mismatch is unlikely by construction.
- Guard is still valuable for future refactors and META-echo mismatch isolation.

2. **EVIDENCE mode is improved but not yet a full persistent orchestrator mode**
- Trigger/cooling is applied per turn based on recent messages.
- A full stateful EVIDENCE entry/exit policy across rounds can still be improved.

3. **Adapter contract smoke is static by default**
- It validates method contract and standardized failure categories.
- Live adapter smoke (selector/captcha/timeout classification by site) still needs a runtime harness using logged-in browser sessions.

4. **20-round regression depends on logged-in + selected models**
- Script works and produces metrics, but live execution requires existing authenticated browser sessions.

## How To Run the 3 Required Regression Scripts

### A) Protocol parser tests
```powershell
python selftest_protocol_envelope.py
```

### B) WebUI API smoke
```powershell
python selftest_webui_api_smoke.py
```

### C) 20-round group regression + metrics
Use an already running WebUI with selected/logged-in models:
```powershell
python tools\group_regression_20r.py --port 8765 --rounds 20 --wait-seconds 360
```

Analyze-only mode (no message send):
```powershell
python tools\group_regression_20r.py --port 8765 --analyze-only
```

