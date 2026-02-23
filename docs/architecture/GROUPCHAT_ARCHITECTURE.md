# AI Group Chat Architecture (Frozen v1)

This document defines the fixed architecture for this project's "human-like free group chat" behavior.
Goal: stop ad-hoc patching, keep one stable model and iterate only within this frame.

## 1) Product Goal (User Perspective)

The system should behave like a normal group chat:

- Any model can speak, stay silent, or rejoin later.
- Group owner can insert a new topic at any time; the room should pivot quickly.
- Models can focus in pairs (or small subsets) while others observe.
- Messages shown in the UI should be clean conversation text (not prompt templates, status pills, or suggestion chips).

## 2) Research-Based Design Inputs

This architecture is based on established multi-party dialogue findings:

1. Multi-party dialogue needs joint modeling of addressee + response, not plain next-turn text generation.
Source: EMNLP 2016, "Addressee and Response Selection for Multi-Party Conversation"  
https://aclanthology.org/D16-1231/

2. Speaker roles (sender/addressee/observer) are dynamic and should be encoded explicitly.
Source: "Addressee and Response Selection in Multi-Party Conversations with Speaker Interaction RNNs"  
https://arxiv.org/abs/1709.04005

3. Multi-party chat often contains concurrent threads; disentanglement is required to avoid topic mixing.
Source: "Online Conversation Disentanglement with Pointer Networks"  
https://arxiv.org/abs/2010.11080

4. Topic transitions in group chat should be tracked explicitly for better response relevance.
Source: EMNLP 2020, "Response Selection for Multi-Party Conversations with Dynamic Topic Tracking"  
https://aclanthology.org/2020.emnlp-main.533/

5. Conversation turn-taking is locally managed and interaction-driven, not rigid round-robin.
Source: Sacks, Schegloff, Jefferson (1974) reference archive  
https://www.conversationanalysis.org/schegloff-media-archive/simplest-systematics-for-turn-taking-language-1974/

6. For modern multi-agent systems, explicit conversation protocol and role separation improves stability.
Source: AutoGen paper  
https://arxiv.org/abs/2308.08155

7. Multi-party agents require semantic + social dynamic modeling (state/action flow).
Source: "Multi-Party Conversational Agents: A Survey"  
https://arxiv.org/abs/2505.18845

## 3) Frozen Core Architecture

### A. Message Plane (Public Timeline)

- Public messages are the only content shown in group view.
- Each message carries:
  - `msg_id`
  - `sender`
  - `timestamp`
  - `visibility` (`public` / `shadow`)
  - `model_key`
  - `thread_hint` (optional)
  - `reply_to` (optional)

### B. Dialogue Control Plane

- Group owner interjection is a **hard event**:
  - immediate enqueue
  - current round can be interrupted
  - next rounds enter "topic lock window"
- Focus mode:
  - default full-room rotation
  - if one model strongly addresses another, temporary 2-model focus is allowed
  - automatic recovery back to full-room after idle focus rounds

### C. Dual-Channel Reply Semantics

- `public`: forwardable visible speech
- `private`: hidden thought/scratchpad (never forwarded to other models in public stream)
- The parser must strip private content from public rendering.

### D. Extraction Sanitization Gate (Mandatory)

Before any model output is added to UI:

1. Strip prompt echoes/instruction mirrors.
2. Strip status-only/process-only text.
3. Strip suggestion chips/tool launchers.
4. Deduplicate repeated blocks.
5. Reject low-value or unfinished segments.

If rejected in auto-rotation, treat as silent pass (`[PASS]`) instead of polluting chat.

### E. Input Integrity Gate (Mandatory)

User input transport must use multi-path decode with quality scoring:

- plain text JSON field
- UTF-8 base64 field
- URI-encoded field

Pick the highest-quality decoded candidate to prevent mojibake (`????`) overwrite.

## 4) Runtime Behavior Rules

### New Topic Rule

- On host interjection:
  - reset focus
  - mark active user topic
  - apply topic lock for at least 2 full rounds
- During topic lock:
  - replies must align to latest host topic
  - obvious off-topic carry-over is dropped

### Fairness Rule

- For 2-model rooms, alternate starter to avoid one side always first.
- For 3+ models, use full-room order with temporary focus override.

### Timeout Rule

- Keep sequential stability but use larger soft timeout for slow web UIs.
- Apply per-room-size cap and per-model soft cap.
- Never block indefinitely; if extract fails, skip safely and continue loop.

## 5) Acceptance Criteria (Must Pass)

1. Topic Pivot Test:
- In active old discussion, host inserts new topic.
- Within 1 round, at least one model responds to new topic.
- Within 2 rounds, majority of visible replies align to new topic.

2. Projection Quality Test:
- No prompt template leak in public messages.
- No status-only text as final visible reply.
- No recommendation-chip text captured as reply body.

3. Latency Test:
- Measure "web response appears" to "UI message appended" latency.
- Track avg/p95 per model (Qwen/Doubao/others).

4. Stability Test:
- Continuous 30+ rounds, no dead loop, no repeated stale extraction flood.

## 6) Non-Goals

- Not pursuing perfect semantic truth validation.
- Not forcing all models to always speak.
- Not exposing private thought channel in public UI.

## 7) Change Control

Any future bug fix must map to one of the layers above.
If a fix requires architecture change, update this file first, then implement.
