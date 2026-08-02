## Context

The HL loop measures policy-KL by re-running two agent versions over a frozen
reference set ν and comparing the primitive each emits at each decision point
(`reference.py`, `probe.py`, `distribution.py`). The math and probe are sound;
the symptom is `policy_kl = 0.0` on every act of the real run
(`hl-run-0730-fix`).

Investigation of the candidate (`lostspace/candidates/v1/agent.py`) and the
logic (`AgentBench/.../gamecode_logic/src/player.py`, `communicate.py`) shows
**why** the signal is flat:

- The logic *does* fold `Player.get_legal_actions()` into the in-turn
  `roundbegin` frame (`player.py:227` `ret.update(get_legal_actions())`,
  `communicate.py:115` `inround_mes.update(statedic)`). So the frame the AI
  receives carries the true `attack/move/detect/interprops`.
- But the candidate **ignores** those frame fields. `start_turn()` reads only
  `state/inturn/status/hp/keys/tools/others`. `play()` decides from
  `self.view` / `self.map` (`get_neighbors` → `self.map.elevator_beside`).
- That world model is built **incrementally**: from `birth_pos` (the `id`
  frame) and from the `view` deltas the judger returns inside action-reply
  frames (`move()` → `update_view(self.root["view"])`, agent.py:70).
- The current ν (`reference_seed.py`) sends a **single synthetic `roundbegin`**
  with no frame history. The candidate's map is empty → it emits the same
  degenerate out-of-support primitive (`interact KeyMachine`) regardless of
  edit → both versions uniform → `KL = 0`.

So the fix is not "capture legal_actions" (the candidate doesn't use them) — it
is to present each decision point as the candidate actually encounters it: by
**replaying the real frame transcript** that builds its world model, then
reading the emission at the decision-point `roundbegin`.

Constraints: the candidate is what Claude edits, so the mechanism must be
**edit-agnostic** (work on any candidate speaking the Saiblo protocol, never
depend on candidate internals). The probe must stay hard-timeout-bounded (the
recent hang/speed fixes must hold). ν must stay frozen within a `spec_id` for
cross-version comparability.

## Goals / Non-Goals

**Goals:**
- A ν whose samples carry the real frame transcript up to each decision point,
  recorded from a frozen reference game.
- A probe that replays a transcript through a candidate to obtain a faithful,
  in-support emitted primitive.
- `policy_kl > 0` when an edit changes the candidate's decision logic at a
  reached state (the acceptance signal).

**Non-Goals:**
- On-policy evaluation. ν is frozen; states are those the *reference* policy
  reached (off-policy). Edits that change *which* states are reached are only
  partially visible (via `occupancy_shift`), not via `policy_kl`.
- Changing the candidate. Replay uses the wire protocol only.
- Changing the KL math or the epsilon-smoothed measurement channel.
- Multi-map ν diversity (one frozen game per `spec_id`; benchmark spec already
  fixes opponents/map/seat).

## Decisions

**D1 — Sample schema = ordered transcript prefix.** `ReferenceSample` gains a
`transcript`: the ordered list of judger→AI frames from the `id` frame through
the decision-point `roundbegin` (inclusive). `observation`/`legal_actions`/
`inventory`/`status` remain, taken from the real decision-point frame so `A(s)`
is consistent with the state the candidate sees.
*Alt considered:* store the decision-point observation + a serialized
`self.view`. Rejected — the candidate has no "load view" API and we must not
depend on internals Claude may rewrite. Frame replay is edit-agnostic.
*Alt considered:* store only `legal_actions` + observation (the memory's
original "ν recorder" idea). Rejected — the candidate ignores frame legal
fields; this does not fix KL=0.

**D2 — Probe replays the transcript, not one frame.** `probe_one` spawns a fresh
candidate, then feeds the transcript's frames in order (ASCII-digit framing,
unchanged). The existing reader thread drains the candidate's interleaved
out-frames (its prefix actions) continuously, preventing stdout-buffer
deadlock. After the decision-point `roundbegin` (last in-frame), read the first
emitted primitive — that is the measurement. Feeding the **identical** prefix
to both versions reconstructs the same world model, so emission differences are
pure decision logic — exactly the off-policy `policy_kl` contract.
*Why feeding recorded replies is valid off-policy:* the candidate updates its
model from the judger's reply `view`/`success` fields, not from validating its
own action. So even if the replayed (edited) version "would choose" a different
prefix action, feeding the recorded replies drives its model along the recorded
trajectory. Both versions end at the identical decision-point state.

**D3 — Recorder = parse the match's existing bidirectional trace.**
`run_match` already accepts `trace_path` and writes the **complete
bidirectional transcript** as JSONL (`match.py:519-524`): every judger→AI frame
as `{"state","type":"observation","player","content":<parsed frame JSON>}` and
every AI→judger reply as `{"type":"action",...}`. The evaluator already exposes
`save_traces`. So the recorder needs **zero `match.py` instrumentation** — it is
a pure parser: run one frozen reference roll (seat 0 = the current v1 / sample
AI; opponents = benchmark-spec; fixed `mapconf2.map`) with `trace_path` set,
then walk the trace keeping every `observation` frame for seat 0 in order.

A decision point = a seat-0 `observation` whose frame is a `roundbegin` with
`inturn == 0` and `status == Alive`. Its prefix is **all** seat-0 `observation`
frames up to and including that `roundbegin` — on-turn *and* off-turn
(off-turn `see`/`interprops_status_update`/`getkey` notifications build the
candidate's map too, so they must be replayed). The sample's
`observation`/`legal_actions`/`inventory`/`status` are taken from that
decision-point frame (whose `content` already carries the logic's
`get_legal_actions()` merge — `player.py:227`). `action` trace entries are not
fed during replay (the replayed candidate generates its own); only drained.
Capture **all** Alive decision points from the one roll (Q2).

**D4 — Keep the probe's hard-timeout + prompt-return guarantees.** Replay adds
prefix I/O per sample but is finite (bounded transcript). The worker-thread
hard timeout + force-kill (`probe_one`) and the return-after-first-action rule
at the decision point both still apply unchanged. Cap ν size / prefix rounds so
per-sample replay stays well under the timeout.

**D5 — Retire the hand-authored seed as production ν; keep as test fixture.**
`reference_seed.py` stays for unit tests that need a deterministic single-frame
sample; the controller loads a recorded ν. Loading an old single-frame ν into
the new probe fails fast with a clear "re-record" message (no silent KL=0).

## Risks / Trade-offs

- [Replay deadlock: writing prefix frames while candidate stdout fills]
  → reader thread drains stdout continuously (already present); write frames
  only with the reader active; hard-timeout backstops any stall.
- [Replay slower than single-frame probe] → cap ν sample count and prefix
  length; stratified subsample (early/mid/late rounds) for diversity.
- [Off-policy blind spot: edits changing *reached* states don't show in KL]
  → inherent to frozen ν; `occupancy_shift` partially captures it. Documented,
  not fixed here.
- [Reference trajectory depends on the reference policy chosen] → pin the
  reference policy in the benchmark spec so ν is reproducible; one frozen game
  per `spec_id`.
- [Schema BREAKING for old ν JSON] → fail-fast load error; re-record command
  documented.

## Open Questions

Resolved in brainstorming (2026-07-30):
1. **Capture source** → parse `run_match`'s existing `trace_path` JSONL
   (D3). No `match.py` change; recorder is a pure parser.
2. **ν size** → capture **all** seat-0 Alive decision points from the one
   frozen roll (~20-30/game). Revisit a cap only if probe time regresses.
3. **Reference policy at seat 0** → the current v1 / sample AI (the iterated
   candidate), pinned in the benchmark spec. ν = states the candidate actually
   reaches.
