## Why

The HL iteration loop's learning signal is flat: `policy_kl = 0.0` and
`win_rate = 0.0` across all acts of the real run (`hl-run-0730-fix`). Root cause:
the hand-authored reference set ν (`reference_seed.py`) presents each decision
point as a **single synthetic `roundbegin` frame**, but the LostSpace candidate
builds its world model (`self.view` / `self.map`) incrementally from `birth_pos`
plus the `view` deltas the judger returns in action-reply frames. With no
preceding frame history, the candidate's map is empty, so it emits the same
degenerate out-of-support primitive (`interact KeyMachine`) regardless of any
code edit. Both versions collapse to uniform → `policy_kl = 0`. The probe and
KL math are correct; the ν is not faithful to the states the candidate actually
reasons over. We need a ν recorded from a **real** reference game, replayed
end-to-end through the candidate, so an edit that changes the decision logic
registers as `KL > 0`.

## What Changes

- Add a **reference recorder**: run a frozen reference game (fixed opponents,
  fixed `mapconf2.map`, seat 0) and capture the full bidirectional frame
  transcript up to selected decision points for seat 0 — the `id` frame, all
  `offround`/in-round notifications, and the judger's action-reply frames
  (which carry the `view` deltas that build the candidate's map), through the
  decision-point `roundbegin`.
- Extend the `ReferenceSample` schema to carry that **transcript prefix** (the
  ordered frames + recorded replies), not just a single observation. The
  `legal_actions`/`inventory`/`status` fields remain (taken from the real
  decision-point frame, so `A(s)` is consistent with the state the candidate
  sees). **BREAKING** for existing ν JSON files and the hand-authored seed.
- Change `ReferenceProbe` to **replay the transcript prefix** for each sample:
  feed the recorded frames in order, supply the recorded judger replies when
  the candidate blocks on a per-action read, then capture the emitted primitive
  at the decision-point `roundbegin`. Feeding the identical prefix to both
  versions reconstructs the same world model, so emissions differ only due to
  the edited decision logic — the clean off-policy `policy_kl` measurement the
  design intends.
- Retire `reference_seed.py`'s hand-authored synthetic seed as the production ν
  source (keep it only as a test fixture / fallback). Wire the controller to
  load a recorded ν.

## Capabilities

### New Capabilities
- `reference-set`: The frozen reference distribution ν for policy-KL — how
  decision-point samples are **captured from a real reference game** (full frame
  transcript per decision point), the sample schema, and how the probe
  **replays** a transcript through a candidate to obtain a faithful, in-support
  emitted primitive. Replaces the hand-authored single-frame seed with recorded
  transcripts so policy-KL is a real signal.

### Modified Capabilities
<!-- openspec/specs/ is empty in this repo; reference-set is introduced new. -->

## Impact

- `src/agentbench_frame/hl/reference.py` — `ReferenceSample` gains a transcript
  field; serialization changes (BREAKING for old ν JSON).
- `src/agentbench_frame/hl/probe.py` — `ReferenceProbe` replays a transcript
  prefix instead of sending one synthetic `roundbegin`.
- `src/agentbench_frame/hl/reference_seed.py` — demoted to test fixture; new
  recorder module produces the recorded ν (source TBD in design: parse an
  existing harness replay log vs. instrument a dedicated reference roll).
- `src/agentbench_frame/hl/controller.py` — loads the recorded ν; the
  `_probe_version` / `_measure_policy_kl` path unchanged in shape.
- Tests: `tests/hl/test_probe.py`, `test_reference_seed.py`, plus new recorder
  tests. Probe hard-timeout + prompt-return behavior (recently fixed) must hold
  under the replay path.
- No change to the candidate (`candidates/v1/agent.py`) — replay is
  edit-agnostic (works on any candidate speaking the Saiblo protocol).
