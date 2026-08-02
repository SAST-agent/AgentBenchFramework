## ADDED Requirements

### Requirement: Reference sample carries the full frame transcript

A `ReferenceSample` SHALL carry the ordered judger→AI frame transcript from the
`id` frame through the decision-point `roundbegin` (inclusive), recorded from a
real game. It SHALL also carry the decision-point `observation`,
`legal_actions`, `inventory`, and `status`, taken from that real
decision-point frame so the enumerated action set `A(s)` is consistent with the
state the candidate reasons over.

#### Scenario: Sample built from a recorded roll contains the prefix

- **WHEN** a reference sample is captured from a frozen reference game at a
  seat-0 Alive roundbegin
- **THEN** the sample's transcript SHALL begin with the `id` frame and end with
  that decision-point `roundbegin`, in chronological order, and its
  `legal_actions`/`inventory`/`status` SHALL equal the fields of that real
  `roundbegin` frame.

#### Scenario: Missing transcript is rejected, not silently coerced

- **WHEN** the replay probe loads a reference sample whose transcript is absent
  (e.g. a legacy hand-authored single-frame ν)
- **THEN** the loader SHALL raise a clear "re-record this ν" error and SHALL
  NOT silently produce a uniform distribution that masks `policy_kl = 0`.

### Requirement: Recorder captures decision points from one frozen reference roll

The recorder SHALL run one frozen reference game — fixed opponents, fixed
`mapconf2.map`, seat 0 played by a pinned reference policy — and capture every
judger→AI frame delivered to seat 0. It SHALL emit one reference sample per
seat-0 `roundbegin` whose `status == Alive` (a genuine decision point); the
sample's transcript is the prefix of captured frames up to and including that
`roundbegin`. Decision points with status Died/Escaped/Skip/Error produce no
sample.

#### Scenario: One sample per reached Alive decision point

- **WHEN** the recorder runs a reference game in which seat 0 is Alive on
  rounds 1, 2, and 4 and is skipped (status Skip) on round 3
- **THEN** it SHALL emit exactly one reference sample for each of rounds 1, 2,
  and 4, and none for round 3.

#### Scenario: Recorder output is reproducible within a spec_id

- **WHEN** the recorder is run twice against the same benchmark spec (same
  opponents, map, seat, reference policy)
- **THEN** the two captured ν sample sets SHALL be identical (same transcript
  prefixes and decision-point frames).

### Requirement: Probe replays the transcript to obtain a faithful emission

The `ReferenceProbe` SHALL, for each sample, spawn a fresh candidate process
and replay the transcript prefix: feed the recorded judger→AI frames in order
using the Saiblo ASCII-digit framing, while continuously draining the
candidate's interleaved out-frames to avoid stdout-buffer deadlock. After the
decision-point `roundbegin` (the final in-frame), it SHALL capture the first
primitive the candidate emits and return it as the sample's emitted action,
flagging it out-of-support if it is not in `A(s)`. The replay SHALL NOT depend
on any candidate-internal API.

#### Scenario: Replayed candidate emits an in-support primitive

- **WHEN** the probe replays a transcript ending at a real decision point
  through the reference candidate that produced the roll
- **THEN** the captured primitive SHALL be in `A(s)` (out_of_support == False),
  because the candidate's world model was faithfully reconstructed.

#### Scenario: Replay feeds identical prefix to both versions

- **WHEN** two candidate versions are probed over the same recorded sample
- **THEN** both SHALL receive the identical transcript prefix, so any difference
  in their emitted primitive reflects only their decision logic.

### Requirement: policy_kl is a real signal under a behavioral edit

For a candidate version pair where the newer version's decision logic differs
from the older's at a state present in the recorded ν, the measured
`policy_kl` over the recorded ν SHALL be greater than zero on at least one
decision point (non-degenerate), instead of collapsing to 0.0 on every sample.

#### Scenario: An edit that changes a reached decision registers KL > 0

- **WHEN** version v2 is derived from v1 by changing the action chosen at a
  decision point that the reference roll reached, and both are probed over the
  recorded ν
- **THEN** `local_policy_kl_trace` SHALL contain at least one strictly-positive
  entry at that decision point.

#### Scenario: An edit that changes nothing registers KL = 0

- **WHEN** v2 is byte-identical to v1 at the decision logic
- **THEN** the policy_kl trace SHALL be all-zero (the signal is discriminative,
  not artificially inflated).

### Requirement: Replay stays within probe time bounds

Transcript replay SHALL stay within the `ReferenceProbe` hard wall-clock
timeout and SHALL still return promptly after capturing the first action at the
decision point. A candidate that stalls during replay SHALL yield a missing
emission (None) within the hard bound, not a hang.

#### Scenario: Stalling candidate during replay returns None in time

- **WHEN** a candidate spawns but blocks during transcript replay and never
  emits at the decision point
- **THEN** `probe_one` SHALL return None within the hard wall-clock timeout
  (worker force-kill), not hang.
