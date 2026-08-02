# Plan — recorded-reference-set

> Implementation plan for `openspec/changes/recorded-reference-set`. Every
> task traces to `tasks.md` (§1–6) and to `specs/reference-set/spec.md`
> requirements. Brainstorming already resolved the open questions (see
> `design.md` §Open Questions) — this plan does not re-decide them.
>
> Branch: `liuzhuo/lostspace`. TDD: write the test first, implement to pass.
> All Python via `uv run` with `PYTHONPATH=src`. Never edit `match.py` /
> `ladder.py` / `evaluator.py` logic behavior — the recorder is a pure parser.

## Anchors (verified before planning)

- `match.py:430-437` — every judger→AI frame is appended to the trace as
  `{"state":<state>, "type":"observation", "player":<int>, "content":<parsed frame dict>}`.
- `match.py:486-494` — every AI→judger reply is appended as
  `{"state":<state>, "type":"action", "player":<int>, "content":<action>}`.
- `match.py:519-524` — the trace list is serialized to `trace_path` as JSONL
  (one JSON object per line) in the `finally` block.
- `evaluator.py:91,133,164,184-202,223-224` — `save_traces=True` sets
  `trace_path = <artifact_dir>/<stem>.trace.jsonl` per match and records
  `record["trace"]` (relative path) in `matches.jsonl`.
- `reference.py:41-70` — `ReferenceSample` dataclass (frozen): `observation`,
  `legal_actions`, `inventory`, `status`, `seat`, `opponent` (+ `to_dict` /
  `from_dict`). No `transcript` field yet.
- `probe.py:201-317` — `probe_one` currently sends ONE synthesized `roundbegin`
  (built from `sample.observation` via `frame.setdefault(...)` at 258-272) and
  captures the first action. This is the single-frame path being replaced.
- `cli.py:271-272,390,432` — `--reference PATH` loads `ReferenceStateSet.load`
  and passes it to the controller. Unchanged by this plan.
- `reference_seed.py` — hand-authored 8-sample seed; becomes a test fixture
  (its samples get an empty `transcript` so unit tests of `ReferenceSample`
  round-tripping still work, but the probe rejects them — see §1.4).

## Terminology

- **judger→AI frame** = a dict the candidate reads off stdin (ASCII-4-digit
  length + JSON). In the trace these are `type:"observation"` entries.
- **decision point** = a seat-0 judger→AI frame whose `content.type ==
  "roundbegin"` AND `content.inturn == 0` AND `content.status == Alive (0)`.
- **transcript prefix** = the ordered list of ALL seat-0 `type:"observation"`
  trace entries from the first one (the `id` frame) through the decision-point
  `roundbegin` (inclusive), each reduced to its `content` dict.

---

## §1. ReferenceSample schema + transcript  (tasks 1.1–1.3)

### 1.1 Add `transcript` field  →  `reference.py`

Edit `ReferenceSample` (`reference.py:41-70`):

- Add field `transcript: Tuple[Dict[str, Any], ...] = ()` (default empty tuple
  so existing unit-test constructions that omit it still construct; the PROBE
  path enforces non-empty — see §3).
- `__post_init__`-style freeze is unnecessary (frozen dataclass; tuples are
  immutable). But `to_dict` / `from_dict` / `ReferenceStateSet.save` / `.load`
  must round-trip it.
- `to_dict()`: add `"transcript": [dict(f) for f in self.transcript]`.
- `from_dict()`: read `d.get("transcript") or ()` and freeze to
  `tuple(dict(f) for f in ...)`.
- `ReferenceStateSet.save` already serializes via `to_dict` — no extra change
  beyond §1.1's `to_dict`.
- `ReferenceStateSet.load` already goes via `from_dict` — likewise covered.

### 1.2 Test: round-trip preserves transcript + decision-point fields  →  `tests/hl/test_reference.py`

New test `test_reference_sample_transcript_roundtrip`:
- Build a `ReferenceSample` with a 3-frame `transcript`: an `id` frame, an
  off-turn `see` notification, a `roundbegin` decision-point frame. Set
  `legal_actions`/`inventory`/`status` to fields taken FROM that roundbegin
  frame (so they are consistent).
- `save` to a tmp file, `load` it back.
- Assert: `loaded.transcript == original.transcript` (same length, same dicts,
  same order); `loaded.legal_actions == original.legal_actions`;
  `loaded.status == original.status`; `loaded.inventory == original.inventory`.

### 1.3 Test: missing transcript rejected, not silently coerced  →  `tests/hl/test_reference.py`

New test `test_missing_transcript_raises_not_silent`:
- Build a `ReferenceSample` with `transcript=()` (the legacy single-frame shape).
- Assert the bare constructor and `from_dict` ACCEPT it (unit fixtures and
  `reference_seed.py` rely on this) and round-trip to `()`. The fail-fast is at
  the PROBE boundary, added in §3.1 — tested there, not here. Leave a TODO
  pointing to §3. (The spec requirement "Missing transcript is rejected" is
  satisfied by the probe in §3, not by the dataclass in §1 — `ReferenceSample`
  is a dumb data carrier.)

> Note: the bare constructor and `from_dict` MUST still accept `transcript=()`
> (unit fixtures and `reference_seed.py` rely on it). The fail-fast is at the
> PROBE boundary, not at construction — this keeps `ReferenceSample` a dumb
> data carrier. The spec scenario "Missing transcript is rejected" is satisfied
> by the probe rejecting it.

---

## §2. Recorder: parse the match trace into ν  (tasks 2.1–2.5)

### 2.1 New `hl/reference_recorder.py` (pure parser)  →  new file

Module docstring states: pure parser of a `run_match` trace JSONL; no
`match.py`/logic change. Public function:

```python
def record_reference_states(
    trace_path: Path, *, spec_id: str, opponent: str, seat: int = 0,
) -> ReferenceStateSet
```

Algorithm (implement exactly):

1. Read the trace JSONL line by line (`json.loads` each non-empty line).
2. Collect, in file order, every entry where `e["type"] == "observation"` and
   `int(e["player"]) == seat`. Keep each as its `content` dict. This is the
   seat's full judger→AI frame stream. (Entries of `type:"action"` / `"ai_error"`
   are skipped — the candidate regenerates its own actions; only judger→AI
   frames are replayed.)
3. Walk this ordered stream. A **decision point** is a frame `f` where
   `f.get("type") == "roundbegin"` and `int(f.get("inturn", -1)) == 0` and
   `int(f.get("status", -1)) == STATUS_ALIVE` (`STATUS_ALIVE = 0`, import from
   `distribution`). Frames with `status` in
   `{DIED, ESCAPED, SKIP, ERROR, WAIT_FOR_ESCAPE}` are NOT decision points
   (consistent with `enumerate_legal_actions`).
4. For each decision point at index `k` in the stream: the **transcript prefix**
   = `tuple(stream[0 : k+1])` — ALL seat-0 observation frames from the `id`
   frame through this `roundbegin` (inclusive). On-turn AND off-turn frames
   both go in (off-turn `see`/`getkey`/`interprops_status_update` notifications
   build the candidate's map — they must be replayed).
5. The sample's `observation` / `legal_actions` / `inventory` / `status` are
   taken FROM the decision-point frame `f`:
   - `observation = dict(f)` (the full roundbegin content; the candidate reads
     top-level `inturn/status/state/hp/keys/tools/others`).
   - `legal_actions` = `{"attack": f.get("attack", []), "move": f.get("move", []),
     "detect": f.get("detect", False), "interprops": f.get("interprops", [])}`
     (the logic merges `get_legal_actions()` into the roundbegin — design D3).
   - `inventory` = derived from `f.get("tools", {})`: `{"LandMine": <count>,
     "Sticky": <count>, "Transport": <int>, "Kit": <int>}`. The roundbegin
     `tools` field shape is `{"LandMine":[made,used], "Sticky":[...], "Kit":int,
     "Transport":int}` (see `reference_seed.py:50-53` / `context.py` blurb) —
     inventory count = `tools[name][0] - tools[name][1]` for traps,
     `tools[name]` for Kit/Transport. Mirror `enumerate_legal_actions`'s
     expectation (it reads `inv.get("LandMine")` etc. as ints).
   - `status = int(f.get("status", 0))`, `seat = seat`, `opponent = opponent`.
6. Emit a `ReferenceStateSet(spec_id=spec_id, samples=tuple(samples))` with one
   `ReferenceSample` per decision point. Include ALL Alive decision points from
   the roll (no subsampling in v1 — design D4 / Open Q 2).

Edge cases:
- A trace with zero seat-0 `roundbegin` Alive frames → emit an empty
  `ReferenceStateSet` (zero samples) and log a warning; do NOT raise (a stub
  reference policy that never gets a turn is a legitimate, if useless, result).
- A frame whose `content` is not a dict (e.g. a raw string on a malformed
  trace) → skip it defensively (log), never raise.

### 2.2 CLI  →  `hl/reference_recorder.py` `main`

`python -m agentbench_frame.hl.reference_recorder --trace <trace.jsonl>
--spec-id <id> --opponent <name> [--seat 0] --out nu.json`

- argparse: `--trace` (Path, req), `--spec-id` (req), `--opponent` (req),
  `--seat` (int, default 0), `--out` (Path, req).
- Call `record_reference_states(...)`, `.save(args.out)`.
- Print `wrote N reference samples to <out> (spec_id=...)` and `return 0`.
- Add the module to the `hl` package's `__main__`-style discoverability if
  needed (it is runnable via `python -m agentbench_frame.hl.reference_recorder`
  by virtue of having `if __name__ == "__main__": raise SystemExit(main())`).

### 2.3 Test: synthetic trace → correct samples  →  `tests/hl/test_reference_recorder.py` (new)

`test_recorder_emits_one_sample_per_alive_decision_point`:
- Build a synthetic trace JSONL (list of dicts written one per line) where seat 0
  receives, in order:
  - `id` frame: `{"type":"id","id":0,"birth_pos":[0,0]}`
  - round 1 `roundbegin`: `inturn=0, status=0 (Alive)` with some `move`/`attack`.
  - off-turn `see` notification (player 0, `type:"see"` — included in prefix).
  - round 2 `roundbegin`: `inturn=0, status=0`.
  - round 3 `roundbegin`: `inturn=0, status=3 (Skip)` — NOT a decision point.
  - round 4 `roundbegin`: `inturn=0, status=0`.
  Plus an `action` entry (player 0) somewhere — must be ignored by the recorder.
- Run `record_reference_states`.
- Assert: exactly 3 samples (rounds 1, 2, 4); round 3 (Skip) produced none.
- Assert each sample's transcript begins with the `id` frame and ends with its
  own `roundbegin`, and includes the off-turn `see` frame in round 2's prefix.

### 2.4 Test: decision-point fields come from the roundbegin  →  same file

`test_recorder_sample_fields_match_roundbegin`:
- Synthetic trace with one Alive roundbegin whose `content` has
  `attack=[1]`, `move=[T,F,...]`, `detect=True`, `interprops=["KeyMachine"]`,
  `tools={...}`.
- Assert the emitted sample's `legal_actions` equals that dict (projection),
  `status==0`, and `inventory` matches the derived counts.

### 2.5 Test: recorder is deterministic  →  same file

`test_recorder_is_deterministic`:
- Same trace input → two calls produce `ReferenceStateSet`s whose
  `samples` tuples are equal element-wise (compare `to_dict()` lists).
  (The recorder reads no clocks, no randomness — `Date.now`/`random` are not
  used. Determinism is structural.)

---

## §3. Probe replay  (tasks 3.1–3.5)

### 3.1 Refactor `ReferenceProbe.probe_one` to replay the transcript  →  `probe.py`

This is the core change. Replace the single-frame body
(`probe.py:239-317` `_probe_one_impl`) with a transcript-replay body.

**New `_probe_one_impl(sample, las)`:**

1. If `len(sample.transcript) == 0`: raise `ReferenceSampleError(
   "reference sample has no transcript — re-record this ν (legacy single-frame "
   "ν cannot build the candidate's world model)")`. This is the §1.3 fail-fast.
   (Define `ReferenceSampleError(ValueError)` at module top in `probe.py`.)
2. Spawn a fresh candidate (`_start()` unchanged — sends the `id` frame from
   `_start`, NOT from the transcript... see note below).
   - **NOTE / decision to make at implementation:** `_start()` currently sends a
     hardcoded `{"type":"id","id":0,"birth_pos":[0,0]}` and drains the ack
     (`probe.py:194-199`). But the recorded transcript's FIRST frame IS the
     real `id` frame. To be faithful, `_start` should send the recorded `id`
     frame (i.e. `transcript[0]`) instead of a hardcoded one, and the
     transcript fed in step 3 should be `transcript[1:]` (skip the already-sent
     `id`). **Implement it this way**: refactor `_start` to accept the `id`
     frame to send (defaulting to the hardcoded one for non-replay callers /
     tests that don't pass a transcript). Concretely:
     - `_start(self, id_frame: Optional[dict] = None)`: if `id_frame` is None,
       use the current hardcoded `{"type":"id","id":0,"birth_pos":[0,0]}`;
       else `_write_frame(self._proc.stdin, id_frame)`.
     - In `_probe_one_impl`: `id_frame = sample.transcript[0] if
       sample.transcript and sample.transcript[0].get("type")=="id" else None`;
       call `self._start(id_frame=id_frame)`; then `prefix =
       sample.transcript[1:]` if an `id` frame was consumed, else
       `sample.transcript`.
     - If `transcript[0]` is NOT an `id` frame, send the default `id` then feed
       the whole transcript (defensive — the recorder always emits `id` first,
       but don't crash on a hand-built transcript).
3. Feed the remaining transcript frames in order via `_write_frame`. The reader
   thread (`_read_loop`, already running from `_start`) continuously drains the
   candidate's interleaved out-frames (its prefix actions) so the candidate's
   stdout never fills and blocks the write. **Crucial**: keep feeding even while
   the candidate emits actions — its emissions are drained, not replied to. We
   feed the RECORDED judger frames (which carry the `view` deltas that build the
   candidate's map), exactly as design D2/D3 specify.
4. The LAST frame in the transcript is the decision-point `roundbegin`. After
   feeding it, capture the first primitive the candidate emits (same
   first-action rule as today, `probe.py:291-312`): loop on `_read_frame`,
   take the first `type=="action"` frame, decode its `action` list, flag
   `out_of_support` if the token is not in `las.tokens`, break. If the
   candidate emits `finish` first, record `FINISH`.
5. Hard wall-clock timeout: keep the `probe_one` worker-thread + `join(timeout)`
   + `_force_kill` structure (`probe.py:201-227`) UNCHANGED. The body budget
   grows with prefix length; bump the worker join deadline to
   `self.timeout + self._ACK_DRAIN + <prefix_feed_slack>` where
   `prefix_feed_slack` scales with transcript length (e.g.
   `min(self.timeout, 0.05 * len(transcript))`). Document the bound.
6. Return `EmittedAction(primitive=..., out_of_support=..., sample_index=...)`
   as today; `None` only for a truly unresponsive candidate (no emission at the
   decision point within the hard bound).

**What stays unchanged:** `_write_frame` / `_read_exact` / `_read_loop` /
`_read_frame` / `_force_kill` / `_close_proc` / the per-sample fresh-process
discipline / `probe_set`. The asymmetric ASCII-in / binary-out framing is
untouched.

### 3.2 Test: preserve hard-timeout + force-kill + return-after-first-action

`test_probe_replay_preserves_hard_timeout` (in `tests/hl/test_probe.py`):
- A candidate that, after the decision-point `roundbegin`, sleeps forever
  (never emits). Assert `probe_one` returns `None` within ~
  `self.timeout + grace` and the process is reaped (no lingering proc —
  check via a marker the test can assert, e.g. the worker is not alive).
- Reuse the ECHO_CANDIDATE pattern but with a sleep-on-roundbegin variant.

### 3.3 Test: replaying a recorded transcript yields an in-support primitive

`test_probe_replay_in_support_emission` (the spec's headline scenario):
- Build a small transcript by hand (or record one from a tiny synthetic trace via
  the §2 recorder) whose decision-point `roundbegin` has `move=[T,...]` open.
- Probe a candidate that, on a roundbegin with an open move, emits `["move",0]`
  (in support). Assert `out_of_support == False` and the primitive is
  `("move",0)`.
- This validates the world-model was faithfully reconstructed (the candidate
  reached a state where it could emit a legal action).

### 3.4 Test: replay does not deadlock the writer

`test_probe_replay_no_deadlock_on_many_prefix_outframes`:
- A candidate that emits MANY out-frames during the prefix (e.g. echoes an
  action after every input frame). The reader thread must drain stdout so the
  probe's `_write_frame` never blocks on a full pipe.
- Assert `probe_one` completes within the hard bound and returns an
  `EmittedAction` (or None), never hangs. Use a generous test timeout
  (pytest marker / `pytest.mark.timeout`-free; just assert monotonic elapsed).

### 3.5 Test: stall → None in time

`test_probe_replay_stall_returns_none_in_time`:
- A candidate that blocks during replay (spawns, then reads stdin but never
  writes stdout, and never reaches the decision point). Assert `probe_one`
  returns `None` within the hard timeout and the test itself doesn't hang
  (bound the probe's `timeout=` small, e.g. 2 s).

---

## §4. Signal validation  (tasks 4.1–4.3)

### 4.1 Fixture: v1→v2 changes a reached decision → KL>0  →  `tests/hl/test_signal_validation.py` (new)

`test_behavioral_edit_registers_positive_kl`:
- Build a recorded ν (small transcript set) and TWO candidate scripts:
  - v1: on the decision-point roundbegin, emits `["move",0]`.
  - v2: same, but emits `["move",1]` (different legal action at that point).
- Probe both over the same ν, run `local_policy_kl_trace(chosen_new,
  chosen_old, legal_sets, epsilon)`.
- Assert the trace has a strictly-positive entry at the changed decision point.
  (Both emissions are in-support; epsilon-smoothed onehots differ → KL>0.)

### 4.2 Fixture: byte-identical v1→v1' → all-zero KL

`test_identical_versions_register_zero_kl`:
- v1 vs a byte-identical copy. Assert the trace is all-zero (discriminative —
  no artificial inflation).

### 4.3 End-to-end: real ν from a short reference roll → policy_kl>0

`test_e2e_recorded_nu_gives_real_kl` — **marked `@pytest.mark.slow`** (needs
the real logic; skip in the fast suite if the logic path isn't available):
- Run one short reference match (sample AI at seat 0 vs a weak baseline) with
  `save_traces`; run `reference_recorder` on the trace → `nu.json`.
- Probe v1 (sample AI) and a v2 with a one-line decision edit over that ν.
- Assert `policy_kl > 0` on ≥1 sample. This is the acceptance signal from the
  spec requirement "policy_kl is a real signal under a behavioral edit".

> If the real logic is unavailable in CI, this test is skipped
> (`pytest.importorskip`-style guard on the logic path). The §4.1/§4.2
> unit-level signal tests are the always-green contract.

---

## §5. Wiring + retire hand-authored seed  (tasks 5.1–5.3)

### 5.1 Document the record→iterate workflow  →  `hl/README.md`, `lostspace/ITERATE.md`

- Add a "Record a ν, then iterate" section to `hl/README.md`:
  1. Run one reference match with `save_traces` (seat 0 = the v1 / sample AI;
     opponents = the benchmark spec; fixed `mapconf2.map`).
  2. `python -m agentbench_frame.hl.reference_recorder --trace <trace.jsonl>
     --spec-id <id> --opponent <name> --out nu.json`.
  3. `python -m agentbench_frame.hl --reference nu.json ...` to iterate.
- Mirror a short version in `lostspace/ITERATE.md`.
- The CLI `--reference` flag already loads any `ReferenceStateSet` JSON — no
  CLI code change (verified `cli.py:390,432`).

### 5.2 Demote `reference_seed.py` to a test fixture  →  `reference_seed.py`, `tests/hl/test_reference_seed.py`

- Update `reference_seed.py` module docstring: it is now a TEST FIXTURE only,
  not a production ν source. Its samples carry `transcript=()` (the default) —
  unit tests of `ReferenceSample` / `ReferenceStateSet` round-tripping still
  use it; the PROBE rejects them (§1.3). Keep `build_seed` / `main` working.
- `tests/hl/test_reference_seed.py`: adjust any assertion that assumed the seed
  is a valid PROBE input. The seed is still a valid `ReferenceStateSet` for
  schema tests; it is explicitly NOT a valid probe input.

### 5.3 Update `lostspace/README.md` / `ITERATE.md`

- Covered by §5.1's doc edits. Ensure both mention the recorder.

---

## §6. Verify + archive  (tasks 6.1–6.3)

### 6.1 Full HL + lostspace suites green

- `uv run pytest tests/hl tests/lostspace` (PYTHONPATH=src).
- Update the existing tests that construct single-frame `ReferenceSample`s and
  drive them through the PROBE — they will now hit the §1.3 fail-fast and must
  be migrated to provide a transcript (or to use the new recorder to build one).
  Known-affected (audit at implementation time):
  - `tests/hl/test_probe.py` (many `ReferenceSample(...)` constructions + probe
    calls — these are the direct probe contract tests; migrate to small
    transcripts).
  - `tests/hl/test_controller.py:49` (`ReferenceSample(...)` in a controller
    probe path).
  - `tests/hl/test_e2e_smoke.py:82` (builds samples; likely needs transcripts
    or a recorded ν).
  - `tests/hl/test_phase8_wiring.py` (wiring smoke; check).
  - `tests/hl/test_cli.py:36` (`build_seed("hl-test").save(p)` — the seed is
    loadable but probing it now fails fast; if the CLI test drives a probe it
    must switch to a recorded ν, else it only checks `load` which is fine).
- Each migrated test keeps its intent; only the sample-construction helper gains
  a transcript.

### 6.2 `/opsx:verify`

- Run against `proposal.md` / `design.md` / `specs/` / `tasks.md`: every
  requirement scenario has a passing test; every `tasks.md` checkbox is checked
  and traces to a plan §.

### 6.3 `/opsx:archive`

- On `liuzhuo/lostspace`: sync `specs/reference-set/spec.md` deltas into
  `openspec/specs/reference-set/spec.md`, then move the change to
  `openspec/changes/archive/2026-08-01-recorded-reference-set/` (with this
  `plan.md`).

---

## Execution order (dependency-respecting)

1. §1 (schema + tests) — unblocks everything.
2. §2 (recorder + tests) — independent of §3; can run after §1.
3. §3 (probe replay + tests) — the core; depends on §1.
4. §6.1 audit of existing probe tests — happens alongside §3 (they break as
   soon as §3 lands the fail-fast).
5. §4 (signal validation) — depends on §2 + §3.
6. §5 (docs + seed demotion) — after §3/§4 stable.
7. §6.2 verify, §6.3 archive.

Each § is one implementer subagent task (TDD: test first, implement to pass),
then a reviewer subagent, per `superpowers:subagent-driven-development`.
