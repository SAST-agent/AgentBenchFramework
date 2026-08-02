# Tasks — recorded-reference-set

> Capture source is the leading D3a (`match.py` instrumentation); confirm in
> brainstorming before 2.x. Each task traces to `specs/reference-set/spec.md`.

## 1. ReferenceSample schema + transcript (spec: "carries the full frame transcript")

- [x] 1.1 Add a `transcript` field to `ReferenceSample` (ordered list of judger→AI frames, `id` … decision-point `roundbegin`); update `to_dict`/`from_dict`/`save`/`load`.
- [x] 1.2 Test: a round-tripped sample preserves transcript order and the decision-point `legal_actions`/`inventory`/`status` equal the last frame's fields.
- [x] 1.3 Test (fail-fast): loading a sample whose `transcript` is absent raises a clear "re-record this ν" error (no silent KL=0).

## 2. Recorder: parse the match trace into ν (spec: "one sample per reached Alive decision point")

> `run_match` already writes the full bidirectional trace to `trace_path`
> (`match.py:519-524`); evaluator already has `save_traces`. No `match.py`
> change — the recorder is a pure parser.

- [x] 2.1 New `hl/reference_recorder.py`: load a trace JSONL; keep seat-0 `observation` entries in order; a decision point = frame `roundbegin` with `inturn==0` and `status==Alive`; its `transcript` prefix = all seat-0 observation frames up to & incl. that roundbegin (on-turn + off-turn); take `observation`/`legal_actions`/`inventory`/`status` from the decision-point frame's `content`. Emit a `ReferenceStateSet` pinned to the benchmark `spec_id`. Include all Alive decision points.
- [x] 2.2 CLI: `python -m agentbench_frame.hl.reference_recorder --trace <trace.jsonl> --spec-id <id> --opponent <name> --out nu.json` (opponent recorded for provenance).
- [x] 2.3 Test: given a synthetic trace (Alive roundbegins at rounds 1,2,4; Skip at 3; plus off-turn notifications), the recorder emits exactly 3 samples whose prefixes include the off-turn frames.
- [x] 2.4 Test: the decision-point sample's `legal_actions` equals the `attack/move/detect/interprops` fields of the recorded roundbegin `content`.
- [x] 2.5 Test: recorder is deterministic — same trace in → identical ν out.

## 3. Probe replay (spec: "replays the transcript to obtain a faithful emission")

- [x] 3.1 Refactor `ReferenceProbe.probe_one` to replay: spawn fresh candidate, feed transcript frames in order (ASCII-digit framing unchanged) while the reader thread drains out-frames, then capture the first primitive after the decision-point `roundbegin`.
- [x] 3.2 Preserve the hard wall-clock timeout + force-kill and the return-after-first-action rule under the replay path.
- [x] 3.3 Test: replaying a recorded transcript through the SAME reference candidate that produced it yields an in-support primitive (out_of_support == False) — the map was faithfully built.
- [x] 3.4 Test (replay no deadlock): a candidate that emits many prefix out-frames does not deadlock the writer (reader drains stdout); probe completes within the hard bound.
- [x] 3.5 Test (stall → None in time): a candidate that blocks during replay returns None within the hard timeout, no hang.

## 4. Signal validation (spec: "policy_kl is a real signal under a behavioral edit")

- [x] 4.1 Build a v1→v2 fixture where v2 changes only the action at a reached decision point; assert `local_policy_kl_trace` has a strictly-positive entry there.
- [x] 4.2 Build a byte-identical v1→v1' fixture; assert the trace is all-zero (discriminative).
- [x] 4.3 End-to-end: record a real ν from a short reference roll (sample AI at seat 0), probe v1 and an edited v2, confirm `policy_kl > 0` on ≥1 sample (the acceptance signal).

## 5. Wiring + retire hand-authored seed

- [x] 5.1 Controller/CLI already loads ν via `--reference nu.json` (unchanged). Document the record→iterate workflow: run one reference match with `save_traces` (seat 0 = v1/sample AI) → `reference_recorder` parses the trace into `nu.json` → `--reference nu.json` for the loop.
- [x] 5.2 Demote `reference_seed.py` to a test fixture; update its docstring; keep single-frame samples only for unit tests.
- [x] 5.3 Update `lostspace/README.md` / `ITERATE.md` with the record-then-iterate workflow.

## 6. Verify + archive

- [x] 6.1 Full HL + lostspace test suites green (`uv run pytest tests/hl tests/lostspace`).
- [x] 6.2 `/opsx:verify` against proposal/design/specs/tasks.
- [x] 6.3 `/opsx:archive` on `liuzhuo/lostspace`.
