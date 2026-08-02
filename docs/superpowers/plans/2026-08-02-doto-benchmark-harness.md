# DOTO Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible `23_doto` harness in which an LLM edits the original C++ `playerAI.cpp`, the harness compiles and evaluates immutable versions with the official server, reads replays, records strict deterministic KL status, and exports aligned score/IG/budget curves.

**Architecture:** A package-local CLI orchestrates a vendored, hash-verified official server and fixed C++ SDK. Focused modules own framing, compilation, match execution, replay parsing, decision canonicalization, IG, OpenAI-compatible streaming, Run storage, scoring, and the iteration loop. Formal matches use the untouched 300-second map; CI uses protocol fakes and a separately named short test map that cannot enter benchmark Results.

**Tech Stack:** Python 3.11 standard library, original Python DOTO server, GNU g++/make, original C++ SDK/jsoncpp, TOML configuration, JSON/JSONL, ZIP replay, pytest.

## Global Constraints

- Work only on `feature/doto`, forked from `origin/main` at `98d5b0a`; do not cherry-pick Miracle commits.
- The evaluated model owns exactly one complete file: `playerAI.cpp`.
- Preserve official rule functions and `Maps/0.json`; launch-time seeding belongs outside official source.
- Formal benchmark episodes use `realtime_scale=1.0`; short fixtures are test-only and cannot be exported as benchmark scores.
- Expose one entry point: `uv run python -m agentbench_frame.doto <build|match|replay|ig|loop>`.
- Use OpenAI-compatible Chat Completions, SSE streaming by default, no default `max_tokens`, and a configurable 1,000,000-token per-request context limit checked from provider usage.
- Never write API keys to configs, requests, responses, events, or Results.
- Preserve build failures, malformed proposals, process failures, incomplete episodes, and missing IG with explicit stages and reasons.
- Deterministic joint actions have strict KL `0` when identical and infinity when different; invalid or unavailable comparisons are missing. Do not label proxy metrics as KL or IG.
- Use `apply_patch` for source edits; generated binaries, build trees, replays, traces, caches, and Results remain untracked.

---

## File map

Create these production units:

- `src/agentbench_frame/doto/protocol.py`: bounded historical frame I/O.
- `src/agentbench_frame/doto/process.py`: subprocess groups and cleanup.
- `src/agentbench_frame/doto/build.py`: immutable SDK build trees and metadata.
- `src/agentbench_frame/doto/match.py`: official server/AI routing and traces.
- `src/agentbench_frame/doto/replay.py`: safe ZIP/frame/event parsing.
- `src/agentbench_frame/doto/decision_space.py`: typed observations/actions, masks, canonicalization.
- `src/agentbench_frame/doto/ig.py`: same-trace old/new native policy comparison.
- `src/agentbench_frame/doto/loop_config.py`: strict TOML schema.
- `src/agentbench_frame/doto/llm_client.py`: streaming Chat Completions and complete-source proposal parsing.
- `src/agentbench_frame/doto/run_store.py`: atomic Run artifacts and cumulative budgets.
- `src/agentbench_frame/doto/score.py`: episode aggregation, gain, AUC, curves.
- `src/agentbench_frame/doto/loop.py`: baseline/update/evaluate/IG orchestration.
- `src/agentbench_frame/doto/population.py`: corpus population parsing and build report.
- `src/agentbench_frame/doto/cli.py`, `__main__.py`, `__init__.py`: one public CLI/package surface.
- `src/agentbench_frame/doto/official_server/`: vendored source-equivalent server and formal map.
- `src/agentbench_frame/doto/sdk/`: fixed original client SDK excluding mutable `playerAI.cpp`.
- `skills/doto-harness/SKILL.md`, `skills/doto-replay-reader/SKILL.md`: formal model guidance.
- `examples/doto-initial-playerAI.cpp`, `examples/doto-loop.toml`: runnable inputs.

Tests live under `tests/doto/`; fake protocol programs and the short map live under `tests/doto/fixtures/`.

---

### Task 1: Vendor provenance-checked official assets and package entry point

**Files:**
- Create: `src/agentbench_frame/doto/__init__.py`
- Create: `src/agentbench_frame/doto/__main__.py`
- Create: `src/agentbench_frame/doto/cli.py`
- Create: `src/agentbench_frame/doto/assets.py`
- Create: `src/agentbench_frame/doto/PROVENANCE.json`
- Create: `src/agentbench_frame/doto/official_server/{Arguments.py,BaseClass.py,MySTL.py,main.py,Maps/0.json}`
- Create: `src/agentbench_frame/doto/sdk/{main.cpp,playerAI.h,logic.cpp,logic.h,geometry.cpp,geometry.h,const.h,makefile,jsoncpp/**}`
- Test: `tests/doto/test_assets.py`
- Test: `tests/doto/test_cli.py`

**Interfaces:**
- Produces: `official_server_dir() -> Path`, `sdk_dir() -> Path`, `verify_assets() -> dict[str, str]`.
- Produces: `cli.build_parser() -> argparse.ArgumentParser`, `cli.main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write failing asset and CLI tests**

```python
def test_vendored_assets_match_recorded_hashes():
    hashes = verify_assets()
    assert "official_server/main.py" in hashes
    assert "official_server/Maps/0.json" in hashes
    assert "sdk/main.cpp" in hashes
    assert "sdk/playerAI.cpp" not in hashes


def test_only_five_public_subcommands_exist():
    parser = build_parser()
    action = next(a for a in parser._actions if a.dest == "cmd")
    assert set(action.choices) == {"build", "match", "replay", "ig", "loop"}
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `uv run --with pytest python -m pytest tests/doto/test_assets.py tests/doto/test_cli.py -q`

Expected: collection fails because `agentbench_frame.doto` does not exist.

- [ ] **Step 3: Add package skeleton and source-equivalent assets**

Copy the exact corpus files listed in the design, omit the mutable sample
`playerAI.cpp`, and record SHA-256 values relative to the DOTO package in
`PROVENANCE.json`. Implement `verify_assets()` as:

```python
def verify_assets() -> dict[str, str]:
    manifest = json.loads((PACKAGE_DIR / "PROVENANCE.json").read_text())
    actual = {}
    for relative, expected in manifest["sha256"].items():
        digest = hashlib.sha256((PACKAGE_DIR / relative).read_bytes()).hexdigest()
        if digest != expected:
            raise AssetIntegrityError(f"asset hash mismatch: {relative}")
        actual[relative] = digest
    return actual
```

Make `__main__.py` call `cli.main()`. Add all five parsers immediately, with
handlers that raise a clear `DotoNotImplementedError` until their owning task
replaces them; this establishes one CLI without creating duplicate entry points.

- [ ] **Step 4: Run asset/CLI tests and the existing suite**

Run: `uv run --with pytest python -m pytest tests/doto/test_assets.py tests/doto/test_cli.py -q`

Expected: PASS.

Run: `uv run --with pytest python -m pytest -q`

Expected: 49 existing tests and 6 existing subtests still pass, plus new tests.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/doto tests/doto/test_assets.py tests/doto/test_cli.py
git commit -m "feat(doto): vendor official server and fixed SDK"
```

---

### Task 2: Implement framing, process lifecycle, and native candidate builds

**Files:**
- Create: `src/agentbench_frame/doto/protocol.py`
- Create: `src/agentbench_frame/doto/process.py`
- Create: `src/agentbench_frame/doto/build.py`
- Create: `examples/doto-initial-playerAI.cpp`
- Create: `tests/doto/test_protocol.py`
- Create: `tests/doto/test_process.py`
- Create: `tests/doto/test_build.py`
- Create: `tests/doto/fixtures/valid_playerAI.cpp`
- Create: `tests/doto/fixtures/invalid_playerAI.cpp`
- Modify: `src/agentbench_frame/doto/cli.py`

**Interfaces:**
- Produces: `read_exact(stream, size, timeout, label) -> bytes`.
- Produces: `read_server_frame(stream, timeout) -> ServerFrame` and `read_ai_frame(stream, timeout) -> bytes`.
- Produces: `write_ai_observation(stream, payload: bytes) -> None`, `write_server_action(stream, faction: int, payload: bytes) -> None`.
- Produces: `ManagedProcess.start(argv: Sequence[str], cwd: Path, label: str) -> ManagedProcess`, `.terminate() -> None`.
- Produces: `BuildResult(source_hash, executable, command, exit_code, stdout, stderr, seconds)`.
- Produces: `build_candidate(player_ai: Path, output_dir: Path, compiler: str = "g++") -> BuildResult`.

- [ ] **Step 1: Write framing and cleanup tests**

```python
def test_server_frame_keeps_type_target_and_json():
    payload = b'{"frame":0,"faction":1}'
    raw = struct.pack(">Iii", len(payload) + 8, 0, 1) + payload
    frame = read_server_frame(pipe_with(raw), 0.2)
    assert (frame.message_type, frame.target, frame.payload) == (0, 1, payload)


def test_managed_process_terminates_child_group(tmp_path):
    proc = ManagedProcess.start(
        [sys.executable, str(FIXTURES / "spawn_child.py"), str(tmp_path / "pid")],
        cwd=tmp_path,
        label="tree",
    )
    child_pid = wait_for_pid(tmp_path / "pid")
    proc.terminate()
    assert_process_gone(child_pid)
```

- [ ] **Step 2: Run protocol/process tests and verify failure**

Run: `uv run --with pytest python -m pytest tests/doto/test_protocol.py tests/doto/test_process.py -q`

Expected: FAIL because framing and managed processes are undefined.

- [ ] **Step 3: Implement bounded framing and process groups**

Use big-endian signed server lengths exactly as the historical server does,
reject negative/oversized frames, use `selectors` deadlines, start every process
with `start_new_session=True`, and terminate its process group with SIGTERM then
SIGKILL after one second. Capture stderr to a file supplied by the caller rather
than discarding it.

- [ ] **Step 4: Write build success/failure/stale-binary tests**

```python
def test_builds_complete_player_ai_in_isolated_sdk(tmp_path):
    result = build_candidate(FIXTURES / "valid_playerAI.cpp", tmp_path / "build")
    assert result.exit_code == 0
    assert result.executable.is_file()
    assert json.loads((tmp_path / "build/build.json").read_text())["source_hash"]


def test_failed_rebuild_removes_stale_executable(tmp_path):
    output = tmp_path / "build"
    assert build_candidate(FIXTURES / "valid_playerAI.cpp", output).exit_code == 0
    result = build_candidate(FIXTURES / "invalid_playerAI.cpp", output)
    assert result.exit_code != 0
    assert result.executable is None
    assert not (output / "main.out").exists()
```

- [ ] **Step 5: Implement immutable SDK builds and `build` CLI**

Validate that source is UTF-8, nonempty, contains `playerAI::playerAI`, and is
not larger than the configured source limit. Build in a fresh temporary sibling,
write `build.json` atomically, then rename the successful tree into place. On
failure retain source, command, stdout/stderr, and exit code but no executable.

The CLI prints the serialized `BuildResult` and returns nonzero on failure.

- [ ] **Step 6: Run tests and compile the example**

Run: `uv run --with pytest python -m pytest tests/doto/test_protocol.py tests/doto/test_process.py tests/doto/test_build.py tests/doto/test_cli.py -q`

Expected: PASS.

Run: `uv run python -m agentbench_frame.doto build --player-ai examples/doto-initial-playerAI.cpp --output-dir /tmp/agentbench-doto-build`

Expected: exit 0 and JSON naming `/tmp/agentbench-doto-build/main.out`.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/doto examples/doto-initial-playerAI.cpp tests/doto
git commit -m "feat(doto): compile original C++ policy interface"
```

---

### Task 3: Run official-protocol matches and save readable traces

**Files:**
- Create: `src/agentbench_frame/doto/match.py`
- Create: `tests/doto/test_match.py`
- Create: `tests/doto/fixtures/fake_server.py`
- Create: `tests/doto/fixtures/fake_ai.py`
- Create: `tests/doto/fixtures/short_server/Maps/0.json`
- Modify: `src/agentbench_frame/doto/cli.py`

**Interfaces:**
- Produces: `MatchResult(winner, scores, frames, duration, terminated_by, errors, replay_path, trace_path, metadata)`.
- Produces: `run_match(agent0: Path, agent1: Path, *, seed: int, output_dir: Path, tag: str, frame_timeout: float = 1.0, server_timeout: float = 330.0, server_dir: Path | None = None, test_only: bool = False) -> MatchResult`.

- [ ] **Step 1: Write routing, trace, final-score, and failure tests**

```python
def test_routes_two_factions_and_preserves_trace(tmp_path):
    result = run_match(
        FIXTURES / "fake_ai.py", FIXTURES / "fake_ai.py",
        seed=11, output_dir=tmp_path, tag="pair", server_dir=FIXTURES,
        test_only=True,
    )
    assert result.winner == 0
    assert result.scores == (12.0, 7.0)
    rows = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    assert {row["kind"] for row in rows} == {"observation", "action", "final"}
    assert [row["seq"] for row in rows] == list(range(len(rows)))


def test_server_crash_is_not_a_loss(tmp_path):
    with pytest.raises(DotoMatchError, match="process_exit"):
        run_match(
            FIXTURES / "fake_ai.py",
            FIXTURES / "fake_ai.py",
            seed=11,
            output_dir=tmp_path,
            tag="crash",
            server_dir=FIXTURES / "crashing_server",
            test_only=True,
        )
```

- [ ] **Step 2: Run match tests and verify failure**

Run: `uv run --with pytest python -m pytest tests/doto/test_match.py -q`

Expected: FAIL because `run_match` is undefined.

- [ ] **Step 3: Implement official server launcher and router**

Launch the server through a package wrapper that seeds `random` before
`runpy.run_path("main.py", run_name="__main__")`. Set cwd to the server tree and
pass the absolute replay ZIP output path as `sys.argv[1]`. Route targeted
observations to the named AI and AI replies back with faction routing. Save every
raw and decoded message to a flushed JSONL trace.

Reject `server_dir` outside package assets unless `test_only=True`. Include
`test_only` in result metadata and raise if any caller attempts to place a
test-only result under a path containing `runs/23_doto`.

- [ ] **Step 4: Add `match` CLI and seat metadata**

Parse concrete executable paths, verify executability, expose seed/timeouts,
output directory, and tag, and print `MatchResult` JSON. Do not expose alternate
legacy entry points.

- [ ] **Step 5: Run focused and regression tests**

Run: `uv run --with pytest python -m pytest tests/doto/test_match.py tests/doto/test_cli.py -q`

Expected: PASS.

Run: `uv run --with pytest python -m pytest -q`

Expected: all existing and DOTO tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/agentbench_frame/doto tests/doto
git commit -m "feat(doto): run native policies through official protocol"
```

---

### Task 4: Parse official replay ZIPs and author the Replay Reader Skill

**Files:**
- Create: `src/agentbench_frame/doto/replay.py`
- Create: `skills/doto-replay-reader/SKILL.md`
- Create: `tests/doto/test_replay.py`
- Create: `tests/doto/fixtures/real_short_replay.zip`
- Create: `tests/doto/fixtures/real_short_replay.expected.json`
- Modify: `src/agentbench_frame/doto/cli.py`

**Interfaces:**
- Produces: `ReplayFrame(frame: int, humans: list, fireballs: list, meteors: list, balls: list, scores: list[float], bonus: list, events: list)`.
- Produces: `ReplaySummary(final_scores, winner, last_frame, event_counts, by_faction)`.
- Produces: `iter_replay(path: Path) -> Iterator[ReplayFrame]` and `summarize_replay(path: Path) -> ReplaySummary`.

- [ ] **Step 1: Generate one real short replay fixture manually and record provenance**

Use the official server with the separately named short map and two compiled
no-op original-interface AIs. Store only the minimal replay ZIP needed for
parser validation and a JSON file containing its SHA-256, server/map hashes,
final scores, last frame, and event counts. Assert fixture metadata contains
`"test_only": true`.

- [ ] **Step 2: Write safe parsing and metric tests**

```python
def test_real_replay_matches_expected_summary():
    expected = json.loads(EXPECTED.read_text())
    summary = summarize_replay(FIXTURE)
    assert list(summary.final_scores) == expected["final_scores"]
    assert summary.last_frame == expected["last_frame"]
    assert summary.event_counts == expected["event_counts"]


def test_parser_never_executes_stringified_lists(tmp_path):
    replay = make_zip(tmp_path, humans="__import__('os').system('false')")
    with pytest.raises(DotoReplayError, match="invalid humans"):
        list(iter_replay(replay))
```

- [ ] **Step 3: Implement ZIP/frame parser and event aggregation**

Require exactly one replay JSON member, reject absolute/parent paths and
oversized entries, parse JSON frames, use `ast.literal_eval` only for historical
list strings, and validate shapes before producing typed frames. Aggregate event
types 1–12 and attribute damage, deaths, goals, and bonuses using
`human_id % 2`.

- [ ] **Step 4: Write the complete human-authored Replay Reader Skill**

Include the exact rule values and all field/event tables enumerated in section 5
of the design. Include a runnable Python parsing example using `iter_replay`, a
trace alignment example, ten concrete common misreadings, and the real fixture's
self-consistency procedure. Do not state that replay `events` are sent to native
AIs; they are replay-only.

- [ ] **Step 5: Add `replay` CLI and run verification**

Run: `uv run python -m agentbench_frame.doto replay --path tests/doto/fixtures/real_short_replay.zip --jsonl /tmp/doto-events.jsonl`

Expected: printed scores/event counts match `real_short_replay.expected.json`.

Run: `uv run --with pytest python -m pytest tests/doto/test_replay.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agentbench_frame/doto/replay.py src/agentbench_frame/doto/cli.py skills/doto-replay-reader tests/doto
git commit -m "feat(doto): parse replays and document frame semantics"
```

---

### Task 5: Define observation, continuous macro-action, mask, and termination

**Files:**
- Create: `src/agentbench_frame/doto/decision_space.py`
- Create: `skills/doto-harness/SKILL.md`
- Create: `tests/doto/test_decision_space.py`

**Interfaces:**
- Produces: `InitObservation`, `FrameObservation`, `Point`, `DotoAction`, `ActionMask`, `Termination` dataclasses.
- Produces: `parse_observation(raw: dict, faction: int, map_data: dict) -> InitObservation | FrameObservation`.
- Produces: `canonicalize_action(raw: dict) -> DotoAction`.
- Produces: `action_mask(obs: FrameObservation, action: DotoAction) -> ActionMask`.
- Produces: `support_pair(old: DotoAction, new: DotoAction) -> tuple[DotoAction, ...]`.

- [ ] **Step 1: Write parsing and canonical equality tests**

```python
def test_action_requires_exactly_five_humans():
    with pytest.raises(ActionValidationError, match="move must contain 5"):
        canonicalize_action({"flag": 0, "move": [], "shoot": [], "meteor": [], "flash": []})


def test_integer_and_float_coordinates_canonicalize_equal():
    assert canonicalize_action(action_at(1, 2)) == canonicalize_action(action_at(1.0, 2.0))


def test_nonfinite_coordinate_is_rejected():
    with pytest.raises(ActionValidationError, match="finite"):
        canonicalize_action(action_at(float("nan"), 2))
```

- [ ] **Step 2: Write mask tests for every rule family**

Use separate observations proving: dead human, movement distance 0.6, fireball
cooldown, meteor distance 30/use count, flash distance 20/use count, held-ball
flash prohibition, wall destination, map boundary, and always-valid no-op.

- [ ] **Step 3: Implement typed parsing, canonicalization, masks, and support**

Return per-human/per-field mask reasons, not one aggregate boolean. Preserve the
raw observation and map hash. `support_pair` deduplicates equal canonical joint
actions and returns one or two elements; it never enumerates coordinates.

- [ ] **Step 4: Complete the Harness Skill decision contract**

Document the C++ ownership boundary, build/match/replay/IG/loop commands,
observation schema, joint action schema, mask limitations, termination,
continuous theoretical support, finite comparison support, strict KL semantics,
budgets, versions, and Results layout.

- [ ] **Step 5: Run tests and commit**

Run: `uv run --with pytest python -m pytest tests/doto/test_decision_space.py -q`

Expected: PASS.

```bash
git add src/agentbench_frame/doto/decision_space.py skills/doto-harness tests/doto/test_decision_space.py
git commit -m "feat(doto): define continuous joint decision space"
```

---

### Task 6: Compare native policies on the same trace and record strict KL status

**Files:**
- Create: `src/agentbench_frame/doto/ig.py`
- Create: `tests/doto/test_ig.py`
- Create: `tests/doto/fixtures/scripted_ai.py`
- Modify: `src/agentbench_frame/doto/cli.py`

**Interfaces:**
- Produces: `DecisionComparison(frame, faction, observation_hash, old_action, new_action, status, finite_kl, missing_reason)`.
- Produces: `compare_policies_on_trace(trace: Path, old_executable: Path, new_executable: Path, faction: int, iteration: int, old_version: str, new_version: str, timeout: float = 1.0) -> dict`.
- Produces: `build_ig_curve(episode_rows: Sequence[dict], versions: Mapping[int, str]) -> dict`.

- [ ] **Step 1: Write zero/infinite/missing tests**

```python
def test_identical_joint_action_has_zero_strict_kl(tmp_path):
    row = compare(scripted("same"), scripted("same"), one_frame_trace(tmp_path))
    assert row["unchanged_ratio"] == 1.0
    assert row["finite_kl_mean"] == 0.0


def test_one_coordinate_change_is_infinite(tmp_path):
    row = compare(scripted("left"), scripted("right"), one_frame_trace(tmp_path))
    assert row["infinite_ratio"] == 1.0
    assert row["finite_kl_mean"] is None


def test_crashed_policy_is_missing_not_infinite(tmp_path):
    row = compare(scripted("same"), scripted("crash"), one_frame_trace(tmp_path))
    assert row["missing_ratio"] == 1.0
    assert row["decisions"][0]["missing_reason"] == "new_process_exit"
```

- [ ] **Step 2: Implement same-trace native process driving**

Start each executable once, send its original initialization frame, then only
the target faction's observation frames in sequence. Read exactly one reply per
frame with a deadline. Hash canonical observation JSON. Validate each response
with `canonicalize_action` and `action_mask` before equality comparison.

- [ ] **Step 3: Implement episode/iteration aggregation and `ig` CLI**

Write per-decision JSONL and episode JSON atomically. Aggregate counts before
ratios, keep all missing reasons, and emit explicit baseline/missing points for
versions without comparisons.

- [ ] **Step 4: Run tests and commit**

Run: `uv run --with pytest python -m pytest tests/doto/test_ig.py tests/doto/test_decision_space.py -q`

Expected: PASS.

```bash
git add src/agentbench_frame/doto/ig.py src/agentbench_frame/doto/cli.py tests/doto
git commit -m "feat(doto): record strict deterministic KL status"
```

---

### Task 7: Add strict loop configuration and streaming Chat Completions client

**Files:**
- Create: `src/agentbench_frame/doto/loop_config.py`
- Create: `src/agentbench_frame/doto/llm_client.py`
- Create: `examples/doto-loop.toml`
- Create: `tests/doto/test_loop_config.py`
- Create: `tests/doto/test_llm_client.py`

**Interfaces:**
- Produces immutable `LLMConfig`, `EvaluationConfig`, `BudgetConfig`, `LoopConfig` dataclasses and `LoopConfig.load(path: Path) -> LoopConfig`.
- Produces: `PolicyProposal(analysis, player_ai_cpp, usage, latency_seconds, request_body, raw_response)`.
- Produces: `ChatCompletionsClient(config: LLMConfig).propose(messages: list[dict]) -> PolicyProposal`.
- Produces: `LLMRequestError(stage, message, raw_response, usage, latency_seconds)`.

- [ ] **Step 1: Write strict configuration tests**

```python
def test_defaults_to_streaming_one_million_context_and_no_output_cap(tmp_path):
    config = LoopConfig.load(write_minimal_config(tmp_path))
    assert config.llm.stream is True
    assert config.llm.max_context_tokens == 1_000_000
    assert config.llm.max_tokens is None


@pytest.mark.parametrize("field", ["max_iterations", "max_builds", "max_rollouts", "max_episode_reads", "max_frame_reads", "max_total_tokens", "max_wall_seconds"])
def test_budget_limits_must_be_positive(tmp_path, field):
    with pytest.raises(ValueError, match=field):
        LoopConfig.load(write_config(tmp_path, **{field: 0}))
```

- [ ] **Step 2: Implement path-aware TOML parsing**

Resolve `initial_player_ai` and opponent source/executable paths relative to the
config file. Reject unknown `seats`, empty seed lists, duplicate opponent names,
non-boolean stream, invalid reasoning effort, nonpositive limits, and a formal
configuration with `realtime_scale != 1.0`.

- [ ] **Step 3: Write SSE, nonstream, malformed, usage, and secrecy tests**

Use a local `ThreadingHTTPServer`. Verify split `reasoning_content` and `content`
chunks, heartbeat comments, `[DONE]`, usage-only final chunks, malformed JSON,
EOF before `[DONE]`, HTTP errors, optional nonstream JSON, custom User-Agent,
and absence of the API key from exceptions and raw responses.

- [ ] **Step 4: Implement the Chat Completions client**

Send `stream_options={"include_usage": true}` when streaming. Accumulate
reasoning and content separately, preserve chunk count/first-chunk latency,
require `[DONE]`, and do not retry non-idempotently after a partial stream.
Parse exactly one JSON object with nonempty string fields `analysis` and
`player_ai_cpp`; allow one enclosing Markdown JSON fence only.

- [ ] **Step 5: Run tests and commit**

Run: `uv run --with pytest python -m pytest tests/doto/test_loop_config.py tests/doto/test_llm_client.py -q`

Expected: PASS.

```bash
git add src/agentbench_frame/doto/loop_config.py src/agentbench_frame/doto/llm_client.py examples/doto-loop.toml tests/doto
git commit -m "feat(doto): add streamed C++ policy proposal client"
```

---

### Task 8: Store observable Runs, budgets, score curves, and AUC

**Files:**
- Create: `src/agentbench_frame/doto/run_store.py`
- Create: `src/agentbench_frame/doto/score.py`
- Create: `tests/doto/test_run_store.py`
- Create: `tests/doto/test_score.py`

**Interfaces:**
- Produces: `BudgetLedger.charge_build`, `.charge_rollout`, `.charge_read`, `.charge_usage`, `.charge_context`, `.charge_compile_time`, `.charge_battle_time`, `.charge_api_time`, `.snapshot`.
- Produces: `DotoRunStore.create(config, data_dir, run_id=None)`, `.iteration_dir(index)`, `.write_json_atomic`, `.write_text_atomic`, `.write_event`, `.finish(summary)`.
- Produces: `aggregate_episodes(rows, candidate_factions) -> dict`, `trapezoid_auc(points, x_key, y_key) -> dict`, `build_score_curve(iterations) -> dict`.

- [ ] **Step 1: Write cumulative budget and atomic storage tests**

```python
def test_frame_reads_are_cumulative_across_iterations():
    ledger = BudgetLedger(budget(max_frame_reads=3))
    ledger.charge_read(episodes=1, frames=2)
    with pytest.raises(BudgetExceeded, match="frame_reads"):
        ledger.charge_read(episodes=1, frames=2)


def test_run_copies_exact_skills_and_never_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("DOTO_TEST_KEY", "never-save-me")
    store = DotoRunStore.create(config(), tmp_path, run_id="fixed")
    assert (store.run_dir / "skills/doto-harness.SKILL.md").is_file()
    assert "never-save-me" not in read_tree(store.run_dir)
```

- [ ] **Step 2: Implement ledger and Run contract**

Write `run.toml` and `config.json` at start, append/fsync `events.jsonl`, use
temporary sibling plus `os.replace` for atomic JSON/text, copy both Skills, and
finish with AgentBench-required summary fields plus DOTO fields. Store config
API key environment-variable name, never its value.

- [ ] **Step 3: Write scoring, missing-point, and AUC tests**

Test seat-oriented score difference, draw value 0.5, errors excluded from valid
means but included in completion rate, baseline raw alignment, failed iteration
with `evo/gain=null`, and trapezoidal AUC on iteration/rollout/token/read/time
axes using only measured points.

- [ ] **Step 4: Implement scoring and curves**

Preserve raw score components and event metrics. Calculate primary score as
mean candidate-oriented `score_diff`; calculate AUC only when two measured
points with distinct x values exist, otherwise return `value=null` and an exact
reason.

- [ ] **Step 5: Run tests and commit**

Run: `uv run --with pytest python -m pytest tests/doto/test_run_store.py tests/doto/test_score.py -q`

Expected: PASS.

```bash
git add src/agentbench_frame/doto/run_store.py src/agentbench_frame/doto/score.py tests/doto
git commit -m "feat(doto): persist budgets scores and aligned curves"
```

---

### Task 9: Orchestrate the complete LLM iteration loop

**Files:**
- Create: `src/agentbench_frame/doto/loop.py`
- Create: `tests/doto/test_loop.py`
- Create: `tests/doto/test_loop_e2e.py`
- Modify: `src/agentbench_frame/doto/cli.py`
- Modify: `skills/doto-harness/SKILL.md`

**Interfaces:**
- Produces: `build_messages(current_source, skills, sdk_reference, evidence, previous_metrics, budget) -> list[dict]`.
- Produces: `read_evidence(episodes, max_episodes, max_frames) -> list[dict]`.
- Produces: `run_loop(config: LoopConfig, *, client=None, builder=build_candidate, match_runner=run_match, ig_runner=compare_policies_on_trace, data_dir=None, run_id=None) -> Path`.

- [ ] **Step 1: Write exact-context and cumulative-read tests**

```python
def test_messages_contain_only_declared_context():
    messages = build_messages("CPP", skills(), "SDK", [evidence()], metrics(), budget())
    user = messages[1]["content"]
    assert "# Current playerAI.cpp\nCPP" in user
    assert "# DOTO Harness Skill" in user
    assert "# DOTO Replay Reader Skill" in user
    assert "# Fixed SDK Reference\nSDK" in user
    assert "API_KEY" not in user


def test_evidence_budget_does_not_reset_each_iteration(tmp_path):
    result = run_loop(
        config(max_iterations=2, max_frame_reads=1),
        client=FakeClient(),
        builder=FakeBuilder(),
        match_runner=FakeMatchRunner(),
        ig_runner=FakeIGRunner(),
        data_dir=tmp_path,
        run_id="budget-test",
    )
    requests = load_requests(result)
    assert total_observations(requests) == 1
```

- [ ] **Step 2: Write baseline→proposal→build→battle→IG end-to-end test**

Use the real source builder with `valid_playerAI.cpp`, a local streaming mock
server returning a changed complete C++ file, a deterministic short match runner
that writes trace/replay fixtures, and the real store/score/IG aggregators.
Assert iteration 0 and 1 source hashes differ, both builds exist, episode
versions align with curves, and exact request/response artifacts are saved.

- [ ] **Step 3: Implement evidence selection and model messages**

Select accepted-version episodes in deterministic configured order. Each frame
contains observation, candidate action, next events, score delta, and error
context. Charge reads before request creation. Include only the immediately
previous iteration metrics and current accepted source, not rejected source as
the next current policy.

- [ ] **Step 4: Implement baseline and update loop**

For iteration 0: snapshot source, build, evaluate aligned seeds/seats, and record
baseline. For each update: save request, call LLM, charge authoritative usage and
context, save raw response, save candidate source, build, protocol-smoke, run
aligned evaluation, compare old/new policies on each new trace, write iteration,
and advance accepted source. Catch each declared failure stage, save partial
artifacts, append a failed curve point, and continue while budgets allow.

- [ ] **Step 5: Add `loop` CLI and complete Harness Skill usage**

The CLI accepts only `--config` and optional `--data-dir`, prints JSON containing
run directory/status/curve paths, and returns nonzero only when the Run itself
cannot be initialized or finalized. Individual failed iterations do not erase a
completed Run.

- [ ] **Step 6: Run focused and complete tests**

Run: `uv run --with pytest python -m pytest tests/doto/test_loop.py tests/doto/test_loop_e2e.py tests/doto/test_cli.py -q`

Expected: PASS.

Run: `uv run --with pytest python -m pytest tests/doto -q`

Expected: all DOTO tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/doto skills/doto-harness tests/doto
git commit -m "feat(doto): complete replay-driven LLM harness loop"
```

---

### Task 10: Import the historical population with transparent build status

**Files:**
- Create: `src/agentbench_frame/doto/population.py`
- Create: `src/agentbench_frame/doto/population.toml`
- Create: `tests/doto/test_population.py`
- Modify: `src/agentbench_frame/doto/cli.py`
- Modify: `skills/doto-harness/SKILL.md`

**Interfaces:**
- Produces: `PopulationPolicy(name, source, origin, split, expected_sha256, leaderboard, enabled)`.
- Produces: `load_population(path: Path) -> list[PopulationPolicy]`.
- Produces: `build_population(policies, corpus_root: Path, output_dir: Path) -> dict`.

- [ ] **Step 1: Write manifest validation and failed-build retention tests**

```python
def test_population_preserves_declared_splits_and_hashes():
    policies = load_population(PACKAGE_MANIFEST)
    assert {p.split for p in policies} >= {"train", "validation"}
    assert all(len(p.expected_sha256) == 64 for p in policies)


def test_failed_policy_remains_in_build_report(tmp_path):
    report = build_population([broken_policy()], FIXTURES, tmp_path)
    assert report["policies"][0]["status"] == "build_failed"
    assert report["policies"][0]["stderr"]
```

- [ ] **Step 2: Implement safe corpus resolution and isolated builds**

Reject absolute and parent-traversing manifest sources, verify source-tree hash
before build, preserve split/provenance, and write one `population-build.json`.
Mark eligibility only after build and a protocol smoke match; never silently
drop disabled, missing, hash-mismatched, or failed policies.

- [ ] **Step 3: Expose population build under the existing `build` command**

Add mutually exclusive `build --player-ai ...` and
`build --population-manifest ... --corpus-root ...` modes. Do not add a sixth
top-level command.

- [ ] **Step 4: Run tests and commit**

Run: `uv run --with pytest python -m pytest tests/doto/test_population.py tests/doto/test_cli.py -q`

Expected: PASS.

```bash
git add src/agentbench_frame/doto/population.py src/agentbench_frame/doto/population.toml src/agentbench_frame/doto/cli.py skills/doto-harness tests/doto
git commit -m "feat(doto): register historical policy population"
```

---

### Task 11: Validate Results, document formal acceptance, and clean artifacts

**Files:**
- Create: `docs/doto-harness.md`
- Create: `docs/doto-official-acceptance.md`
- Modify: `.gitignore`
- Modify: `README.md`
- Test: `tests/doto/test_docs_examples.py`

**Interfaces:**
- Consumes all prior public commands and artifacts.
- Produces a user-facing quick start and a formal 300-second acceptance procedure; no new runtime interface.

- [ ] **Step 1: Write docs command/config consistency tests**

Extract shell command prefixes and TOML keys from docs, assert every documented
top-level command is one of the five CLI choices, parse `examples/doto-loop.toml`,
and assert generated-artifact paths are ignored.

- [ ] **Step 2: Write complete usage documentation**

Document prerequisites (`python>=3.11`, `g++`, `make`, `zip`), build, match,
seat swap, replay, IG, API environment variable, loop config, model response
contract, context contents, cumulative budgets, score/IG interpretation, Results
tree, failure semantics, population build, and secret handling. Link both Skills
and the design/spec.

- [ ] **Step 3: Document the opt-in official-duration acceptance**

Provide exact commands to build the example and known-good opponent, run seed 11
in factions 0 and 1 with official server/map and `realtime_scale=1.0`, parse both
replays, verify hashes/normal termination, and validate exported Results. State
that the pair takes approximately ten minutes and is intentionally excluded from
ordinary CI.

- [ ] **Step 4: Run complete verification**

Run: `uv run --with pytest python -m pytest tests/doto -q`

Expected: all DOTO tests pass.

Run: `uv run --with pytest python -m pytest -q`

Expected: all repository tests and subtests pass.

Run: `git diff --check`

Expected: no output.

Run: `git status --short`

Expected: only intended source/docs/test files before the final commit; no
`.venv`, `uv.lock`, `main.out`, object files, replay ZIPs, traces, temporary
configs, API responses, or Results.

- [ ] **Step 5: Commit**

```bash
git add .gitignore README.md docs/doto-harness.md docs/doto-official-acceptance.md tests/doto/test_docs_examples.py
git commit -m "docs(doto): publish harness and acceptance guide"
```

---

## Final review gate

- [ ] Confirm `git merge-base feature/doto origin/main` is `98d5b0a` or a later fetched `origin/main` explicitly approved before implementation.
- [ ] Confirm `git log --oneline origin/main..feature/doto` contains no Miracle commits.
- [ ] Confirm official server/map hashes match `PROVENANCE.json`.
- [ ] Confirm every generated Run records server, map, policy, and compiler identity.
- [ ] Confirm failures remain visible in `events.jsonl`, iteration JSON, and curves.
- [ ] Confirm strict KL statuses are only zero, infinite, or missing with reason.
- [ ] Confirm no API key or temporary endpoint appears in tracked files.
- [ ] Confirm `AgentBenchResults` and all generated binaries/replays are untracked.
