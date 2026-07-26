# Generals HL Closed-Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible official-engine Generals pilot that evaluates `v0`, gives twelve separate learning replays to one non-interactive Codex act, evaluates `v1` on the same frozen 18-case matrix, and emits complete Framework/CI research artifacts.

**Architecture:** `AgentBench` remains the immutable source of official game logic, historical submissions, the pilot manifest, documentation, and the editable baseline template. `AgentBenchFramework` gains a bounded `agentbench_frame.generals` package that validates those assets, runs SDK-compatible player subprocesses against the official Python logic, orchestrates the frozen benchmark and Codex act, and writes generic tracking/evaluation records. The existing in-process `GeneralsEnv` is not used for pilot scores.

**Tech Stack:** Python 3.11+, standard library (`argparse`, `dataclasses`, `enum`, `hashlib`, `json`, `pathlib`, `selectors`, `shutil`, `signal`, `struct`, `subprocess`, `tempfile`, `tomllib`), pytest, uv, g++ with C++14, official Generals Python logic, Codex CLI `exec --json`.

## Global Constraints

- Benchmark ID is exactly `generals-hl-pilot-v1`.
- Framework implementation starts from `origin/worktree/framework` commit `1a61320` while retaining the current Generals branch and AquaWar commit.
- Official rules come only from `AgentBench/backend_sources/corpus/28_generals/logic/gamecode_logic`.
- `agentbench_frame.env.GeneralsEnv` must not produce pilot results.
- Frozen opponents are `advanced-rank02-robinliu-v18`, `advanced-rank08-nashjunheng-v20`, and `popular-rank16-xiaoaojianghu-v1`.
- Evaluation seeds are exactly `280101`, `280202`, and `280303`; every opponent is played from both seats, producing 18 cases per evaluated version.
- Learning seeds are exactly `281101`, `281202`, and `281303`; only medium and low opponents are used from both seats, producing 12 learning episodes.
- There is exactly one Codex act and no accept/reject, best-version substitution, or automatic rollback.
- Codex uses `exec --json --sandbox workspace-write`; missing provider usage remains unknown.
- No Docker is used. Local subprocess controls improve reliability but are not described as a security sandbox.
- Player limits are 10 seconds for startup, 2 seconds per decision, 600 seconds per match, 64 KiB per protocol packet, and 10 MiB per retained protocol/stderr artifact.
- Official illegal-action and per-decision TLE outcomes are valid losses; harness/process failures make a case invalid.
- Strict policy KL is unavailable. Record action disagreement and occupancy shift without calling either epistemic information gain.
- `events.jsonl` is append-only; incomplete evaluations have missing aggregate score/gain/AUC and are never interpolated.
- Preserve all unrelated user changes. In particular, do not add the existing untracked `AgentBenchFramework/uv.lock`, and do not stage unrelated deletions in `AgentBench`.

## Execution Prerequisites

Before Task 1, invoke `superpowers:using-git-worktrees` for both repositories. Use an isolated Framework worktree based on `zhaoyicheng/Generals` and an isolated AgentBench worktree based on `origin/main`. Record their absolute paths in task-specific variables named `FRAMEWORK_WT` and `ASSET_WT`; never repurpose `HOME` or `CODEX_HOME`.

The two repositories keep separate commits. Commands labelled `[Framework]` run in `$FRAMEWORK_WT`; commands labelled `[Assets]` run in `$ASSET_WT`.

Create an isolated Framework environment without modifying `pyproject.toml`
or creating a project lock:

```bash
cd "$FRAMEWORK_WT"
uv venv .venv
uv pip install --python .venv/bin/python -e . 'pytest>=8,<9'
```

All plan test/CLI commands use this `.venv` directly.

## File Map

### AgentBench

- `backend_sources/corpus/28_generals/benchmark/pilot-v1.toml` — frozen population, seeds, limits, and source paths.
- `backend_sources/corpus/28_generals/benchmark/rules.md` — concise official-rule context sent to Codex.
- `backend_sources/corpus/28_generals/benchmark/replay.md` — exact replay field and action semantics.
- `backend_sources/corpus/28_generals/baselines/hl_v0/main.py` — official SDK-compatible executable entrypoint.
- `backend_sources/corpus/28_generals/baselines/hl_v0/state_view.py` — convert SDK `GameState` to a stable JSON-like strategy view.
- `backend_sources/corpus/28_generals/baselines/hl_v0/strategy.py` — deterministic editable rules.
- `backend_sources/corpus/28_generals/baselines/hl_v0/STRATEGY.md` — human/Codex-readable current strategy.
- `backend_sources/corpus/28_generals/baselines/hl_v0/tests/test_strategy.py` — baseline unit tests.

### AgentBenchFramework

- `src/agentbench_frame/generals/__init__.py` — public Generals pilot interfaces.
- `src/agentbench_frame/generals/models.py` — immutable config, process, match, and evaluation records.
- `src/agentbench_frame/generals/assets.py` — TOML loading, path resolution, hashes, validation, preparation.
- `src/agentbench_frame/generals/protocol.py` — bounded SDK packet and command parsing.
- `src/agentbench_frame/generals/process.py` — managed player subprocess lifecycle and artifacts.
- `src/agentbench_frame/generals/engine.py` — adapter around official `GameState` and rule functions.
- `src/agentbench_frame/generals/match.py` — one official-engine match.
- `src/agentbench_frame/generals/evaluator.py` — frozen case construction and evaluation aggregation.
- `src/agentbench_frame/generals/replay.py` — normalized learning replay/probe-state serialization.
- `src/agentbench_frame/generals/prompt.py` — leak-resistant Codex prompt construction.
- `src/agentbench_frame/generals/measurement.py` — action disagreement and occupancy comparison.
- `src/agentbench_frame/generals/pipeline.py` — `v0 → learning → Codex → v1` orchestration.
- `src/agentbench_frame/generals/cli.py` — `prepare`, `eval`, and `iterate` handlers.
- `src/agentbench_frame/cli.py` — register the `generals` command group.
- `src/agentbench_frame/tracking/providers.py` — retain bounded Codex stderr and CLI metadata.
- `src/agentbench_frame/tracking/run.py` — add generic behavior-change event logging only where the current contracts lack it.
- `tests/generals/fixtures/fake_agent.py` — deterministic protocol player for integration tests.
- `tests/generals/fixtures/fake_codex.py` — deterministic provider executable for closed-loop tests.
- `tests/generals/fixtures/pilot-v1.toml` — relocatable minimal asset manifest.
- `tests/generals/conftest.py` — shared config, limits, fake/real asset, process, match, run, evaluator, and replay fixtures.
- `tests/generals/test_assets.py` — config and asset validation.
- `tests/generals/test_protocol.py` — framing and command parsing.
- `tests/generals/test_process.py` — process limits and cleanup.
- `tests/generals/test_engine.py` — official engine initialization and commands.
- `tests/generals/test_match.py` — seat/round/result behavior.
- `tests/generals/test_evaluator.py` — frozen matrix and score semantics.
- `tests/generals/test_replay_prompt.py` — replay normalization and leakage exclusions.
- `tests/generals/test_measurement.py` — disagreement and occupancy records.
- `tests/generals/test_pipeline.py` — fake-provider end-to-end evidence chain.
- `tests/generals/test_cli.py` — command surface and exit codes.
- `tests/generals/test_live.py` — explicitly gated real-opponent and real-Codex checks.

---

### Task 1: Integrate the Research Measurement Foundation

**Files:**
- Merge: `origin/worktree/framework` into the isolated Framework feature worktree
- Preserve: `docs/superpowers/specs/2026-07-26-generals-hl-closed-loop-design.md`
- Preserve: `src/agentbench_frame/aquawar/`
- Test: existing `tests/`

**Interfaces:**
- Consumes: current branch `zhaoyicheng/Generals` at or after design commit `b95378f`; remote research head `1a61320`
- Produces: one Framework history containing `BenchmarkSpec`, `GameResult`, `CodingAgentController`, `CodexProvider`, `BudgetLedger`, snapshots, quality diagnostics, AquaWar, and this Generals design

- [ ] **Step 1: Verify both required histories are present**

Run:

```bash
git branch --show-current
git merge-base --is-ancestor 1a61320 origin/worktree/framework
git merge-base --is-ancestor b95378f HEAD
git status --short
```

Expected: the worktree is on its isolated Generals feature branch; both ancestry checks exit `0`; no unrelated file is staged.

- [ ] **Step 2: Merge the research foundation**

Run:

```bash
git merge --no-ff origin/worktree/framework -m "merge: integrate research measurement foundation"
```

Expected: the merge retains both `src/agentbench_frame/aquawar/` and `src/agentbench_frame/tracking/providers.py`. If Git reports an overlapping core file, resolve it by retaining the research contracts from `origin/worktree/framework` and the AquaWar-specific imports/registrations from the current branch; do not remove either package.

- [ ] **Step 3: Run the merged regression suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: exit `0` with no failed or errored tests.

- [ ] **Step 4: Verify repository hygiene**

Run:

```bash
git status --short
git log -1 --oneline
```

Expected: no staged or modified implementation files remain after the merge; the existing checkout’s unrelated `uv.lock` never entered this worktree.

### Task 2: Freeze and Validate Generals Pilot Assets

**Files:**
- Create [Assets]: `backend_sources/corpus/28_generals/benchmark/pilot-v1.toml`
- Create [Assets]: `backend_sources/corpus/28_generals/benchmark/rules.md`
- Create [Assets]: `backend_sources/corpus/28_generals/benchmark/replay.md`
- Create [Framework]: `src/agentbench_frame/generals/__init__.py`
- Create [Framework]: `src/agentbench_frame/generals/models.py`
- Create [Framework]: `src/agentbench_frame/generals/assets.py`
- Create [Framework]: `tests/generals/__init__.py`
- Create [Framework]: `tests/generals/conftest.py`
- Create [Framework]: `tests/generals/fixtures/pilot-v1.toml`
- Create [Framework]: `tests/generals/test_assets.py`

**Interfaces:**
- Consumes: an explicit `Path` to the AgentBench repository root
- Produces: `PilotConfig`, `OpponentSpec`, `ProcessLimits`, `AssetLayout`, `load_pilot_config(path)`, `resolve_assets(config, agentbench_root)`, and `validate_assets(layout)`

- [ ] **Step 1: Write failing manifest tests**

Create `tests/generals/test_assets.py` with these behaviors:

```python
from dataclasses import replace
from pathlib import Path

import pytest

from agentbench_frame.generals.assets import (
    AssetValidationError,
    load_pilot_config,
    resolve_assets,
)


FIXTURE = Path(__file__).parent / "fixtures" / "pilot-v1.toml"


def test_manifest_freezes_expected_matrix_and_disjoint_learning_seeds():
    config = load_pilot_config(FIXTURE)

    assert config.benchmark_id == "generals-hl-pilot-v1"
    assert config.evaluation_seeds == (280101, 280202, 280303)
    assert config.learning_seeds == (281101, 281202, 281303)
    assert set(config.evaluation_seeds).isdisjoint(config.learning_seeds)
    assert tuple(item.tier for item in config.opponents) == ("high", "medium", "low")
    assert tuple(item.opponent_id for item in config.learning_opponents) == (
        "advanced-rank08-nashjunheng-v20",
        "popular-rank16-xiaoaojianghu-v1",
    )


def test_resolver_rejects_source_path_outside_agentbench_root(tmp_path):
    config = load_pilot_config(FIXTURE)
    escaped = replace(config, engine_path=Path("../outside"))

    with pytest.raises(AssetValidationError, match="escapes AgentBench root"):
        resolve_assets(escaped, tmp_path)
```

Create a fixture manifest containing the exact IDs, seeds, and limit values from Global Constraints, with fixture-local paths for source files.

Create `tests/generals/conftest.py` with fixtures named `config`, `limits`,
`fixture_assets`, `real_assets`, `fake_agent`, `run_fake_match`,
`fake_match_runner`, `run`, `learning_replays`, `rules_text`, and
`replay_text`. `real_assets` reads `AGENTBENCH_ASSET_ROOT` and skips with
the exact reason `AGENTBENCH_ASSET_ROOT is not set` when absent.
`fake_spec(mode, tmp_path)` and
`assert_no_process_with_recorded_pid(process_json)` are ordinary test
helpers in the same file. Both `fixture_assets` and `real_assets` expose
`.config`, `.layout`, and `.manifest_path`.

- [ ] **Step 2: Run the tests and verify RED**

Run [Framework]:

```bash
.venv/bin/python -m pytest tests/generals/test_assets.py -q
```

Expected: collection fails because `agentbench_frame.generals.assets` does not exist.

- [ ] **Step 3: Implement immutable models and strict TOML loading**

Implement these public signatures:

```python
@dataclass(frozen=True)
class ProcessLimits:
    startup_timeout_s: float
    decision_timeout_s: float
    match_timeout_s: float
    max_packet_bytes: int
    max_artifact_bytes: int


@dataclass(frozen=True)
class OpponentSpec:
    opponent_id: str
    tier: str
    language: str
    source: Path
    argv: tuple[str, ...]
    build_argv: tuple[str, ...] = ()
    learning: bool = False


@dataclass(frozen=True)
class PilotConfig:
    benchmark_id: str
    engine_path: Path
    baseline_path: Path
    evaluation_seeds: tuple[int, ...]
    learning_seeds: tuple[int, ...]
    opponents: tuple[OpponentSpec, ...]
    limits: ProcessLimits

    @property
    def learning_opponents(self) -> tuple[OpponentSpec, ...]:
        return tuple(item for item in self.opponents if item.learning)


@dataclass(frozen=True)
class AssetLayout:
    root: Path
    engine_root: Path
    baseline_root: Path
    opponents: tuple[OpponentSpec, ...]
    engine_hash: str
```

`load_pilot_config(path: Path) -> PilotConfig` must reject duplicate tiers or IDs, overlapping seeds, a benchmark ID other than `generals-hl-pilot-v1`, limits not equal to the frozen values, a learning flag on the high tier, and any count other than three opponents.

`resolve_assets(config, agentbench_root) -> AssetLayout` must resolve every manifest path below `agentbench_root` without following a path escape. `validate_assets(layout) -> list[str]` returns stable diagnostic strings and raises `AssetValidationError` only from `require_valid_assets(layout)`.

- [ ] **Step 4: Add the real frozen manifest and concise human-facing docs**

Write `pilot-v1.toml` with:

```toml
benchmark_id = "generals-hl-pilot-v1"
engine_path = "backend_sources/corpus/28_generals/logic/gamecode_logic"
baseline_path = "backend_sources/corpus/28_generals/baselines/hl_v0"
evaluation_seeds = [280101, 280202, 280303]
learning_seeds = [281101, 281202, 281303]

[limits]
startup_timeout_s = 10
decision_timeout_s = 2
match_timeout_s = 600
max_packet_bytes = 65536
max_artifact_bytes = 10485760
```

Add the three exact `[[opponents]]` entries from the design, including `make`/`./main` for rank 2 and `python main.py` for the Python entries. Mark only medium and low with `learning = true`.

`rules.md` must state board dimensions, turn order, command codes 1–9, resources, generals, terrain, skills, technologies, superweapons, game termination, and the fact that official code is authoritative. `replay.md` must define `Round`, `Player`, `Action`, `Cells`, `Generals`, `Weapons`, `Weapon_cds`, `Tech_level`, `Coins`, `Cell_type`, macro-action/environment-step semantics, and all termination types.

- [ ] **Step 5: Run tests and real-asset validation**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_assets.py -q
AGENTBENCH_ASSET_ROOT="$ASSET_WT" .venv/bin/python -c 'import os; from pathlib import Path; from agentbench_frame.generals.assets import load_pilot_config, resolve_assets, require_valid_assets; root=Path(os.environ["AGENTBENCH_ASSET_ROOT"]); p=root/"backend_sources/corpus/28_generals/benchmark/pilot-v1.toml"; c=load_pilot_config(p); require_valid_assets(resolve_assets(c, root)); print(c.benchmark_id)'
```

Expected: tests pass and the command prints `generals-hl-pilot-v1`.

- [ ] **Step 6: Commit both repositories separately**

Run [Assets]:

```bash
git add backend_sources/corpus/28_generals/benchmark
git commit -m "feat(generals): freeze HL pilot assets"
```

Run [Framework]:

```bash
git add src/agentbench_frame/generals tests/generals
git commit -m "feat(generals): validate pilot assets"
```

### Task 3: Build the Deterministic Official-SDK Baseline

**Files:**
- Create [Assets]: `backend_sources/corpus/28_generals/baselines/hl_v0/main.py`
- Create [Assets]: `backend_sources/corpus/28_generals/baselines/hl_v0/state_view.py`
- Create [Assets]: `backend_sources/corpus/28_generals/baselines/hl_v0/strategy.py`
- Create [Assets]: `backend_sources/corpus/28_generals/baselines/hl_v0/STRATEGY.md`
- Create [Assets]: `backend_sources/corpus/28_generals/baselines/hl_v0/tests/test_strategy.py`

**Interfaces:**
- Consumes: official SDK `run_ai`, `GameState`, and command integers
- Produces: `normalize_state(state, my_seat) -> dict`, `choose_actions(round_number, my_seat, view) -> list[list[int]]`, and SDK-compatible `main.py`

- [ ] **Step 1: Write failing pure-strategy tests**

Tests must construct dictionaries rather than importing the game engine:

```python
from strategy import choose_actions


def state(main=(7, 7), coins=40, neutral=(7, 8), neutral_army=3):
    return {
        "round": 1,
        "my_seat": 0,
        "coins": [coins, 40],
        "tech_level": [[2, 0, 0, 0], [2, 0, 0, 0]],
        "cells": {
            "7,7": {"type": 0, "player": 0, "army": 10, "general_id": 0},
            f"{neutral[0]},{neutral[1]}": {
                "type": 0, "player": -1, "army": neutral_army, "general_id": 2
            },
        },
        "generals": {
            "0": {"id": 0, "player": 0, "type": "main", "position": list(main),
                  "produce_level": 1, "defense_level": 1, "mobility_level": 1}
        },
    }


def test_captures_adjacent_neutral_resource_then_ends_turn():
    assert choose_actions(1, 0, state()) == [[1, 7, 7, 4, 9], [8]]


def test_same_state_always_produces_same_macro_action():
    view = state(neutral=(5, 5))
    assert choose_actions(1, 0, view) == choose_actions(1, 0, view)


def test_every_macro_action_ends_turn():
    assert choose_actions(4, 0, state(coins=9))[-1] == [8]
```

Add focused cases for an unaffordable upgrade, blocked mountain, no legal move, enemy adjacency, and seat 1 coordinate handling.

- [ ] **Step 2: Run tests and verify RED**

Run [Assets]:

```bash
python -m pytest backend_sources/corpus/28_generals/baselines/hl_v0/tests/test_strategy.py -q
```

Expected: collection fails because `strategy.py` does not exist.

- [ ] **Step 3: Implement the minimal deterministic rules**

`state_view.py` must convert official SDK objects into only JSON-compatible primitives and sort cells/generals by stable coordinates/IDs. `strategy.py` must:

```python
END_TURN = [8]
DIRECTION = {(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}


def choose_actions(round_number: int, my_seat: int, view: dict) -> list[list[int]]:
    actions: list[list[int]] = []
    adjacent = capture_adjacent_neutral(view, my_seat)
    if adjacent is not None:
        actions.append(adjacent)
    else:
        routed = move_main_toward_nearest_resource(view, my_seat)
        if routed is not None:
            actions.extend(routed)
    upgrade = affordable_production_upgrade(view, my_seat)
    if upgrade is not None:
        actions.append(upgrade)
    advance = advance_spare_army(view, my_seat)
    if advance is not None:
        actions.append(advance)
    return actions + [END_TURN]
```

Implement the four helpers used above with these exact interfaces:
`capture_adjacent_neutral(view: dict, my_seat: int) -> list[int] | None`,
`move_main_toward_nearest_resource(view: dict, my_seat: int) ->
list[list[int]] | None`, `affordable_production_upgrade(view: dict,
my_seat: int) -> list[int] | None`, and `advance_spare_army(view: dict,
my_seat: int) -> list[int] | None`.

Use breadth-first routing over non-mountain cells, row-major tie-breaking,
official upgrade costs, ownership checks, and an army move amount of
`max(0, source_army - 1)`. A helper returns `None` when it has no legal
action. Do not use `random`, time, search, network, or files.

`main.py` must import the preserved official player controller from the engine root supplied through `PYTHONPATH`, normalize its `GameState`, call `choose_actions`, and pass that callback to `run_ai`.

- [ ] **Step 4: Document the exact initial policy**

`STRATEGY.md` must list the six rule priorities, deterministic tie-breaking, protected files, the `choose_actions` interface, the test command, and the rule that future Codex acts may change strategy/tests/docs but not SDK protocol code.

- [ ] **Step 5: Run baseline tests**

Run [Assets]:

```bash
"$FRAMEWORK_WT/.venv/bin/python" -m pytest "$ASSET_WT/backend_sources/corpus/28_generals/baselines/hl_v0/tests/test_strategy.py" -q
"$FRAMEWORK_WT/.venv/bin/python" -m compileall -q "$ASSET_WT/backend_sources/corpus/28_generals/baselines/hl_v0"
```

Expected: both commands exit `0`.

- [ ] **Step 6: Commit the baseline**

Run [Assets]:

```bash
git add backend_sources/corpus/28_generals/baselines/hl_v0
git commit -m "feat(generals): add deterministic HL baseline"
```

### Task 4: Implement the Bounded SDK Process Protocol

**Files:**
- Create [Framework]: `src/agentbench_frame/generals/protocol.py`
- Create [Framework]: `src/agentbench_frame/generals/process.py`
- Modify [Framework]: `src/agentbench_frame/generals/models.py`
- Modify [Framework]: `src/agentbench_frame/generals/assets.py`
- Create [Framework]: `tests/generals/fixtures/fake_agent.py`
- Create [Framework]: `tests/generals/test_protocol.py`
- Create [Framework]: `tests/generals/test_process.py`

**Interfaces:**
- Consumes: `ProcessLimits` and `AgentProcessSpec`
- Produces: `ProtocolError`, `PacketTooLarge`, `DecisionTimeout`, `parse_command_packet`, `encode_peer_commands`, and `ManagedAgentProcess`

- [ ] **Step 1: Write failing protocol tests**

Create these core assertions:

```python
import io
import struct

import pytest

from agentbench_frame.generals.protocol import (
    MissingEndTurn,
    PacketTooLarge,
    encode_peer_commands,
    parse_command_packet,
    read_packet,
)


class FragmentedStream:
    def __init__(self, data: bytes, chunk: int):
        self._data = io.BytesIO(data)
        self._chunk = chunk

    def read(self, size: int = -1) -> bytes:
        return self._data.read(min(size, self._chunk))


def test_reads_fragmented_length_prefixed_packet():
    payload = b"1 7 7 4 9\n8\n"
    stream = FragmentedStream(struct.pack(">I", len(payload)) + payload, chunk=1)
    assert read_packet(stream, max_bytes=65536) == payload


def test_rejects_packet_larger_than_limit_before_reading_body():
    stream = io.BytesIO(struct.pack(">I", 65537))
    with pytest.raises(PacketTooLarge):
        read_packet(stream, max_bytes=65536)


def test_commands_require_exactly_one_terminal_end_turn():
    with pytest.raises(MissingEndTurn):
        parse_command_packet(b"1 7 7 4 9\n")
    assert parse_command_packet(b"1 7 7 4 9\n8\n") == ((1, 7, 7, 4, 9), (8,))


def test_peer_commands_use_plain_newline_protocol():
    assert encode_peer_commands(((1, 7, 7, 4, 9), (8,))) == b"1 7 7 4 9\n8\n"
```

Also cover empty reads, invalid UTF-8, blank lines, non-integers, trailing commands after `8`, and commands with invalid arity.

- [ ] **Step 2: Write failing process tests**

Use `fake_agent.py` modes selected by argv:

```python
def test_managed_agent_round_trip_and_artifacts(tmp_path, limits):
    spec = AgentProcessSpec(
        agent_id="fake",
        argv=(sys.executable, str(FAKE_AGENT), "valid"),
        cwd=tmp_path,
        env={},
    )
    with ManagedAgentProcess(spec, limits, tmp_path / "artifacts") as agent:
        agent.send_initial({"Player": 0})
        assert agent.request_turn() == ((8,),)
    assert (tmp_path / "artifacts" / "agent.protocol.bin").exists()


def test_decision_timeout_is_typed_and_kills_process_group(tmp_path, limits):
    spec = fake_spec("hang", tmp_path)
    with pytest.raises(DecisionTimeout):
        with ManagedAgentProcess(spec, limits, tmp_path / "artifacts") as agent:
            agent.send_initial({"Player": 0})
            agent.request_turn()
    assert_no_process_with_recorded_pid(tmp_path / "artifacts" / "process.json")
```

Also test stderr truncation at `max_artifact_bytes`, premature exit, startup failure, environment allowlisting, and cleanup after an exception.

- [ ] **Step 3: Run tests and verify RED**

Run [Framework]:

```bash
.venv/bin/python -m pytest tests/generals/test_protocol.py tests/generals/test_process.py -q
```

Expected: collection fails because the protocol/process modules do not exist.

- [ ] **Step 4: Implement protocol parsing**

Use exact command arities:

```python
COMMAND_ARITY = {1: 5, 2: 4, 3: 3, 4: None, 5: 2, 6: None, 7: 3, 8: 1, 9: 1}
```

Command `4` accepts 3 integers for non-target skills or 5 for target skills. Command `6` accepts 4 integers except transmission, which accepts 6. Parse with strict UTF-8 and reject any bytes/lines after terminal `8`.

The public functions are `read_packet(stream: BinaryIO, max_bytes: int) ->
bytes`, `parse_command_packet(payload: bytes) -> tuple[tuple[int, ...],
...]`, and `encode_peer_commands(commands:
Sequence[Sequence[int]]) -> bytes`.

- [ ] **Step 5: Implement managed process lifecycle**

Add `AgentProcessSpec` to `models.py`:

```python
@dataclass(frozen=True)
class AgentProcessSpec:
    agent_id: str
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
```

Launch with `shell=False`, `start_new_session=True`, binary pipes, an explicit environment containing only `PATH`, `PYTHONPATH`, `LANG`, and manifest-provided overrides. Tee protocol bytes to `agent.protocol.bin`, drain stderr on a daemon thread into a bounded file, record PID/argv/return code without secrets, and terminate the process group with `SIGTERM` then `SIGKILL`.

`ManagedAgentProcess` implements context-manager methods plus
`send_initial(observation: Mapping[str, object]) -> None`,
`send_peer_commands(commands: Sequence[Sequence[int]]) -> None`, and
`request_turn() -> tuple[tuple[int, ...], ...]`.

Add `prepare_opponents(layout: AssetLayout, preparation_root: Path,
python_executable: Path) -> tuple[AgentProcessSpec, ...]` to `assets.py`.
It copies every preserved submission into its own preparation directory,
runs C++ `make` there for rank 2, verifies the resulting executable, and
returns source-copy-local process specs. Add
`build_baseline_process(workspace: Path, engine_root: Path,
python_executable: Path) -> AgentProcessSpec` to `process.py`; its
`PYTHONPATH` contains the official engine root and the baseline workspace.

- [ ] **Step 6: Run tests and commit**

Run [Framework]:

```bash
.venv/bin/python -m pytest tests/generals/test_protocol.py tests/generals/test_process.py -q
git add src/agentbench_frame/generals/models.py src/agentbench_frame/generals/assets.py src/agentbench_frame/generals/protocol.py src/agentbench_frame/generals/process.py tests/generals/conftest.py tests/generals/fixtures/fake_agent.py tests/generals/test_protocol.py tests/generals/test_process.py
git commit -m "feat(generals): add bounded player protocol"
```

Expected: tests pass and the commit contains no other paths.

### Task 5: Run One Match Against the Official Engine

**Files:**
- Create [Framework]: `src/agentbench_frame/generals/engine.py`
- Create [Framework]: `src/agentbench_frame/generals/match.py`
- Modify [Framework]: `src/agentbench_frame/generals/models.py`
- Create [Framework]: `tests/generals/test_engine.py`
- Create [Framework]: `tests/generals/test_match.py`

**Interfaces:**
- Consumes: `AssetLayout`, `AgentProcessSpec`, official game modules, and `ManagedAgentProcess`
- Produces: `OfficialGeneralsEngine`, `GeneralsMatchRunner`, `MatchResult`, `TurnRecord`, and canonical per-match artifacts

- [ ] **Step 1: Write failing official-engine tests**

Tests load the real official logic from a root provided by `AGENTBENCH_ASSET_ROOT` or skip with an explicit reason:

```python
def test_same_seed_produces_same_initial_state(real_assets):
    one = OfficialGeneralsEngine(real_assets.engine_root, seed=280101)
    two = OfficialGeneralsEngine(real_assets.engine_root, seed=280101)
    assert one.initial_observation(0) == two.initial_observation(0)
    assert one.state_id() == two.state_id()


def test_player_one_end_turn_updates_round(real_assets):
    engine = OfficialGeneralsEngine(real_assets.engine_root, seed=280101)
    assert engine.round_number == 1
    engine.apply_turn(0, ((8,),))
    engine.apply_turn(1, ((8,),))
    assert engine.round_number == 2


def test_illegal_official_command_is_valid_ia_loss(real_assets):
    engine = OfficialGeneralsEngine(real_assets.engine_root, seed=280101)
    outcome = engine.apply_turn(0, ((1, -1, -1, 1, 10), (8,)))
    assert outcome.done is True
    assert outcome.winner == 1
    assert outcome.termination_type == "illegal_action"
```

- [ ] **Step 2: Write failing match tests**

```python
def test_match_sends_player_specific_seats_and_records_every_turn(real_assets, tmp_path):
    result = GeneralsMatchRunner(real_assets).run(
        case=MatchCase(case_id="case", seed=280101, evaluated_seat=1),
        players=(fake_agent("end_turn"), fake_agent("end_turn")),
        artifact_dir=tmp_path,
    )
    assert result.seed == 280101
    assert result.turns[0].player == 0
    assert result.turns[1].player == 1
    assert json.loads((tmp_path / "metadata.json").read_text())["evaluated_seat"] == 1
    assert (tmp_path / "replay.jsonl").exists()


def test_agent_decision_timeout_is_valid_tle_loss(real_assets, tmp_path):
    result = run_fake_match(real_assets, tmp_path, player0_mode="hang")
    assert result.valid is True
    assert result.winner == 1
    assert result.termination_type == "time_limit"
```

Also cover match-controller timeout as invalid, process crash as invalid, surrender as a valid loss, player 0/player 1 forwarding order, and `MOVE_ARMY` clamping.

- [ ] **Step 3: Run tests and verify RED**

Run [Framework]:

```bash
AGENTBENCH_ASSET_ROOT="$ASSET_WT" .venv/bin/python -m pytest tests/generals/test_engine.py tests/generals/test_match.py -q
```

Expected: collection fails because engine/match modules do not exist.

- [ ] **Step 4: Implement the official engine boundary**

`OfficialGeneralsEngine` exposes
`__init__(engine_root: Path, seed: int, replay_path: Path | None = None)`,
the `round_number: int` property,
`initial_observation(player: int) -> dict`,
`apply_turn(player: int, commands: Sequence[Sequence[int]]) ->
TurnOutcome`, `state_id() -> str`, and `normalized_state() -> dict`.

Load the official modules from the explicit engine root, call `random.seed`,
`GameState`, `update_map`, `init_generals`, and set coins to `[40, 40]`.
Use official `execute_single_command`, `update_round`, and the exact
`is_game_over` logic extracted without changing official source. Canonical
state IDs are SHA-256 of sorted compact JSON from the normalized board,
generals, coins, weapons, cooldowns, technologies, round, and next actor.

- [ ] **Step 5: Implement canonical match records and artifact writing**

Add to `models.py`:

```python
@dataclass(frozen=True)
class MatchCase:
    case_id: str
    seed: int
    evaluated_seat: int
    opponent_id: str = "fake"


@dataclass(frozen=True)
class TurnRecord:
    step: int
    round_number: int
    player: int
    state_id_before: str
    state_before: Mapping[str, Any]
    commands: tuple[tuple[int, ...], ...]
    state_id_after: str
    state_after: Mapping[str, Any]


@dataclass(frozen=True)
class MatchResult:
    case_id: str
    valid: bool
    winner: int | None
    termination_type: str
    seed: int
    evaluated_seat: int
    turns: tuple[TurnRecord, ...]
    elapsed_time_s: float
    engine_hash: str
    error: str | None = None
```

`GeneralsMatchRunner.run(case: MatchCase, players:
tuple[AgentProcessSpec, AgentProcessSpec], artifact_dir: Path) ->
MatchResult` must write finalized `metadata.json` and one
JSON object per official/normalized event to `replay.jsonl`. It must always
close both players and retain artifacts in `finally`. Player artifacts live
under `players/0/agent.protocol.bin`, `players/0/agent.stderr.log`,
`players/1/agent.protocol.bin`, and `players/1/agent.stderr.log`.
`metadata.json` includes the benchmark ID, engine hash, agent versions,
opponent ID/tier, seed, evaluated seat, winner, validity, termination type,
runtime, and process statuses.

- [ ] **Step 6: Run tests and commit**

Run [Framework]:

```bash
AGENTBENCH_ASSET_ROOT="$ASSET_WT" .venv/bin/python -m pytest tests/generals/test_engine.py tests/generals/test_match.py -q
git add src/agentbench_frame/generals/models.py src/agentbench_frame/generals/engine.py src/agentbench_frame/generals/match.py tests/generals/conftest.py tests/generals/test_engine.py tests/generals/test_match.py
git commit -m "feat(generals): run official-engine matches"
```

Expected: tests pass and the commit contains only Generals integration/test paths.

### Task 6: Build the Frozen Evaluation Matrix and Match Logs

**Files:**
- Create [Framework]: `src/agentbench_frame/generals/evaluator.py`
- Create [Framework]: `tests/generals/test_evaluator.py`
- Modify [Framework]: `src/agentbench_frame/tracking/run.py`
- Test [Framework]: `tests/test_tracking_contracts.py`

**Interfaces:**
- Consumes: `PilotConfig`, `AssetLayout`, `GeneralsMatchRunner`, `BenchmarkSpec`, `GameResult`, `Run`
- Produces: `build_evaluation_spec(config)`, `build_learning_cases(config)`, `GeneralsEvaluator.evaluate`, per-tier score, seat gap, and phase-separated budgets

- [ ] **Step 1: Write failing matrix tests**

```python
def test_evaluation_matrix_is_exactly_three_by_three_by_two(config):
    spec = build_evaluation_spec(config)
    assert spec.version == "generals-hl-pilot-v1"
    assert len(spec.cases) == 18
    assert len({case.case_id for case in spec.cases}) == 18
    assert {case.seed for case in spec.cases} == {280101, 280202, 280303}
    assert {case.first_player for case in spec.cases} == {0, 1}


def test_learning_cases_exclude_high_and_evaluation_seeds(config):
    cases = build_learning_cases(config)
    assert len(cases) == 12
    assert {case.opponent for case in cases} == {
        "advanced-rank08-nashjunheng-v20",
        "popular-rank16-xiaoaojianghu-v1",
    }
    assert {case.seed for case in cases}.isdisjoint({280101, 280202, 280303})
```

- [ ] **Step 2: Write failing aggregation and logging tests**

```python
def test_complete_evaluation_scores_draw_as_half(config, fake_match_runner, run):
    fake_match_runner.outcomes = ["win"] * 8 + ["draw"] * 2 + ["loss"] * 8
    result = GeneralsEvaluator(config, fake_match_runner).evaluate(
        workspace=Path("/agent"), version="v0", phase="evaluation", run=run
    )
    assert result.status == "complete"
    assert result.score == 0.5
    assert result.per_tier.keys() == {"high", "medium", "low"}


def test_one_invalid_case_keeps_raw_results_but_removes_score(config, fake_match_runner, run):
    fake_match_runner.invalid_case = build_evaluation_spec(config).cases[0].case_id
    result = GeneralsEvaluator(config, fake_match_runner).evaluate(
        workspace=Path("/agent"), version="v0", phase="evaluation", run=run
    )
    assert len(result.results) == 18
    assert result.status == "incomplete"
    assert result.score is None
```

Assert that evaluation budgets add 18 episodes and the exact summed
environment steps, while learning budgets remain unchanged.

- [ ] **Step 3: Run tests and verify RED**

Run [Framework]:

```bash
.venv/bin/python -m pytest tests/generals/test_evaluator.py tests/test_tracking_contracts.py -q
```

Expected: failures for missing evaluator and phase-specific Generals logging.

- [ ] **Step 4: Implement evaluation construction and aggregation**

Define the evaluation record:

```python
@dataclass(frozen=True)
class GeneralsEvaluation:
    version: str
    status: str
    score: float | None
    wins: int
    losses: int
    draws: int
    per_tier: Mapping[str, float | None]
    seat_gap: float | None
    results: tuple[GameResult, ...]
    matches: tuple[MatchResult, ...]
```

Implement `build_evaluation_spec(config: PilotConfig) -> BenchmarkSpec`,
`build_learning_cases(config: PilotConfig) -> tuple[BenchmarkCase, ...]`,
`GeneralsEvaluator.__init__(config: PilotConfig, execute_match:
Callable[[BenchmarkCase, Path, str, Path], MatchResult])`, and
`GeneralsEvaluator.evaluate(workspace: Path, version: str, phase: str,
run: Run, cases: Sequence[BenchmarkCase] | None = None) ->
GeneralsEvaluation`. The injected callable receives case, evaluated
workspace, version, and per-match artifact directory; production wiring
builds it from `GeneralsMatchRunner` and the prepared opponent specs.

Execute cases sequentially in stable opponent/seed/seat order. Convert valid
winner values relative to `first_player`. Keep invalid matches as
`GameResult(valid=False, error=match.error)`. Derive aggregate score only through
the existing `evaluate_benchmark` contract.

- [ ] **Step 5: Add only generic missing Run event helpers**

If the merged `Run` lacks a generic method, add:

```python
def log_game_result(self, phase: str, version: str, payload: Mapping[str, Any]) -> None:
    self.write("game_result", phase=phase, version=version, **dict(payload))
```

Use existing `log_budget`, `record_act_evaluation`, and event writing instead
of adding Generals fields to core tracking classes.

- [ ] **Step 6: Run tests and commit**

Run [Framework]:

```bash
.venv/bin/python -m pytest tests/generals/test_evaluator.py tests/test_tracking_contracts.py -q
git add src/agentbench_frame/generals/evaluator.py src/agentbench_frame/tracking/run.py tests/generals/test_evaluator.py tests/test_tracking_contracts.py
git commit -m "feat(generals): add frozen evaluation matrix"
```

Expected: tests pass; `git diff HEAD^ --name-only` contains only the listed evaluator/tracking tests and sources.

### Task 7: Normalize Learning Replays, Build the Prompt, and Measure Behavior Change

**Files:**
- Create [Framework]: `src/agentbench_frame/generals/replay.py`
- Create [Framework]: `src/agentbench_frame/generals/prompt.py`
- Create [Framework]: `src/agentbench_frame/generals/measurement.py`
- Create [Framework]: `tests/generals/test_replay_prompt.py`
- Create [Framework]: `tests/generals/test_measurement.py`

**Interfaces:**
- Consumes: twelve learning `MatchResult` artifacts, rules/replay docs, v0/v1 baseline workspaces
- Produces: `LearningReplay`, `build_codex_prompt`, `ProbeState`, `measure_action_disagreement`, and occupancy samples

- [ ] **Step 1: Write failing replay and leakage tests**

```python
def test_learning_replay_keeps_decisions_and_redacts_opponent_identity(match_result):
    replay = build_learning_replay(match_result, evaluated_agent_id="baseline")
    assert replay.seed == 281101
    assert replay.decisions
    encoded = replay.to_json()
    assert "rank08" not in encoded
    assert "top_algorithms" not in encoded


def test_prompt_contains_learning_evidence_but_no_evaluation_material(
    learning_replays, rules_text, replay_text
):
    prompt = build_codex_prompt(
        benchmark_id="generals-hl-pilot-v1",
        strategy_doc="Current rules",
        rules_text=rules_text,
        replay_guide=replay_text,
        learning_replays=learning_replays,
    )
    assert "281101" in prompt
    assert "280101" not in prompt
    assert "opponent source" not in prompt.lower()
    assert "codex exec" not in prompt
```

The prompt test must also reject absolute AgentBench paths, evaluation
scores, high-tier IDs, and any source-code extension from an opponent path.

- [ ] **Step 2: Write failing behavior-change tests**

```python
def test_action_disagreement_compares_same_probe_states():
    probes = (
        ProbeState("s1", {"round": 1}, ((8,),)),
        ProbeState("s2", {"round": 2}, ((1, 0, 0, 4, 2), (8,))),
    )
    measured = measure_action_disagreement(
        probes,
        new_actions={"s1": ((8,),), "s2": ((8,),)},
    )
    assert measured.trace == (0.0, 1.0)
    assert measured.mean == 0.5


def test_measurement_reports_policy_kl_unavailable():
    measured = BehaviorChange(trace=(0.0,), mean=0.0)
    assert measured.policy_kl is None
    assert measured.policy_kl_status == "complete_macro_action_distribution_unavailable"
```

Add an occupancy test using normalized state-ID histograms and smoothing
`1e-12`, asserting it remains a separate field from action disagreement.

- [ ] **Step 3: Run tests and verify RED**

Run [Framework]:

```bash
.venv/bin/python -m pytest tests/generals/test_replay_prompt.py tests/generals/test_measurement.py -q
```

Expected: collection fails because replay/prompt/measurement modules do not exist.

- [ ] **Step 4: Implement normalized replay and prompt objects**

Define:

```python
@dataclass(frozen=True)
class DecisionRecord:
    state_id: str
    round_number: int
    seat: int
    state: Mapping[str, Any]
    action: tuple[tuple[int, ...], ...]
    outcome: str


@dataclass(frozen=True)
class LearningReplay:
    replay_id: str
    seed: int
    evaluated_seat: int
    opponent_tier: str
    termination_type: str
    decisions: tuple[DecisionRecord, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)
```

Implement `build_codex_prompt(benchmark_id: str, strategy_doc: str,
rules_text: str, replay_guide: str, learning_replays:
Sequence[LearningReplay]) -> str`.

The prompt must tell Codex to inspect the editable baseline, change only
`strategy.py`, `STRATEGY.md`, and tests, preserve the pure
`choose_actions` interface, run tests, avoid random/network/opponent-specific
hard-coding, and finish with a concise change summary. Embed all twelve
learning replays in stable seed/seat order.

- [ ] **Step 5: Implement probe and occupancy measurement**

Define:

```python
@dataclass(frozen=True)
class ProbeState:
    state_id: str
    state: Mapping[str, Any]
    old_action: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BehaviorChange:
    trace: tuple[float, ...]
    mean: float
    policy_kl: None = None
    policy_kl_status: str = "complete_macro_action_distribution_unavailable"
```

Implement `measure_action_disagreement(probes: Sequence[ProbeState],
new_actions: Mapping[str, tuple[tuple[int, ...], ...]]) ->
BehaviorChange` by generating `1.0` for unequal canonical macro-actions and
`0.0` for equal actions in probe order, then setting `mean` to
`sum(trace) / len(trace)`; reject an empty probe set or missing/extra state
IDs.

Canonicalize integer commands and use exact macro-action equality. Invoke
the v1 strategy on each stored normalized state in a fresh subprocess and
require one response for every probe state. Reuse the merged Framework
occupancy function for normalized state IDs; do not add disagreement and
occupancy values together.

- [ ] **Step 6: Run tests and commit**

Run [Framework]:

```bash
.venv/bin/python -m pytest tests/generals/test_replay_prompt.py tests/generals/test_measurement.py -q
git add src/agentbench_frame/generals/replay.py src/agentbench_frame/generals/prompt.py src/agentbench_frame/generals/measurement.py tests/generals/conftest.py tests/generals/test_replay_prompt.py tests/generals/test_measurement.py
git commit -m "feat(generals): add replay-driven behavior measurement"
```

Expected: tests pass with no warnings or unknown test markers.

### Task 8: Orchestrate the One-Act Codex Pipeline and Version Artifacts

**Files:**
- Create [Framework]: `src/agentbench_frame/generals/pipeline.py`
- Create [Framework]: `tests/generals/fixtures/fake_codex.py`
- Create [Framework]: `tests/generals/test_pipeline.py`
- Modify [Framework]: `src/agentbench_frame/tracking/providers.py`
- Modify [Framework]: `src/agentbench_frame/tracking/controller.py`
- Modify [Framework]: `src/agentbench_frame/tracking/snapshot.py`
- Test [Framework]: `tests/test_provider_integration_contracts.py`
- Test [Framework]: `tests/test_tracking_contracts.py`

**Interfaces:**
- Consumes: `PilotConfig`, `AssetLayout`, `GeneralsEvaluator`, `CodexProvider`, `Run`
- Produces: `GeneralsHLPipeline.run() -> PipelineResult`, exact run artifacts, raw/evo/gain, and budget/AUC records

- [ ] **Step 1: Write a failing fake-provider closed-loop test**

`fake_codex.py` must emit valid Codex JSONL and replace one deterministic
line in `strategy.py`. The test:

```python
def test_pipeline_writes_complete_v0_v1_evidence_chain(
    tmp_path, fixture_assets, fake_evaluator, fake_codex_executable
):
    pipeline = GeneralsHLPipeline(
        config=fixture_assets.config,
        assets=fixture_assets.layout,
        data_dir=tmp_path / "data",
        provider=CodexProvider(executable=str(fake_codex_executable)),
        evaluator=fake_evaluator,
    )
    result = pipeline.run()

    assert result.raw_score == 0.25
    assert result.evo_score == 0.50
    assert result.gain == 0.25
    assert result.act_count == 1
    assert (result.run_dir / "provider" / "codex.raw.jsonl").exists()
    assert (result.run_dir / "versions" / "v0" / "manifest.json").exists()
    assert (result.run_dir / "versions" / "v1" / "manifest.json").exists()
    assert (result.run_dir / "versions" / "v0-to-v1.patch").read_text()
    assert (result.run_dir / "quality.json").exists()
```

Assert event order:

```text
benchmark_spec → version(v0) → evaluation(raw) → learning games
→ coding_agent_act → version(v1) → behavior_change → evaluation(evo)
```

Assert exactly one act, 18 raw cases, 12 learning cases, and 18 evolved cases.

- [ ] **Step 2: Write failing failure-path tests**

Add tests proving:

- provider failure still writes raw act/version artifacts but skips v1 evaluation;
- no file change still creates logical `v1`;
- changed protected files mark `version_status=invalid` and skip evaluation;
- one invalid evolved case removes evo/gain/AUC while retaining raw cases;
- unknown token usage remains `None` in events and summary;
- prompt/stderr/raw JSONL are bounded and stored in their specified directories.

- [ ] **Step 3: Run tests and verify RED**

Run [Framework]:

```bash
.venv/bin/python -m pytest tests/generals/test_pipeline.py tests/test_provider_integration_contracts.py tests/test_tracking_contracts.py -q
```

Expected: failures for missing pipeline and artifact behavior.

- [ ] **Step 4: Extend provider/snapshot boundaries minimally**

Add command metadata fields without changing existing callers.
`CodexProvider.cli_version() -> str | None` runs
`[executable, "--version"]` with the provider timeout and returns stripped
stdout only on exit `0`. `LocalWorkspaceSnapshotter.copy_snapshot(root,
destination) -> WorkspaceManifest` copies regular non-symlink files using
the existing excludes and writes the captured manifest.
`LocalWorkspaceSnapshotter.write_unified_patch(before_root, after_root,
destination) -> None` writes stable UTF-8 unified diffs for changed text
files and a one-line binary marker for changed binary files.

Store stderr in a dedicated bounded artifact and keep only its relative
reference in provider metadata. Redact environment values and authentication
paths. The provider command remains an argv list and never uses `shell=True`.

- [ ] **Step 5: Implement the pipeline in exact lifecycle order**

Define:

```python
@dataclass(frozen=True)
class PipelineResult:
    run_dir: Path
    raw_score: float | None
    evo_score: float | None
    gain: float | None
    act_count: int
    status: str
```

`GeneralsHLPipeline.from_paths(agentbench_root: Path, manifest_path: Path,
data_dir: Path, provider: ProviderAdapter) -> GeneralsHLPipeline` resolves
the production dependencies used by the CLI and live test.
`GeneralsHLPipeline.run() -> PipelineResult` implements the lifecycle below.

Implementation order:

1. require valid assets and write benchmark/population snapshots;
2. copy the immutable baseline template to a run workspace and snapshot `v0`;
3. evaluate `v0` on the 18 frozen cases;
4. run the 12 learning cases and build redacted replays/probes;
5. write the exact prompt;
6. call one `CodingAgentController.run_act`;
7. snapshot `v1`, write copied source snapshots and unified patch;
8. reject changes outside `strategy.py`, `STRATEGY.md`, and `tests/`;
9. run baseline tests and probe v1 on the shared learning states;
10. log behavior change and occupancy shift;
11. evaluate `v1` on the same 18 frozen cases;
12. compute raw/evo/gain and unit-labelled AUC values only when both points and axes exist;
13. run quality diagnostics, derive summary, and finish the run in `finally`.

Use `Run.create_coding_agent_controller` and existing budget/evaluation
contracts. Do not implement an alternate event writer.

- [ ] **Step 6: Run tests and commit**

Run [Framework]:

```bash
.venv/bin/python -m pytest tests/generals/test_pipeline.py tests/test_provider_integration_contracts.py tests/test_tracking_contracts.py -q
git add src/agentbench_frame/generals/pipeline.py src/agentbench_frame/tracking/providers.py src/agentbench_frame/tracking/controller.py src/agentbench_frame/tracking/snapshot.py tests/generals/fixtures/fake_codex.py tests/generals/test_pipeline.py tests/test_provider_integration_contracts.py tests/test_tracking_contracts.py
git commit -m "feat(generals): orchestrate one-act Codex loop"
```

Expected: tests pass; provider changes retain all existing Codex and Claude Code contract tests.

### Task 9: Expose CLI Commands, Quality Checks, and Local Report Data

**Files:**
- Create [Framework]: `src/agentbench_frame/generals/cli.py`
- Create [Framework]: `tests/generals/test_cli.py`
- Modify [Framework]: `src/agentbench_frame/cli.py`
- Modify [Framework]: `src/agentbench_frame/report/builder.py`
- Modify [Framework]: `src/agentbench_frame/report/templates/index.html`
- Test [Framework]: `tests/test_local_report_research.py`

**Interfaces:**
- Consumes: asset root, manifest, baseline workspace, data directory, Codex executable
- Produces: `agentbench generals prepare`, `agentbench generals eval`, `agentbench generals iterate`, CI-compatible summaries, and visible missing/incomplete states

- [ ] **Step 1: Write failing CLI tests**

```python
def test_generals_prepare_prints_validated_population(monkeypatch, capsys):
    code = main([
        "generals", "prepare",
        "--agentbench-root", str(FIXTURE_ROOT),
        "--manifest", str(FIXTURE_MANIFEST),
    ])
    assert code == 0
    assert "3 opponents" in capsys.readouterr().out


def test_generals_iterate_requires_explicit_asset_and_data_roots(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["generals", "iterate"])
    assert exc.value.code == 2
```

Also test `eval --version v0`, `iterate --codex-executable`, non-zero exit for
invalid assets, and non-zero exit for incomplete evaluation while retaining
the printed run directory.

- [ ] **Step 2: Write failing report tests**

Build a report from one complete and one incomplete fixture run. Assert:

```python
assert "raw score" in html.lower()
assert "evo score" in html.lower()
assert "gain" in html.lower()
assert "AUC_coding_agent_act" in html
assert "action disagreement" in html.lower()
assert "incomplete" in html.lower()
assert "unknown" in html.lower()
```

Do not assert interpolated points or substitute zero for missing fields.

- [ ] **Step 3: Run tests and verify RED**

Run [Framework]:

```bash
.venv/bin/python -m pytest tests/generals/test_cli.py tests/test_local_report_research.py -q
```

Expected: CLI tests fail because the command group is missing; report tests fail for missing Generals fields.

- [ ] **Step 4: Implement the command group**

Expose:

```text
agentbench generals prepare --agentbench-root PATH --manifest PATH
agentbench generals eval --agentbench-root PATH --manifest PATH --workspace PATH --version v0 --data-dir PATH
agentbench generals iterate --agentbench-root PATH --manifest PATH --data-dir PATH --codex-executable PATH
```

`prepare` validates and builds rank 2 in a generated build directory without
editing its source tree. `eval` runs the frozen 18 cases. `iterate` runs the
complete pipeline. Print the run directory before returning failure for an
incomplete run. Return `0` only for structurally complete commands/runs.

- [ ] **Step 5: Extend local report data without redesigning templates**

Use existing research sections from `origin/worktree/framework`. Add
Generals fields through generic event/summary lookups:

- raw/evo/gain;
- unit-labelled AUC;
- learning/evaluation/total budgets;
- action-disagreement mean/trace;
- evaluation and quality status.

Render `unknown`/`incomplete` explicitly and leave missing chart points absent.

- [ ] **Step 6: Run tests and commit**

Run [Framework]:

```bash
.venv/bin/python -m pytest tests/generals/test_cli.py tests/test_local_report_research.py -q
git add src/agentbench_frame/cli.py src/agentbench_frame/generals/cli.py src/agentbench_frame/report/builder.py src/agentbench_frame/report/templates/index.html tests/generals/test_cli.py tests/test_local_report_research.py
git commit -m "feat(generals): expose pilot CLI and reports"
```

Expected: tests pass and existing non-Generals CLI/report tests remain green.

### Task 10: Verify Real Opponents and Run the One-Act Codex Pilot

**Files:**
- Create [Framework]: `tests/generals/test_live.py`
- Modify [Assets]: `backend_sources/corpus/28_generals/README.md`
- Modify [Framework]: `README.md`
- Generated, do not commit: C++ binaries, temporary opponent copies, run workspaces, match logs, provider JSONL, reports

**Interfaces:**
- Consumes: completed CLI, real AgentBench asset worktree, authenticated Codex CLI
- Produces: one verified real run directory and reproduction commands

- [ ] **Step 1: Write explicitly gated live tests**

```python
LIVE = os.environ.get("AGENTBENCH_RUN_LIVE_GENERALS") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="set AGENTBENCH_RUN_LIVE_GENERALS=1")


def test_all_frozen_opponents_start_and_complete_one_decision(real_assets, tmp_path):
    prepared = prepare_opponents(
        real_assets.layout, tmp_path / "prepared", Path(sys.executable)
    )
    for opponent in prepared:
        engine = OfficialGeneralsEngine(real_assets.layout.engine_root, seed=280101)
        with ManagedAgentProcess(
            opponent, real_assets.config.limits, tmp_path / opponent.agent_id
        ) as player:
            player.send_initial(engine.initial_observation(0))
            commands = player.request_turn()
        assert commands[-1] == (8,), opponent.agent_id


def test_real_codex_one_act_produces_finalized_run(real_assets, tmp_path):
    pipeline = GeneralsHLPipeline.from_paths(
        agentbench_root=real_assets.layout.root,
        manifest_path=real_assets.manifest_path,
        data_dir=tmp_path / "data",
        provider=CodexProvider(executable=shutil.which("codex") or "codex"),
    )
    result = pipeline.run()
    assert result.act_count == 1
    assert (result.run_dir / "provider" / "codex.raw.jsonl").stat().st_size > 0
    assert (result.run_dir / "quality.json").exists()
```

The Codex test asserts evidence integrity, not positive gain.

- [ ] **Step 2: Run the complete non-live verification suite**

Run [Framework]:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src tests
git diff --check
```

Expected: all tests pass, compileall exits `0`, and diff check prints nothing.

- [ ] **Step 3: Prepare and smoke-test real opponents**

Run [Framework]:

```bash
.venv/bin/agentbench generals prepare \
  --agentbench-root "$ASSET_WT" \
  --manifest "$ASSET_WT/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml"

AGENTBENCH_ASSET_ROOT="$ASSET_WT" \
AGENTBENCH_RUN_LIVE_GENERALS=1 \
.venv/bin/python -m pytest tests/generals/test_live.py::test_all_frozen_opponents_start_and_complete_one_decision -q
```

Expected: the C++ opponent builds outside its source tree and all three opponents complete an SDK protocol decision.

- [ ] **Step 4: Run one real raw evaluation before spending a Codex act**

Run [Framework]:

```bash
.venv/bin/agentbench generals eval \
  --agentbench-root "$ASSET_WT" \
  --manifest "$ASSET_WT/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml" \
  --workspace "$ASSET_WT/backend_sources/corpus/28_generals/baselines/hl_v0" \
  --version v0 \
  --data-dir "$FRAMEWORK_WT/agentbench_data"
```

Expected: 18 finalized cases, an explicit complete/incomplete evaluation status, and a printed run directory. If incomplete, inspect the retained failing case and fix the harness through a new failing regression test before continuing.

- [ ] **Step 5: Run the real one-act Codex closed loop**

Run [Framework]:

```bash
.venv/bin/agentbench generals iterate \
  --agentbench-root "$ASSET_WT" \
  --manifest "$ASSET_WT/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml" \
  --data-dir "$FRAMEWORK_WT/agentbench_data" \
  --codex-executable "$(command -v codex)"
```

Expected: one run directory containing 18 raw evaluation cases, 12 learning
cases, one raw Codex JSONL stream, v0/v1 manifests and patch, behavior-change
records, 18 evolved evaluation cases when v1 is runnable, `quality.json`,
`events.jsonl`, and `summary.json`. A negative gain is acceptable. An
incomplete status must remain explicit.

- [ ] **Step 6: Validate data and render the local report**

Run [Framework]:

```bash
.venv/bin/agentbench data check --data-dir "$FRAMEWORK_WT/agentbench_data"
.venv/bin/agentbench report --data-dir "$FRAMEWORK_WT/agentbench_data" --output "$FRAMEWORK_WT/_site"
test -s "$FRAMEWORK_WT/_site/index.html"
```

Expected: data validation exits `0` for structurally valid complete or explicitly incomplete runs; report generation exits `0`; index HTML is non-empty.

- [ ] **Step 7: Update reproduction documentation**

Add to both READMEs:

- exact prepare/eval/iterate commands;
- frozen opponent IDs and seed split;
- output layout;
- no-Docker security limitation;
- meaning of valid IA/TLE versus invalid harness failure;
- action disagreement versus unavailable strict policy KL;
- local Codex installation/authentication requirement.

- [ ] **Step 8: Run final verification and commit docs/live test**

Run [Framework]:

```bash
.venv/bin/python -m pytest -q
git diff --check
git status --short
git add README.md tests/generals/test_live.py
git commit -m "docs(generals): document reproducible HL pilot"
```

Run [Assets]:

```bash
"$FRAMEWORK_WT/.venv/bin/python" -m pytest "$ASSET_WT/backend_sources/corpus/28_generals/baselines/hl_v0/tests" -q
git diff --check
git status --short
git add backend_sources/corpus/28_generals/README.md
git commit -m "docs(generals): document pilot assets"
```

Expected: all tests pass immediately before each completion claim; only intended files are committed; generated run data and binaries remain untracked or ignored.

## Final Verification Checklist

- [ ] Framework full pytest suite passes after the last implementation change.
- [ ] Baseline pytest suite passes after the last asset change.
- [ ] `compileall` and `git diff --check` pass in both repositories where applicable.
- [ ] All three real opponents complete a protocol smoke decision.
- [ ] The frozen evaluation spec contains exactly 18 unique cases.
- [ ] Learning contains exactly 12 cases and no high-tier/evaluation evidence.
- [ ] One real Codex act is recorded with raw JSONL and bounded stderr.
- [ ] v0/v1 manifests and `v0-to-v1.patch` exist.
- [ ] Raw/evo/gain and unit-labelled AUC are present only when their prerequisites are complete.
- [ ] Action disagreement is labelled separately from occupancy shift; strict policy KL is explicitly unavailable.
- [ ] Missing values remain unknown/absent rather than zero.
- [ ] `quality.json`, `events.jsonl`, `summary.json`, and local report all exist.
- [ ] No authentication data, historical source modifications, generated binaries, run data, unrelated `uv.lock`, or unrelated AgentBench deletions are committed.
