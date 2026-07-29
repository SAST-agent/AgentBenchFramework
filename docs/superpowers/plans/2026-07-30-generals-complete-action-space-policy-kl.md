# Generals Complete Action Space and Policy-KL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible Generals measurement pipeline that exactly counts the complete canonical macro-action support on 12 frozen real states and produces strict uniform-epsilon policy KL for v0→v6.

**Architecture:** Add a backward-compatible `CanonicalActionSpace` boundary to the Framework, serialize complete official-engine measurement states, define Generals macro-actions as an official-transition prefix graph, and count suffix macros with an arbitrary-precision memoized recurrence. A separate append-only measurement pipeline collects human-self-play states, probes immutable historical policies, calculates controlled-reference KL, and exposes complete or explicitly missing results to CI.

**Tech Stack:** Python 3.11, standard-library `decimal`, `sqlite3`, `dataclasses`, `hashlib`, `subprocess`, pytest, TOML benchmark assets, Jinja2 static reports, official Generals Python engine.

## Global Constraints

- Preserve every existing v0-v6 run, policy source, score, replay, and `summary.json` byte-for-byte.
- The action support is the complete normalized behavioral macro support; do not impose `MAX_PRIMITIVES`.
- Every legal primitive is decided by a cloned official engine transition, not by a second reimplementation of legality.
- Use strict \(U_s(a)=1/|A(s)|\); no sampling, truncation, lower bound, or approximate support count may enter KL.
- Use natural logarithms and nats. Primary epsilon is `0.01`; sensitivity values are `0.001`, `0.01`, `0.05`, and `0.1`.
- The primary domain is exactly 12 states from strongest-human self-play at seeds `289101`, `289202`, and `289303`, both seats, seat-local decisions 2 and 10.
- All seven historical policies use one common frozen reference set and exact content hashes from the approved design.
- The metric name is `controlled_reference_policy_kl`; never rename it to episode KL, trajectory KL, epistemic information gain, or reward.
- A missing primary state makes the transition aggregate `null`; never calculate a subset mean or interpolate.
- Cardinalities and high-precision nats are persisted as decimal strings; display floats are derived convenience values.
- RL remains unavailable until it exposes a distribution on the exact same action-space `spec_id`.
- Use `apply_patch` for hand-written file changes, TDD for every behavior change, and one focused commit after each task.
- Framework worktree: `/home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl`.
- Assets worktree: `/home/cathy/AgentBench/AgentBench/.worktrees/generals-assets`.

---

## File and responsibility map

### Framework files to create

- `src/agentbench_frame/eval/action_space.py` — generic canonical action-space protocol.
- `src/agentbench_frame/generals/measurement_state.py` — lossless official `GameState` JSON round-trip.
- `src/agentbench_frame/generals/macro_action_space.py` — Generals command grammar, canonicalization, and official legal transitions.
- `src/agentbench_frame/generals/macro_counter.py` — exact arbitrary-precision recurrence and resumable SQLite cache.
- `src/agentbench_frame/generals/historical_policy.py` — immutable source resolution and isolated state probing.
- `src/agentbench_frame/generals/policy_probe_worker.py` — one-shot subprocess worker that invokes frozen `main.agent`.
- `src/agentbench_frame/generals/policy_kl_pipeline.py` — reference collection, counting, probing, KL events, summary, and recovery.
- `tests/generals/test_measurement_state.py` — state completeness and round-trip contracts.
- `tests/generals/test_macro_action_space.py` — command 1-8 and canonicalization tests.
- `tests/generals/test_macro_counter.py` — exact recurrence and resume tests.
- `tests/generals/test_historical_policy.py` — source hash, isolation, determinism, and legality tests.
- `tests/generals/test_policy_kl_pipeline.py` — fake end-to-end and missing-value contracts.
- `tests/generals/test_policy_kl_cli.py` — CLI surface and wiring tests.

### Framework files to modify

- `src/agentbench_frame/eval/information_gain.py` — exact deterministic uniform-epsilon KL.
- `src/agentbench_frame/eval/__init__.py` — public exports.
- `src/agentbench_frame/generals/models.py` — reference and historical-policy manifest records.
- `src/agentbench_frame/generals/assets.py` — strict reference-manifest loader.
- `src/agentbench_frame/generals/engine.py` — measurement snapshot, reconstruction, clone, and single-primitive boundary.
- `src/agentbench_frame/generals/match.py` — optional pre-decision measurement-state stream.
- `src/agentbench_frame/generals/cli.py` — `measure-policy-kl` and `recover-policy-kl`.
- `src/agentbench_frame/tracking/quality.py` — four new finalized event types.
- `src/agentbench_frame/report/builder.py` — controlled-reference KL research projection.
- `src/agentbench_frame/report/templates/index.html` — KL-iteration curve, sensitivity, coverage, and support table.
- `tests/test_measurement_contracts.py` — generic protocol and exact formula contracts.
- `tests/generals/test_assets.py` — reference-manifest validation.
- `tests/generals/test_match.py` — measurement capture does not change ordinary replay behavior.
- `tests/test_local_report_research.py` — CI projection and missing-gap behavior.

### Assets files to create or modify

- Create `backend_sources/corpus/28_generals/benchmark/policy-kl-reference-v1.toml` — frozen collector, states, epsilon, and history.
- Create `backend_sources/corpus/28_generals/tests/test_policy_kl_reference_contract.py` — byte-level manifest requirements.
- Modify `backend_sources/corpus/28_generals/README.md` — exact reproduction command and metric limitations.

---

### Task 1: Generic canonical support and exact deterministic KL

**Files:**
- Create: `src/agentbench_frame/eval/action_space.py`
- Modify: `src/agentbench_frame/eval/information_gain.py`
- Modify: `src/agentbench_frame/eval/__init__.py`
- Modify: `tests/test_measurement_contracts.py`

**Interfaces:**
- Consumes: canonical state/action values supplied by a game adapter.
- Produces: `CanonicalActionSpace`, `uniform_smoothed_deterministic_kl(new_action, old_action, support_size, epsilon, precision=80) -> Decimal`.

- [ ] **Step 1: Write failing protocol and formula tests**

Append tests that require the shared protocol and compare the closed form with an explicitly materialized probability vector:

```python
from decimal import Decimal


def test_deterministic_uniform_kl_matches_explicit_distribution():
    from agentbench_frame.eval.information_gain import (
        policy_kl,
        uniform_smoothed_deterministic_kl,
    )

    epsilon = Decimal("0.01")
    support_size = 5
    measured = uniform_smoothed_deterministic_kl(
        new_action=("b",),
        old_action=("a",),
        support_size=support_size,
        epsilon=epsilon,
    )
    q = 1.0 - float(epsilon) + float(epsilon) / support_size
    r = float(epsilon) / support_size
    explicit = policy_kl(
        [r, q, r, r, r],
        [q, r, r, r, r],
    )
    assert float(measured) == pytest.approx(explicit)


def test_deterministic_uniform_kl_is_zero_for_equal_action():
    from agentbench_frame.eval.information_gain import (
        uniform_smoothed_deterministic_kl,
    )

    assert uniform_smoothed_deterministic_kl(
        new_action=((8,),),
        old_action=((8,),),
        support_size=10**30,
        epsilon="0.01",
    ) == Decimal(0)


def test_deterministic_uniform_kl_rejects_invalid_support():
    from agentbench_frame.eval.information_gain import (
        uniform_smoothed_deterministic_kl,
    )

    with pytest.raises(ValueError, match="support_size"):
        uniform_smoothed_deterministic_kl("b", "a", 0, "0.01")
    with pytest.raises(ValueError, match="one legal action"):
        uniform_smoothed_deterministic_kl("b", "a", 1, "0.01")
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_measurement_contracts.py -q
```

Expected: collection or import failure because `CanonicalActionSpace` and `uniform_smoothed_deterministic_kl` do not exist.

- [ ] **Step 3: Add the protocol and high-precision formula**

Implement the protocol in `eval/action_space.py`:

```python
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable


CanonicalMacro = tuple[tuple[int, ...], ...]


@runtime_checkable
class CanonicalActionSpace(Protocol):
    spec_id: str

    def canonicalize(
        self, state: Mapping[str, Any], action: Sequence[Sequence[int]]
    ) -> CanonicalMacro: ...

    def contains(
        self, state: Mapping[str, Any], action: Sequence[Sequence[int]]
    ) -> bool: ...

    def cardinality(self, state: Mapping[str, Any]) -> int: ...


@runtime_checkable
class EnumerableCanonicalActionSpace(CanonicalActionSpace, Protocol):
    def iter_actions(
        self, state: Mapping[str, Any]
    ) -> Iterable[CanonicalMacro]: ...
```

The base protocol is count-only and therefore supports Generals without
materializing its macro support. `EnumerableCanonicalActionSpace` is the
optional refinement for games whose complete support can be iterated.

Implement the exact-support formula using `decimal.localcontext()`:

```python
def uniform_smoothed_deterministic_kl(
    new_action,
    old_action,
    support_size: int,
    epsilon,
    precision: int = 80,
) -> Decimal:
    if support_size < 1:
        raise ValueError("support_size must be positive")
    eps = Decimal(str(epsilon))
    if not Decimal(0) < eps < Decimal(1):
        raise ValueError("epsilon must be strictly between zero and one")
    if new_action == old_action:
        return Decimal(0)
    if support_size == 1:
        raise ValueError("different actions cannot share one legal action")
    with localcontext() as context:
        context.prec = precision
        one_minus = Decimal(1) - eps
        return +(one_minus * (Decimal(1) + Decimal(support_size) * one_minus / eps).ln())
```

Export both names from `eval/__init__.py`.

- [ ] **Step 4: Run focused and existing measurement tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_measurement_contracts.py tests/test_external_boundary_contracts.py -q
```

Expected: all tests pass and the pre-existing vector KL API remains unchanged.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/agentbench_frame/eval/action_space.py \
  src/agentbench_frame/eval/information_gain.py \
  src/agentbench_frame/eval/__init__.py \
  tests/test_measurement_contracts.py
git commit -m "feat(eval): add canonical support KL"
```

---

### Task 2: Freeze the reference and historical lineage manifest

**Files:**
- Create: `backend_sources/corpus/28_generals/benchmark/policy-kl-reference-v1.toml`
- Create: `backend_sources/corpus/28_generals/tests/test_policy_kl_reference_contract.py`
- Modify: `backend_sources/corpus/28_generals/README.md`
- Modify: `src/agentbench_frame/generals/models.py`
- Modify: `src/agentbench_frame/generals/assets.py`
- Modify: `tests/generals/test_assets.py`

**Interfaces:**
- Consumes: `PilotConfig` and the frozen high opponent.
- Produces: `HistoricalPolicyConfig`, `PolicyKLReferenceConfig`, and `load_policy_kl_reference_config(path, pilot)`.

- [ ] **Step 1: Write failing Framework manifest tests**

Add the records and loader expectations:

```python
def test_policy_kl_reference_manifest_is_strict(real_assets):
    pilot = load_pilot_config(real_assets / "benchmark/pilot-v1.toml")
    config = load_policy_kl_reference_config(
        real_assets / "benchmark/policy-kl-reference-v1.toml",
        pilot,
    )
    assert config.measurement_id == "generals-policy-kl-reference-v1"
    assert config.opponent_id == "advanced-rank02-robinliu-v18"
    assert config.seeds == (289101, 289202, 289303)
    assert config.seats == (0, 1)
    assert config.decision_numbers == (2, 10)
    assert config.epsilons == ("0.001", "0.01", "0.05", "0.1")
    assert config.primary_epsilon == "0.01"
    assert tuple(item.version for item in config.history) == (
        "v0", "v1", "v2", "v3", "v4", "v5", "v6",
    )
```

Add negative tests for seed overlap, unordered seats, duplicate versions, malformed 64-character hashes, a non-high opponent, and a primary epsilon absent from the sensitivity set.

- [ ] **Step 2: Run the Framework asset test and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_assets.py -q
```

Expected: import failure for the missing records and loader.

- [ ] **Step 3: Add the strict dataclasses and loader**

Add:

```python
@dataclass(frozen=True)
class HistoricalPolicyConfig:
    version: str
    run_id: str
    content_hash: str


@dataclass(frozen=True)
class PolicyKLReferenceConfig:
    measurement_id: str
    opponent_id: str
    seeds: tuple[int, ...]
    seats: tuple[int, ...]
    decision_numbers: tuple[int, ...]
    epsilons: tuple[str, ...]
    primary_epsilon: str
    history: tuple[HistoricalPolicyConfig, ...]
```

`load_policy_kl_reference_config()` must require the exact measurement ID, seeds, seats, decision numbers, epsilon strings, seven ordered versions, approved run IDs, approved hashes, and the first pilot opponent.

- [ ] **Step 4: Create the frozen asset manifest**

Use exactly:

```toml
measurement_id = "generals-policy-kl-reference-v1"
opponent_id = "advanced-rank02-robinliu-v18"
seeds = [289101, 289202, 289303]
seats = [0, 1]
decision_numbers = [2, 10]
epsilons = ["0.001", "0.01", "0.05", "0.1"]
primary_epsilon = "0.01"

[[history]]
version = "v0"
run_id = "20260726_1631_f56f789f"
content_hash = "bfab30cdaaa0ad8e0ac6ed1b9ab047417c621cc22dd22fd95b0946c3af4bd5fa"

[[history]]
version = "v1"
run_id = "20260726_1631_f56f789f"
content_hash = "17356a28682378c4877f7da7aeb8ac90b8891e2685012dec8cdec990cb282e79"

[[history]]
version = "v2"
run_id = "20260727_0639_89578eec"
content_hash = "1e8a2ce4fa38b4171f6f8d77d42b80d48438f5213b08d543714d8241e292c565"

[[history]]
version = "v3"
run_id = "20260728_1122_57f647d5"
content_hash = "a9f27eb2a02ba452e9363022d97ffa5556c0f3a02eef120061e73e93c68ce815"

[[history]]
version = "v4"
run_id = "20260729_0818_e6bcb9b3"
content_hash = "5c12e7c92843cbc18b742ab7be71c684cfa0a0aac8e5dbb5a24fc7b905fe959f"

[[history]]
version = "v5"
run_id = "20260729_0818_e6bcb9b3"
content_hash = "facd39c8a0c064823da68f3dd3eba4a832c08815f75369407112ca087f1ecd9e"

[[history]]
version = "v6"
run_id = "20260729_1653_af8eda26"
content_hash = "974050ee1a3d4b4c4f96e61f5af39b4e52e2cfbc50f146f9e8b4ac96c4ac798b"
```

The assets contract test must also assert that seeds 289101/289202/289303 occur only in this manifest and its test/documentation.

Add this exact operator-facing section to the Generals README (using the
worktree paths shown here so the command is directly reproducible):

````markdown
## Controlled-reference policy KL

Run the strict uniform-support measurement:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/.venv/bin/python \
  -m agentbench_frame.cli generals measure-policy-kl \
  --agentbench-root /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets \
  --manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml \
  --reference-manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/policy-kl-reference-v1.toml \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data
```

The metric uses exact complete macro-action cardinalities on 12 frozen early
real states and reports deterministic-policy change under uniform epsilon
smoothing. It is not trajectory KL, semantic action distance, reward
information gain, or epistemic information gain. Any incomplete state makes
the affected aggregate unavailable.
````

- [ ] **Step 5: Run both repository tests**

Run in Framework:

```bash
.venv/bin/python -m pytest tests/generals/test_assets.py -q
```

Run in Assets:

```bash
/home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/.venv/bin/python \
  -m pytest -p no:cacheprovider \
  backend_sources/corpus/28_generals/tests/test_policy_kl_reference_contract.py -q
```

Expected: both pass.

- [ ] **Step 6: Commit Task 2 in both repositories**

Framework:

```bash
git add src/agentbench_frame/generals/models.py \
  src/agentbench_frame/generals/assets.py tests/generals/test_assets.py
git commit -m "feat(generals): load policy KL reference spec"
```

Assets:

```bash
git add backend_sources/corpus/28_generals/benchmark/policy-kl-reference-v1.toml \
  backend_sources/corpus/28_generals/tests/test_policy_kl_reference_contract.py \
  backend_sources/corpus/28_generals/README.md
git commit -m "feat(generals): freeze policy KL reference set"
```

---

### Task 3: Lossless official measurement-state round-trip

**Files:**
- Create: `src/agentbench_frame/generals/measurement_state.py`
- Create: `tests/generals/test_measurement_state.py`
- Modify: `src/agentbench_frame/generals/engine.py`
- Modify: `src/agentbench_frame/generals/match.py`
- Modify: `tests/generals/test_match.py`

**Interfaces:**
- Consumes: official `GameState`, actor seat, official engine module.
- Produces: `MEASUREMENT_STATE_SCHEMA`, `serialize_measurement_state(state, actor)`, `measurement_state_id(payload)`, `reconstruct_measurement_state(main_module, payload, replay_path)`, `OfficialGeneralsEngine.from_measurement_state(...)`, `apply_primitive(...)`.

- [ ] **Step 1: Write failing round-trip tests**

Cover initial state, a state after both army and general movement, a state with upgrades/skills/weapons, and the farmer defense value `1.5`:

```python
def test_measurement_state_round_trips_every_semantic_field(real_engine_root, tmp_path):
    engine = OfficialGeneralsEngine(real_engine_root, 289101, tmp_path / "a.jsonl")
    before = engine.measurement_state(actor=0)
    restored = OfficialGeneralsEngine.from_measurement_state(
        real_engine_root,
        before,
        tmp_path / "b.jsonl",
    )
    assert restored.measurement_state(actor=0) == before
    assert restored.measurement_state_id(actor=0) == engine.measurement_state_id(actor=0)
    assert before["state"]["rest_move_step"] == [2, 2]
    assert all("rest_move" in item for item in before["state"]["generals"])
```

Add a negative test that removes `rest_move_step` and expects `MeasurementStateError("missing state field: rest_move_step")`.

- [ ] **Step 2: Run and verify RED**

```bash
.venv/bin/python -m pytest tests/generals/test_measurement_state.py -q
```

Expected: import failure for `measurement_state`.

- [ ] **Step 3: Implement canonical serialization and reconstruction**

The payload shape is:

```python
{
    "schema": "generals-measurement-state-v1",
    "actor": 0,
    "state": {
        "round": 1,
        "coin": [40, 40],
        "active_super_weapon": [],
        "super_weapon_unlocked": [False, False],
        "super_weapon_cd": [-1, -1],
        "tech_level": [[2, 0, 0, 0], [2, 0, 0, 0]],
        "rest_move_step": [2, 2],
        "next_generals_id": 16,
        "winner": -1,
        "generals": [],
        "board": [],
    },
}
```

Preserve general concrete type, floats without integer coercion, general `rest_move`, active-weapon identity, and each cell's active-weapon references. Exclude replay path and changed-cell bookkeeping. Sort generals by ID and cells by row/column before hashing.

- [ ] **Step 4: Add engine and match boundaries**

Add:

```python
@dataclass(frozen=True)
class PrimitiveOutcome:
    valid: bool
    terminal: bool
    winner: int | None


def measurement_state(self, actor: int) -> dict: ...
def measurement_state_id(self, actor: int) -> str: ...

@classmethod
def from_measurement_state(
    cls, engine_root: Path, payload: Mapping[str, Any], replay_path: Path
) -> "OfficialGeneralsEngine": ...

def apply_primitive(
    self, player: int, command: Sequence[int]
) -> PrimitiveOutcome: ...
```

`apply_primitive()` executes exactly one opcode 1-7 and checks `is_game_over`;
it never performs a round update and never converts an illegal measurement
probe into an in-game loss.

Add `capture_measurement_states: bool = False` to `GeneralsMatchRunner.run()`.
When true, write `measurement-state.jsonl` before each player decision with
global step, seat-local decision number, round, player, ID, and snapshot. When
false, ordinary `replay.jsonl` and `MatchResult` remain byte-compatible.

- [ ] **Step 5: Run measurement and match tests**

```bash
.venv/bin/python -m pytest \
  tests/generals/test_measurement_state.py \
  tests/generals/test_match.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/agentbench_frame/generals/measurement_state.py \
  src/agentbench_frame/generals/engine.py \
  src/agentbench_frame/generals/match.py \
  tests/generals/test_measurement_state.py \
  tests/generals/test_match.py
git commit -m "feat(generals): snapshot complete measurement state"
```

---

### Task 4: Complete canonical Generals macro-action graph

**Files:**
- Create: `src/agentbench_frame/generals/macro_action_space.py`
- Create: `tests/generals/test_macro_action_space.py`

**Interfaces:**
- Consumes: measurement snapshot, official engine root and hash.
- Produces: `ActionSpaceSpec`, `PrimitiveTransition`, `GeneralsMacroActionSpaceV1`, stable `spec_id`, canonical macro membership, and legal successor enumeration.

- [ ] **Step 1: Write failing command-family and canonicalization tests**

Use controlled official states to activate each command family:

```python
@pytest.mark.parametrize("opcode", range(1, 9))
def test_action_space_exposes_every_official_command_family(
    action_space, controlled_state_by_opcode, opcode
):
    state = controlled_state_by_opcode[opcode]
    if opcode == 8:
        assert action_space.contains(state, [[8]])
    else:
        assert any(
            transition.command[0] == opcode
            for transition in action_space.legal_transitions(state)
        )


def test_canonicalize_collapses_army_request_sentinel(action_space, army_state):
    assert action_space.canonicalize(
        army_state, [[1, 4, 5, 1, 1_000_000_000], [8]]
    ) == ((1, 4, 5, 1, 9), (8,))


def test_canonicalize_drops_ignored_skill_coordinates(action_space, skill_state):
    assert action_space.canonicalize(
        skill_state, [[4, 0, 3, 7, 8], [8]]
    ) == ((4, 0, 3), (8,))
```

Add tests for general destinations, all three upgrades, five skills, four technologies, four weapons including transmission source/destination ordering, recruitment, immediate end, terminal suffix removal, malformed commands, exactly one final `[8]`, and stable row/column/command ordering.

- [ ] **Step 2: Run and verify RED**

```bash
.venv/bin/python -m pytest tests/generals/test_macro_action_space.py -q
```

Expected: import failure because `GeneralsMacroActionSpaceV1` does not exist.

- [ ] **Step 3: Implement the versioned spec and finite candidate domains**

Add:

```python
@dataclass(frozen=True)
class PrimitiveTransition:
    command: tuple[int, ...]
    state_after: Mapping[str, Any]
    terminal: bool


@dataclass(frozen=True)
class ActionSpaceSpec:
    schema: str
    engine_hash: str
    measurement_state_schema: str
    canonicalization_version: str
    spec_id: str
```

`primitive_candidates()` emits, in deterministic order:

- command 1 for every board source, directions 1-4, and distinct positive amount `1..army-1`;
- command 2 for every general ID and all 225 destinations;
- command 3 for every general ID and qualities 1-3;
- command 4 skill 1-2 over all 225 targets and skill 3-5 without targets;
- command 5 technologies 1-4;
- command 6 weapons 1, 2, and 4 over 225 targets and weapon 3 over all 225×225 destination/source pairs;
- command 7 over all 225 positions.

Each candidate runs through `OfficialGeneralsEngine.from_measurement_state()` and `apply_primitive()`. Only successful transitions are yielded.

- [ ] **Step 4: Implement canonical macro validation**

`canonicalize()` must:

1. require a sequence of integer command arrays;
2. process commands from the original state in order;
3. normalize command shapes and behavior-equivalent values;
4. require every primitive to succeed officially;
5. stop after a terminal primitive or first `[8]`;
6. append exactly one conceptual `[8]`; and
7. reject missing end markers, early end followed by submitted primitives, or malformed opcodes.

`contains()` returns false rather than raising for malformed or illegal input.
The production Generals class implements the count-only
`CanonicalActionSpace`; it does not claim or expose the optional enumerable
protocol.

- [ ] **Step 5: Run action-space and official replay tests**

```bash
.venv/bin/python -m pytest \
  tests/generals/test_macro_action_space.py \
  tests/generals/test_engine.py \
  tests/generals/test_replay_v4.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/agentbench_frame/generals/macro_action_space.py \
  tests/generals/test_macro_action_space.py
git commit -m "feat(generals): define complete macro action graph"
```

---

### Task 5: Exact arbitrary-precision support counter

**Files:**
- Create: `src/agentbench_frame/generals/macro_counter.py`
- Create: `tests/generals/test_macro_counter.py`

**Interfaces:**
- Consumes: `GeneralsMacroActionSpaceV1.legal_transitions()`, measurement-state IDs, operational guards.
- Produces: `ExactCountResult`, `ExactCountIncomplete`, `ExactMacroCounter.count(state)`, resumable SQLite cache, `GeneralsMacroActionSpaceV1.cardinality()`.

- [ ] **Step 1: Write failing recurrence and resume tests**

Use a finite fake graph with a shared suffix:

```python
def test_exact_counter_counts_distinct_prefixes_with_shared_successor(tmp_path):
    graph = FakePrefixGraph({
        "root": [
            PrimitiveTransition((1,), {"id": "shared"}, False),
            PrimitiveTransition((2,), {"id": "shared"}, False),
        ],
        "shared": [PrimitiveTransition((3,), {"id": "terminal"}, True)],
    })
    result = ExactMacroCounter(graph, tmp_path / "cache.sqlite").count({"id": "root"})
    # root end + two * (shared end + terminal command)
    assert result.support_size == 5
    assert result.status == "complete"


def test_counter_resumes_only_exact_completed_subtrees(tmp_path):
    cache = tmp_path / "cache.sqlite"
    with pytest.raises(ExactCountIncomplete):
        ExactMacroCounter(
            deep_graph(), cache, max_expanded_states=2
        ).count({"id": "root"})
    resumed = ExactMacroCounter(
        deep_graph(), cache, max_expanded_states=100
    ).count({"id": "root"})
    assert resumed.status == "complete"
    assert resumed.cache_hits > 0
```

Add tests for immediate end count 1, terminal branch count 1, big integers above `2**53`, cycle detection, spec-ID cache isolation, decimal-string persistence, timeout status, and corrupted cache rejection.

- [ ] **Step 2: Run and verify RED**

```bash
.venv/bin/python -m pytest tests/generals/test_macro_counter.py -q
```

Expected: import failure for `ExactMacroCounter`.

- [ ] **Step 3: Implement SQLite-backed exact recursion**

Add:

```python
@dataclass(frozen=True)
class ExactCountResult:
    status: str
    support_size: int | None
    expanded_states: int
    legal_edges: int
    cache_hits: int
    elapsed_time_s: float
    error: str | None = None


class ExactCountIncomplete(RuntimeError):
    def __init__(self, result: ExactCountResult): ...
```

Use a SQLite table keyed by `(spec_id, actor, measurement_state_id)` with only
completed exact decimal counts. Maintain an in-process recursion stack for
cycle detection. The recurrence is immediate end `1`, plus `1` for each
terminal primitive, plus the exact cached or recursively calculated successor
count for each nonterminal primitive.

Operational `max_wall_time_s` and `max_expanded_states` raise
`ExactCountIncomplete` after committing completed subtrees. They never store a
partial value for the current node.

- [ ] **Step 4: Bind `cardinality()` to the counter**

Allow `GeneralsMacroActionSpaceV1` to accept an `ExactMacroCounter` or cache
path. `cardinality(state)` returns the integer only for a complete result and
propagates `ExactCountIncomplete` otherwise.

- [ ] **Step 5: Run counter, action-space, and generic KL tests**

```bash
.venv/bin/python -m pytest \
  tests/generals/test_macro_counter.py \
  tests/generals/test_macro_action_space.py \
  tests/test_measurement_contracts.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit Task 5**

```bash
git add src/agentbench_frame/generals/macro_counter.py \
  src/agentbench_frame/generals/macro_action_space.py \
  tests/generals/test_macro_counter.py
git commit -m "feat(generals): count exact macro support"
```

---

### Task 6: Probe immutable historical policies on reconstructed states

**Files:**
- Create: `src/agentbench_frame/generals/historical_policy.py`
- Create: `src/agentbench_frame/generals/policy_probe_worker.py`
- Create: `tests/generals/test_historical_policy.py`

**Interfaces:**
- Consumes: `HistoricalPolicyConfig`, data directory, measurement snapshot, official SDK root.
- Produces: `HistoricalPolicySource`, `PolicyProbeResult`, `resolve_historical_policies()`, `probe_historical_policy()`.

- [ ] **Step 1: Write failing source-integrity and worker tests**

```python
def test_history_resolver_requires_exact_manifest_hash(fake_history, tmp_path):
    fake_history.version_manifest.write_text(
        json.dumps({"content_hash": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(HistoricalPolicyError, match="content hash"):
        resolve_historical_policies(fake_history.config, tmp_path)


def test_probe_runs_twice_and_returns_one_canonical_macro(
    historical_policy, measurement_state, sdk_root
):
    result = probe_historical_policy(
        historical_policy,
        measurement_state,
        sdk_root=sdk_root,
        repeats=2,
        timeout_s=5.0,
    )
    assert result.status == "complete"
    assert result.deterministic is True
    assert result.raw_actions[0] == result.raw_actions[1]
    assert result.raw_actions[0][-1] == [8]
```

Add tests for source immutability, process timeout, stderr capture, malformed
JSON, nondeterminism, `main.agent` import failure, SDK controller convention
that appends `[8]`, and a v6-style state view that observes
`rest_move_step` only through reconstructed `GameState`.

- [ ] **Step 2: Run and verify RED**

```bash
.venv/bin/python -m pytest tests/generals/test_historical_policy.py -q
```

Expected: import failure for `historical_policy`.

- [ ] **Step 3: Implement strict source resolution**

Resolve each source as:

```text
<data-dir>/runs/28_generals/generals-hl/<run_id>/versions/<version>/source
```

Read the adjacent `manifest.json`, require the approved content hash, verify
every listed file hash, copy to the new run's `policies/<version>/source`, and
verify the copied tree again. Do not import directly from the historical run.

- [ ] **Step 4: Implement the one-shot worker**

The parent writes one JSON request containing source path, SDK path, round,
seat, and measurement state. The worker:

```python
state = reconstruct_measurement_state(official_main, payload["state"], replay_path)
module = import_frozen_main(payload["source"])
commands = [list(item) for item in module.agent(
    payload["round"], payload["seat"], state
)]
commands.append([8])
write_one_json_response({"commands": commands})
```

Run the worker twice in separate subprocesses with a deterministic environment,
5-second timeout, captured stdout/stderr, and no network or mutable historical
path. The parent records raw outputs and compares them before
canonicalization.

- [ ] **Step 5: Run worker and source tests**

```bash
.venv/bin/python -m pytest tests/generals/test_historical_policy.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit Task 6**

```bash
git add src/agentbench_frame/generals/historical_policy.py \
  src/agentbench_frame/generals/policy_probe_worker.py \
  tests/generals/test_historical_policy.py
git commit -m "feat(generals): probe immutable policy history"
```

---

### Task 7: Append-only historical KL pipeline and CLI

**Files:**
- Create: `src/agentbench_frame/generals/policy_kl_pipeline.py`
- Create: `tests/generals/test_policy_kl_pipeline.py`
- Create: `tests/generals/test_policy_kl_cli.py`
- Modify: `src/agentbench_frame/generals/cli.py`
- Modify: `src/agentbench_frame/tracking/quality.py`

**Interfaces:**
- Consumes: pilot manifest, reference manifest, assets root, data root, exact-count operational guards, optional failed run.
- Produces: `PolicyKLMeasurementResult`, `GeneralsPolicyKLPipeline.from_paths()`, `run()`, `recover()`, `measure-policy-kl`, `recover-policy-kl`.

- [ ] **Step 1: Write failing CLI and fake-pipeline tests**

CLI contract:

```python
def test_measure_policy_kl_cli_exposes_required_inputs(capsys):
    with pytest.raises(SystemExit):
        cli_main(["generals", "measure-policy-kl", "--help"])
    output = capsys.readouterr().out
    assert "--reference-manifest" in output
    assert "--data-dir" in output
    assert "--count-wall-time" in output
    assert "--count-max-states" in output


def test_recover_policy_kl_requires_failed_run(capsys):
    with pytest.raises(SystemExit):
        cli_main(["generals", "recover-policy-kl", "--help"])
    assert "--failed-run" in capsys.readouterr().out
```

Fake end-to-end contract:

```python
def test_pipeline_emits_complete_six_transition_measurement(fake_pipeline):
    result = fake_pipeline.run()
    assert result.status == "complete"
    assert len(result.summary["controlled_reference_policy_kl"]["transitions"]) == 6
    assert all(
        point["coverage"] == {"complete": 12, "total": 12}
        for point in result.summary["controlled_reference_policy_kl"]["transitions"]
    )
    assert result.summary["controlled_reference_policy_kl"]["primary_epsilon"] == "0.01"
```

Add failure tests for invalid collector match, missing selected state, incomplete
count, illegal historical macro, one missing per-state KL causing aggregate
`null`, failed-run recovery with same spec hashes, and attempts to recover a
complete run.

- [ ] **Step 2: Run and verify RED**

```bash
.venv/bin/python -m pytest \
  tests/generals/test_policy_kl_cli.py \
  tests/generals/test_policy_kl_pipeline.py -q
```

Expected: parser and module failures.

- [ ] **Step 3: Implement reference collection**

Start:

```python
run = Run.start(
    game="28_generals",
    agent="generals-policy-kl",
    run_type="measurement",
    data_dir=str(data_dir),
    config={
        "measurement_id": reference.measurement_id,
        "primary_epsilon": reference.primary_epsilon,
    },
)
```

Prepare two independent processes from the same frozen high-opponent source.
Run one self-play game per seed with `capture_measurement_states=True`. For
each seat, select only seat-local decisions 2 and 10. Copy each selected state
to `reference/states/<measurement_state_id>.json` and emit
`reference_state_selected`. Reject duplicate IDs, missing decisions, invalid
matches, wrong actor, or any state selected after historical probing begins.

- [ ] **Step 4: Implement counting, probing, and KL records**

For each selected state:

1. run or resume `ExactMacroCounter`;
2. save `action-space/counts/<state-id>.json`;
3. emit `action_space_count`;
4. probe v0-v6 twice;
5. canonicalize and verify each action;
6. save `policies/<version>/<state-id>.json`;
7. emit `historical_policy_action`; and
8. calculate six adjacent pairs at all four epsilon values.

Save per-state facts in `measurement/per-state-kl.jsonl` and emit one
`controlled_reference_policy_kl` event per version pair/state/epsilon. Use
`uniform_smoothed_deterministic_kl()` and persist:

```python
{
    "version_before": "v0",
    "version_after": "v1",
    "measurement_state_id": "...",
    "support_size": "123456789",
    "epsilon": "0.01",
    "kl_nats_decimal": "...",
    "kl_nats": 1.23,
    "actions_equal": False,
    "status": "complete",
    "action_space_spec_id": "...",
}
```

- [ ] **Step 5: Derive complete-or-null summaries and quality**

`measurement/transition-summary.json` and `summary.json` contain six ordered
transitions. For each epsilon, average exactly 12 `Decimal` values or return
`null` with the union of missing reasons. Include support-size statistics,
coverage, action disagreement, cache statistics, reference provenance, and
resource budgets.

Add these known event types:

```python
"reference_state_selected",
"action_space_count",
"historical_policy_action",
"controlled_reference_policy_kl",
```

Run `inspect_event_file()` before finalization and save `quality.json`.

- [ ] **Step 6: Add measurement and recovery CLI wiring**

Register:

```text
agentbench generals measure-policy-kl
agentbench generals recover-policy-kl
```

Both require `--agentbench-root`, `--manifest`, `--reference-manifest`, and
`--data-dir`. Normal measurement defaults to `--count-wall-time 3600` and
`--count-max-states 5000000`. Recovery additionally requires `--failed-run`,
reuses the same run directory and SQLite cache, verifies every frozen hash,
emits `pipeline_resumed`, and never rewrites finalized event lines.

- [ ] **Step 7: Run pipeline, CLI, and quality tests**

```bash
.venv/bin/python -m pytest \
  tests/generals/test_policy_kl_cli.py \
  tests/generals/test_policy_kl_pipeline.py \
  tests/test_tracking_contracts.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit Task 7**

```bash
git add src/agentbench_frame/generals/policy_kl_pipeline.py \
  src/agentbench_frame/generals/cli.py \
  src/agentbench_frame/tracking/quality.py \
  tests/generals/test_policy_kl_cli.py \
  tests/generals/test_policy_kl_pipeline.py
git commit -m "feat(generals): run historical policy KL measurement"
```

---

### Task 8: CI projection, curve, sensitivity, and missing states

**Files:**
- Modify: `src/agentbench_frame/report/builder.py`
- Modify: `src/agentbench_frame/report/templates/index.html`
- Modify: `tests/test_local_report_research.py`

**Interfaces:**
- Consumes: finalized `controlled_reference_policy_kl` events and summary.
- Produces: `controlled_reference_policy_kl_history`, SVG-ready complete segments, sensitivity table, coverage, and support-state rows.

- [ ] **Step 1: Write failing report projection tests**

```python
def test_report_projects_controlled_reference_kl_without_calling_it_ig(tmp_path):
    from agentbench_frame.report.builder import ReportBuilder

    events = [
        {
            "event_type": "controlled_reference_policy_kl",
            "status": "aggregate",
            "version_before": "v0",
            "version_after": "v1",
            "epsilon": "0.01",
            "mean_kl_nats": 2.5,
            "coverage": {"complete": 12, "total": 12},
        },
        {
            "event_type": "controlled_reference_policy_kl",
            "status": "aggregate",
            "version_before": "v1",
            "version_after": "v2",
            "epsilon": "0.01",
            "mean_kl_nats": None,
            "coverage": {"complete": 11, "total": 12},
        },
    ]
    research = ReportBuilder._derive_research(
        {
            "run_id": "measurement-1",
            "run_type": "measurement",
            "controlled_reference_policy_kl": {
                "primary_epsilon": "0.01",
                "transitions": [
                    {
                        "version_before": "v0",
                        "version_after": "v1",
                        "epsilon": "0.01",
                        "mean_kl_nats": 2.5,
                        "coverage": {"complete": 12, "total": 12},
                    },
                    {
                        "version_before": "v1",
                        "version_after": "v2",
                        "epsilon": "0.01",
                        "mean_kl_nats": None,
                        "coverage": {"complete": 11, "total": 12},
                    },
                ],
            },
        },
        events,
        "/missing/events.jsonl",
    )
    history = research["controlled_reference_policy_kl_history"]
    assert history[0]["mean_kl_nats"] == 2.5
    assert history[1]["mean_kl_nats"] is None
    assert research["ig_history"] == []
```

Render HTML and assert the exact labels:

```python
assert "Controlled-reference policy KL" in html
assert "epsilon = 0.01" in html
assert "early controlled real states" in html
assert "not epistemic information gain" in html
assert "11 / 12" in html
```

- [ ] **Step 2: Run and verify RED**

```bash
.venv/bin/python -m pytest \
  tests/test_local_report_research.py::test_report_projects_controlled_reference_kl_without_calling_it_ig -q
```

Expected: missing projection key or template label.

- [ ] **Step 3: Implement builder projection**

Parse the new events separately from `policy_kl_trace`. Build six ordered
points and split SVG polyline segments at each missing value; never bridge a
gap. Project:

```python
research["controlled_reference_policy_kl_history"] = transitions
research["controlled_reference_policy_kl_sensitivity"] = sensitivity
research["controlled_reference_support_states"] = support_rows
research["controlled_reference_policy_kl_svg_segments"] = segments
```

Do not write these values into `ig_history`.

- [ ] **Step 4: Render the controlled-reference section**

Add an inline SVG with one point per adjacent transition, breaking the line at
missing values, followed by:

- transition, mean nats/state, action disagreement, and coverage table;
- four-epsilon sensitivity table;
- measurement state, decimal support size, `log10 |A(s)|`, count status, and
  count time table; and
- the approved limitation text.

Use the existing `missing`, `amber`, `mint`, `panel`, and `data-table` classes;
do not introduce a new visual framework.

- [ ] **Step 5: Run the full report suite**

```bash
.venv/bin/python -m pytest tests/test_local_report_research.py -q
```

Expected: all report tests pass and old reports without new events still render.

- [ ] **Step 6: Commit Task 8**

```bash
git add src/agentbench_frame/report/builder.py \
  src/agentbench_frame/report/templates/index.html \
  tests/test_local_report_research.py
git commit -m "feat(report): show controlled policy KL curve"
```

---

### Task 9: Full verification, real historical run, audit, and result report

**Files:**
- Create after the run: `docs/experiments/2026-07-30-generals-controlled-policy-kl-result.md`
- Modify if execution exposes a contract defect: only the directly responsible files and their tests from Tasks 1-8.

**Interfaces:**
- Consumes: complete implementation, frozen assets, historical run directory, prepared strongest human.
- Produces: one immutable measurement run, quality audit, static CI report, and evidence-backed result document.

- [ ] **Step 1: Run all targeted tests from a clean process**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider \
  tests/test_measurement_contracts.py \
  tests/generals/test_assets.py \
  tests/generals/test_measurement_state.py \
  tests/generals/test_macro_action_space.py \
  tests/generals/test_macro_counter.py \
  tests/generals/test_historical_policy.py \
  tests/generals/test_policy_kl_cli.py \
  tests/generals/test_policy_kl_pipeline.py \
  tests/test_local_report_research.py -q
```

Expected: zero failures.

- [ ] **Step 2: Run the full Framework and Assets suites**

Framework:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -q
```

Assets:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/.venv/bin/python \
  -m pytest -p no:cacheprovider \
  /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/tests -q
```

Expected: zero failures; environment-dependent skips remain explicit.

- [ ] **Step 3: Run the real measurement**

From the Framework worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m agentbench_frame.cli \
  generals measure-policy-kl \
  --agentbench-root /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets \
  --manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml \
  --reference-manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/policy-kl-reference-v1.toml \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data \
  --count-wall-time 3600 \
  --count-max-states 5000000
```

If exact counting stops operationally, use `recover-policy-kl` with the saved
failed run and increased guard values. Do not change seeds, decisions,
canonicalization, or `spec_id`.

- [ ] **Step 4: Audit the real artifacts**

Verify:

```bash
.venv/bin/python -m agentbench_frame.cli data check \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data
```

Then inspect the new run's `quality.json`, `summary.json`,
`benchmark/action-space-spec.json`, `benchmark/reference-state-spec.json`,
12 count records, 84 policy records, 288 per-state epsilon records, event
counts, cache statistics, and original v0-v6 summary hashes. A complete
numerical claim requires 12 exact support counts, 84 legal deterministic
actions, 72 primary values, and six complete aggregates.

- [ ] **Step 5: Build and inspect the static report**

```bash
.venv/bin/python -m agentbench_frame.cli report \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data \
  --output-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_report
```

Confirm that the controlled-reference curve has no interpolated gap, the
sensitivity table uses four epsilon values, coverage is visible, support sizes
retain precision, and no label calls the metric epistemic information gain.

- [ ] **Step 6: Write the evidence-backed result report**

The report must contain these sections populated only from saved artifacts:

```text
Status and run identity
Frozen reference-state provenance
Action-space specification and exact-count completeness
Historical policy integrity and probe coverage
v0→v6 controlled-reference KL curve
Epsilon sensitivity
Resource and cache budget
Event and artifact quality
Scientific limitations
Reproduction commands
```

If any primary aggregate is missing, state the exact state/version/reason and
do not describe the curve as complete.

- [ ] **Step 7: Re-run verification after documentation**

```bash
git diff --check
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -q
```

Expected: clean diff check and zero test failures.

- [ ] **Step 8: Commit the audited result report**

```bash
git add docs/experiments/2026-07-30-generals-controlled-policy-kl-result.md
git commit -m "docs(generals): publish controlled policy KL result"
```

---

## Plan self-review checklist

- Spec coverage: Tasks 1-9 cover generic interfaces, complete state, all command families, exact support, immutable history, 12-state collection, strict KL, events, recovery, CI, real execution, and limitations.
- Placeholder scan: the plan contains no deferred implementation marker; every task names files, interfaces, failing tests, implementation behavior, verification commands, and commits.
- Type consistency: `PolicyKLReferenceConfig`, `HistoricalPolicyConfig`, `PrimitiveTransition`, `ExactCountResult`, `GeneralsMacroActionSpaceV1`, and `uniform_smoothed_deterministic_kl` retain the same names and roles throughout.
- Scope: the plan produces one coherent measurement subsystem and does not include v7 policy improvement or RL distribution modeling.
