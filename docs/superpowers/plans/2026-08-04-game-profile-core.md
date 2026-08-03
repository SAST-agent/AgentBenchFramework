# Game-Neutral HL Profile Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make HL orchestration load game semantics through a registered profile and consume generic match, metric, prompt, experience, and path contracts while preserving Rollman behavior through its own profile.

**Architecture:** Introduce small immutable contracts in `agentbench_frame.hl.game_profile` and `agentbench_frame.hl.match_record`, register profiles by `game_id`, and move Rollman terminology and runtime construction into `agentbench_frame.games.rollman.hl_profile`. The CLI resolves one profile and receives a complete `HLGameBindings` object; controller, selection, Experience, proposal packets, and reporting consume generic fields only.

**Tech Stack:** Python 3.11+, dataclasses, typing Protocol, PyYAML, pytest, existing AgentBench Framework event and version stores.

## Global Constraints

- Game backends, human submissions, SDKs, rules, seeds, role assignments, and timeouts are content-hashed run inputs.
- The generic HL package contains no Rollman, Ghost, Pacman, AntWar2, P0, or P1 field names.
- Provider session state is optional; every decisive prompt input remains persisted and replayable.
- Candidate source size and conditional count are not selection penalties.
- Faulted matches never count as wins or update Elo.
- A fixed replay cannot authorize promotion.
- Four candidate versions remain siblings from one explicit parent.
- Existing Rollman configurations and tests remain supported through the Rollman profile.

---

## File structure

### New core files

- `src/agentbench_frame/hl/match_record.py` — strict generic match and terminal metric schema.
- `src/agentbench_frame/hl/game_profile.py` — profile, prompt, runtime binding, smoke, replay, and behavior contracts plus registry.
- `tests/hl/test_match_record.py` — generic record validation and comparison tests.
- `tests/hl/test_game_profile.py` — registry, fake-profile, and binding tests.

### Rollman profile files

- `src/agentbench_frame/games/rollman/hl_profile.py` — Rollman prompt terminology, path requirements, evaluator and smoke bindings.
- `tests/rollman/test_hl_profile.py` — Rollman profile compatibility tests.

### Core files to change

- `src/agentbench_frame/hl/local_config.py` — game-defined local path mapping.
- `src/agentbench_frame/hl/selection.py` — generic points, dense margin, role, and opponent comparisons.
- `src/agentbench_frame/hl/experience_ledger.py` — generic score and role evidence.
- `src/agentbench_frame/hl/proposal.py` — game-neutral packet fields.
- `src/agentbench_frame/hl/context.py` — profile-rendered prompts without Rollman nouns.
- `src/agentbench_frame/hl/controller.py` — generic selection and activation facts.
- `src/agentbench_frame/hl/report.py` — generic Elo and score series with configured display names.
- `src/agentbench_frame/hl/cli.py` — resolve profile and consume runtime bindings.
- `src/agentbench_frame/games/rollman/__init__.py` — register Rollman profile.
- `tests/hl/test_local_config.py`, `tests/hl/test_selection.py`, `tests/hl/test_experience_ledger.py`, `tests/hl/test_context.py`, `tests/hl/test_proposal.py`, `tests/hl/test_cli.py`, `tests/hl/test_report.py` — migrated contracts and regression coverage.

---

### Task 1: Strict generic match records

**Files:**
- Create: `src/agentbench_frame/hl/match_record.py`
- Create: `tests/hl/test_match_record.py`

**Interfaces:**
- Produces: `MatchRecord`, `MatchRecord.from_mapping()`, `MatchRecord.to_dict()`, `completed_match_records()`.
- Consumed by: selection, Experience ledger, report, and all game evaluators.

- [ ] **Step 1: Write failing schema tests**

```python
def test_complete_match_requires_generic_scores_and_dense_margin():
    from agentbench_frame.hl.match_record import MatchRecord

    match = MatchRecord.from_mapping({
        "schema_version": "1.0", "game": "fake", "candidate": "v1",
        "opponent": "human-1", "candidate_role": "north", "seed": 7,
        "status": "complete", "result": "win", "points": 1.0,
        "candidate_score": 3.0, "opponent_score": 1.0,
        "dense_margin": 2.0, "terminal_metrics": {"camp_hp": 3.0},
        "rounds": 12, "replay": "r.json", "trace": "t.jsonl",
        "faults": [], "live_opponent": True,
    })
    assert match.comparison_key == ("human-1", "north", 7)
    assert match.to_dict()["dense_margin"] == 2.0

def test_faulted_or_replay_only_match_is_not_promotable():
    from agentbench_frame.hl.match_record import MatchRecord

    faulted = MatchRecord.from_mapping(_valid(status="failed", result=None,
                                               points=None, faults=["RE"]))
    replay = MatchRecord.from_mapping(_valid(live_opponent=False))
    assert faulted.promotable is False
    assert replay.promotable is False
```

- [ ] **Step 2: Run tests and confirm missing module failure**

Run: `pytest -q tests/hl/test_match_record.py`

Expected: FAIL with `ModuleNotFoundError: agentbench_frame.hl.match_record`.

- [ ] **Step 3: Implement the immutable schema**

```python
@dataclasses.dataclass(frozen=True)
class MatchRecord:
    schema_version: str
    game: str
    candidate: str
    opponent: str
    candidate_role: str
    seed: int
    status: Literal["complete", "failed", "timeout", "incomplete"]
    result: Literal["win", "draw", "loss"] | None
    points: float | None
    candidate_score: float | None
    opponent_score: float | None
    dense_margin: float | None
    terminal_metrics: Mapping[str, float]
    rounds: int | None
    replay: str | None
    trace: str | None
    faults: tuple[str, ...]
    live_opponent: bool

    @property
    def comparison_key(self) -> tuple[str, str, int]:
        return self.opponent, self.candidate_role, self.seed

    @property
    def promotable(self) -> bool:
        return self.status == "complete" and not self.faults and self.live_opponent
```

Validation rejects unknown fields, non-finite numerics, points outside `[0,1]`, complete records without result/scores/margin, non-complete records with strategy points, and a result inconsistent with points.

- [ ] **Step 4: Run the focused tests**

Run: `pytest -q tests/hl/test_match_record.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/hl/match_record.py tests/hl/test_match_record.py
git commit -m "feat: add generic HL match records"
```

### Task 2: Game profile and registry contracts

**Files:**
- Create: `src/agentbench_frame/hl/game_profile.py`
- Create: `tests/hl/test_game_profile.py`
- Modify: `src/agentbench_frame/hl/__init__.py`

**Interfaces:**
- Consumes: `CandidateEvaluator`, `MatchRecord`, `ContextBundle`, `Version`.
- Produces: `PromptProfile`, `SmokeResult`, `BehaviorComparison`, `MetricSchema`, `HLGameBindings`, `GameProfile`, `register_game_profile()`, `get_game_profile()`.

- [ ] **Step 1: Write registry and fake-profile tests**

```python
def test_registry_resolves_exact_game_id_and_rejects_duplicates():
    from agentbench_frame.hl.game_profile import (
        get_game_profile, register_game_profile, reset_game_profiles_for_testing,
    )
    reset_game_profiles_for_testing()
    profile = FakeProfile(game_id="fake")
    register_game_profile(profile)
    assert get_game_profile("fake") is profile
    with pytest.raises(ValueError, match="already registered"):
        register_game_profile(profile)

def test_prompt_profile_has_no_game_specific_required_field_names():
    profile = PromptProfile(
        candidate_label="agent", opponent_label="opponent",
        roles=("north", "south"), policy_input="PublicState",
        output_contract="list[AtomicOperation]",
        planner_diversity=("distinct mechanism", "distinct falsifier"),
        prohibited_information=("opponent source", "seed lookup"),
    )
    assert profile.roles == ("north", "south")
```

- [ ] **Step 2: Run tests and confirm missing contracts**

Run: `pytest -q tests/hl/test_game_profile.py`

Expected: FAIL because `game_profile` is absent.

- [ ] **Step 3: Implement frozen contracts and registry**

```python
@dataclasses.dataclass(frozen=True)
class PromptProfile:
    candidate_label: str
    opponent_label: str
    roles: tuple[str, ...]
    policy_input: str
    output_contract: str
    planner_diversity: tuple[str, ...]
    prohibited_information: tuple[str, ...]

@dataclasses.dataclass(frozen=True)
class MetricSchema:
    points_label: str
    dense_margin_label: str
    elo_label: str
    role_labels: tuple[str, ...]

@runtime_checkable
class GameProfile(Protocol):
    game_id: str
    required_local_paths: tuple[str, ...]
    def prompt_profile(self) -> PromptProfile: ...
    def build_bindings(self, *, config: LocalHLConfig,
                       run_root: Path) -> HLGameBindings: ...
```

Registry lookup imports built-in profiles lazily and produces an actionable
`KeyError` listing registered IDs.

- [ ] **Step 4: Run the focused tests**

Run: `pytest -q tests/hl/test_game_profile.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/hl/game_profile.py src/agentbench_frame/hl/__init__.py tests/hl/test_game_profile.py
git commit -m "feat: define registered HL game profiles"
```

### Task 3: Game-defined local path mappings

**Files:**
- Modify: `src/agentbench_frame/hl/local_config.py`
- Modify: `tests/hl/test_local_config.py`

**Interfaces:**
- Consumes: `get_game_profile(run.game).required_local_paths`.
- Produces: `LocalPaths.values`, `LocalPaths.require(name)`, compatibility attribute access during Rollman migration.

- [ ] **Step 1: Write failing arbitrary-profile path test**

```python
def test_local_config_validates_paths_declared_by_profile(tmp_path, monkeypatch):
    register_game_profile(FakeProfile(
        game_id="fake", required_local_paths=("backend", "human_pool", "workspace", "runs_root")
    ))
    config = LocalHLConfig.load(write_config(tmp_path, game="fake", paths={
        "backend": "backend", "human_pool": "humans.json",
        "workspace": "candidate", "runs_root": "runs",
    }))
    assert config.paths.require("backend") == (tmp_path / "backend").resolve()

def test_missing_profile_path_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing paths fields:.*human_pool"):
        LocalHLConfig.load(write_config(tmp_path, game="fake", paths={
            "backend": "backend", "workspace": "candidate", "runs_root": "runs",
        }))
```

- [ ] **Step 2: Run tests and confirm the Rollman-only rejection**

Run: `pytest -q tests/hl/test_local_config.py`

Expected: FAIL with `this local harness currently supports 29_rollman`.

- [ ] **Step 3: Implement mapping-backed paths**

```python
@dataclasses.dataclass(frozen=True)
class LocalPaths:
    values: Mapping[str, Path]

    def require(self, name: str) -> Path:
        try:
            return self.values[name]
        except KeyError as exc:
            raise KeyError(f"local path is not configured: {name}") from exc

    def __getattr__(self, name: str) -> Path:
        return self.require(name)
```

`LocalHLConfig.load()` resolves the profile before paths, accepts exactly its
declared path names, expands environment variables, and preserves repository-
root source-run resolution.

- [ ] **Step 4: Run path and configuration tests**

Run: `pytest -q tests/hl/test_local_config.py tests/hl/test_config.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/hl/local_config.py tests/hl/test_local_config.py
git commit -m "refactor: let HL profiles declare local paths"
```

### Task 4: Generic selection and Experience evidence

**Files:**
- Modify: `src/agentbench_frame/hl/selection.py`
- Modify: `src/agentbench_frame/hl/experience_ledger.py`
- Modify: `tests/hl/test_selection.py`
- Modify: `tests/hl/test_experience_ledger.py`

**Interfaces:**
- Consumes: `MatchRecord` or strict mappings accepted by `MatchRecord.from_mapping()`.
- Produces: `CandidateDiagnostics` with generic `mean_dense_margin`, `worst_dense_margin`, `role_points`, and generic Experience comparisons.

- [ ] **Step 1: Write failing role-sensitive generic selection tests**

```python
def test_candidate_cannot_advance_by_regressing_a_locked_role():
    parent = diagnostics("v0", [match("north", "win", 2), match("south", "win", 1)])
    candidate = diagnostics("v1", [match("north", "win", 5), match("south", "loss", -2)])
    assert select_linear_successor(parent, (candidate,)).search_parent_version_id == "v0"

def test_experience_comparison_keys_include_role():
    record = derive_experience_record(
        iteration_id="iter-000001", act_id="act-000001-b00", branch_index=0,
        parent_version_id="v0", candidate_version_id="v1", selected=True,
        brief={
            "activation_condition": "public threat is visible",
            "mechanism": "route response",
            "preservation_contract": "retain the parent outside the threat",
        },
        parent_matches=(match("north"), match("south")),
        candidate_matches=(match("north"), match("south")),
        activation={"status": "complete", "decision_count": 10,
                    "changed_action_count": 2},
    )
    assert {(row.opponent, row.candidate_role, row.seed) for row in record.comparisons} == {
        ("human", "north", 7), ("human", "south", 7)
    }
```

- [ ] **Step 2: Run focused tests and observe missing generic fields**

Run: `pytest -q tests/hl/test_selection.py tests/hl/test_experience_ledger.py`

Expected: FAIL on `candidate_score`, `dense_margin`, and `candidate_role`.

- [ ] **Step 3: Implement generic diagnostics and evidence joins**

`CandidateDiagnostics.from_matches()` parses each record, groups points and
dense margins by opponent and role, and orders candidates by:

```python
return (
    self.points,
    self.worst_role_points,
    self.worst_dense_margin,
    self.mean_dense_margin,
    self.behavioral_novelty,
    float(-self.branch_index),
)
```

`ExperienceComparison` stores `candidate_role`, parent/candidate points,
parent/candidate dense margins, and `dense_margin_delta`. Its identity join is
`(opponent, candidate_role, seed)`.

- [ ] **Step 4: Run selection, controller, and Experience tests**

Run: `pytest -q tests/hl/test_selection.py tests/hl/test_experience_ledger.py tests/hl/test_controller.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/hl/selection.py src/agentbench_frame/hl/experience_ledger.py tests/hl/test_selection.py tests/hl/test_experience_ledger.py
git commit -m "refactor: make HL selection and experience game neutral"
```

### Task 5: Game-neutral proposal and prompt packets

**Files:**
- Modify: `src/agentbench_frame/hl/proposal.py`
- Modify: `src/agentbench_frame/hl/context.py`
- Modify: `tests/hl/test_proposal.py`
- Modify: `tests/hl/test_context.py`

**Interfaces:**
- Consumes: `PromptProfile`, generic `MatchRecord` mappings, game digest, Experience Skill, research state, candidate code index.
- Produces: generic planner/candidate packets and prompts.

- [ ] **Step 1: Write failing noun-isolation tests**

```python
def test_fake_profile_prompt_contains_profile_terms_and_no_rollman_terms(tmp_path):
    prompt = context.build_planner_prompt(
        act_id="act-000001-planner", iteration_id="iter-000001",
        parent_version_id="v0", workspace=tmp_path / "candidate",
        game_digest_path=write_json(tmp_path / "digest.json", {}),
        research_state_path=write_json(tmp_path / "research.json", {}),
        replay_evidence=[], previous_measurements={}, active_target=None,
        scope_contract_required=True, planner_input_path=None,
        prompt_profile=FAKE_PROMPT_PROFILE,
    )
    assert "north" in prompt
    for forbidden in ("Rollman", "Ghost", "pacman_pos", "rank15", "rank16"):
        assert forbidden not in prompt

def test_planner_packet_uses_generic_match_fields(tmp_path):
    value = json.loads(write_planner_input_packet(
        output_path=tmp_path / "planner.json", iteration_id="iter-000001",
        parent_version_id="v0",
        game_digest_path=write_json(tmp_path / "digest.json", {}),
        context_manifest_path=write_json(tmp_path / "manifest.json", {}),
        research_state_path=write_json(tmp_path / "research.json", {}),
        replay_evidence=[generic_match_evidence()], previous_measurements={},
        active_target=None,
        candidate_source_path=write_text(tmp_path / "ai.py", "def ai_func(state): return 0\n"),
    ).read_text())
    assert value["replay_evidence"][0]["dense_margin"] == 2.0
    assert "rollman_score" not in value["replay_evidence"][0]
```

- [ ] **Step 2: Run tests and confirm Rollman nouns leak**

Run: `pytest -q tests/hl/test_context.py tests/hl/test_proposal.py`

Expected: FAIL because prompts and packet evidence fields are Rollman-specific.

- [ ] **Step 3: Render prompt templates from PromptProfile**

Add `prompt_profile: PromptProfile` to iteration context methods. Render role,
input, output, diversity, and prohibition clauses from that object. Preserve
all isolation, bounded-evidence, checkpoint-first, access audit, no-grid-search,
code-growth, compile, smoke, and Experience requirements.

Proposal evidence fields become:

```python
EVIDENCE_FIELDS = (
    "opponent", "candidate_role", "seed", "status", "result", "phase",
    "points", "candidate_score", "opponent_score", "dense_margin",
)
```

- [ ] **Step 4: Run context and proposal tests**

Run: `pytest -q tests/hl/test_context.py tests/hl/test_proposal.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/hl/context.py src/agentbench_frame/hl/proposal.py tests/hl/test_context.py tests/hl/test_proposal.py
git commit -m "refactor: render HL prompts from game profiles"
```

### Task 6: Rollman profile and runtime bindings

**Files:**
- Create: `src/agentbench_frame/games/rollman/hl_profile.py`
- Create: `tests/rollman/test_hl_profile.py`
- Modify: `src/agentbench_frame/games/rollman/__init__.py`
- Modify: `src/agentbench_frame/hl/cli.py`

**Interfaces:**
- Consumes: `LocalHLConfig`, `HLGameBindings`, Rollman evaluator, smoke fixture, replay tools, measurement functions.
- Produces: registered `RollmanHLProfile(game_id="29_rollman")` and bindings consumed by the generic CLI.

- [ ] **Step 1: Write failing Rollman profile tests**

```python
def test_rollman_profile_declares_all_existing_paths_and_terms():
    profile = get_game_profile("29_rollman")
    assert profile.required_local_paths == (
        "agentbench_root", "official_logic_root", "pacman_sdk_root",
        "human_manifest", "workspace", "runs_root", "opponent_build_root",
    )
    assert profile.prompt_profile().roles == ("rollman",)

def test_rollman_validate_resolves_profile(capsys):
    assert main(["validate", "--config", str(CONFIG)]) == 0
    assert json.loads(capsys.readouterr().out)["game"] == "29_rollman"
```

- [ ] **Step 2: Run tests and confirm profile absence**

Run: `pytest -q tests/rollman/test_hl_profile.py tests/hl/test_cli.py::test_hl_validate_reports_open_ended_k1_rollback_defaults`

Expected: FAIL because the built-in Rollman profile is not registered.

- [ ] **Step 3: Move Rollman bindings behind the profile**

`RollmanHLProfile` supplies the existing path contract, prompt vocabulary,
candidate smoke verifier, human-pool loader, evaluator factory, replay summary
resolver, policy measurement hooks, source audit, and display metric labels.

`hl.cli` performs:

```python
config = LocalHLConfig.load(args.config)
profile = get_game_profile(config.run.game)
bindings = profile.build_bindings(config=config, run_root=run_root)
```

All later orchestration uses `bindings` callables and paths. Rollman-only imports
leave module-level `hl.cli` and live inside `games.rollman.hl_profile` or lazy
Rollman factory functions.

- [ ] **Step 4: Run Rollman CLI and end-to-end fake tests**

Run: `pytest -q tests/rollman/test_hl_profile.py tests/hl/test_cli.py tests/hl/test_e2e_fake.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/games/rollman/hl_profile.py src/agentbench_frame/games/rollman/__init__.py src/agentbench_frame/hl/cli.py tests/rollman/test_hl_profile.py tests/hl/test_cli.py
git commit -m "refactor: route Rollman HL through a game profile"
```

### Task 7: Generic reporting and full regression gate

**Files:**
- Modify: `src/agentbench_frame/hl/report.py`
- Modify: `tests/hl/test_report.py`
- Modify: `tests/test_research_boundaries.py`
- Modify: `docs/hl/README.md`

**Interfaces:**
- Consumes: generic match records, `MetricSchema`, event records.
- Produces: three-panel IG/Elo/win-rate plots with integer iteration and optional role series.

- [ ] **Step 1: Write failing generic report tests**

```python
def test_report_uses_metric_schema_and_integer_iterations(tmp_path):
    outputs = write_hl_report(events(), tmp_path, metric_schema=MetricSchema(
        points_label="Win rate", dense_margin_label="Camp HP margin",
        elo_label="Agent Elo", role_labels=("P0", "P1"),
    ))
    rows = list(csv.DictReader(outputs["curves_csv"].open()))
    assert [int(row["iteration"]) for row in rows] == list(range(len(rows)))
    svg = outputs["curves_svg"].read_text()
    assert "Agent Elo vs HL Iteration" in svg
    assert "rollman_elo" not in rows[0]
```

- [ ] **Step 2: Run report tests and observe Rollman-only keys**

Run: `pytest -q tests/hl/test_report.py tests/test_research_boundaries.py`

Expected: FAIL on `rollman_elo` and Rollman score assumptions.

- [ ] **Step 3: Implement configured generic report names**

Curve rows expose `population_elo`, `win_rate`, `mean_local_policy_kl`, and
optional `role_win_rates`. Three plotted panels remain IG, population Elo, and
live win rate. No-promotion iterations inherit parent values and retain their
integer iteration point.

- [ ] **Step 4: Run the complete test suite**

Run: `pytest -q`

Expected: PASS with no Rollman regression and no generic-core game nouns.

- [ ] **Step 5: Run the generic boundary scan**

Run:

```bash
rg -n "Rollman|Ghost|pacman|rollman_score|ghosts_score" \
  src/agentbench_frame/hl
```

Expected: no game-semantic occurrence outside explicitly documented migration
compatibility aliases; tests assert the allowlist is empty for runtime logic.

- [ ] **Step 6: Commit**

```bash
git add src/agentbench_frame/hl/report.py tests/hl/test_report.py tests/test_research_boundaries.py docs/hl/README.md
git commit -m "refactor: finish game-neutral HL core"
```

## Core-plan completion gate

The core plan is complete only when:

- a fake second profile validates and builds bindings through the same CLI;
- Rollman configuration, audit, resume, k=4 proposal, measurement, and reports
  pass their existing tests through `RollmanHLProfile`;
- generic HL source contains no game-semantic score, role, state, or opponent
  field names;
- full `pytest -q` passes;
- the worktree is clean and every task has an independently reviewable commit.

The next implementation plan adds the AntWar2 profile, atomic decision-space
implementation, Replay Skill, live evaluator, behavior KL, and historical
oracle certification. The experiment plan then runs blind v22 and model-
bootstrap k=4 phases.
