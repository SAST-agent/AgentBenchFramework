# Unified HL Framework and AntWar2 Positive-Control Design

## Objective

Build one reproducible human-level iteration framework whose game-specific
inputs are limited to frozen rules, an atomic decision-space definition, a
replay-reading Skill, the candidate interface, and evaluator wiring. Validate
the framework on AntWar2 with two positive-control phases:

1. blind learning from the weak historical `ifelse_v22` policy;
2. model-bootstrap learning from scratch under the same protocol.

The experiment continues toward dominance over every runnable frozen human
submission. The historical `ifelse_v239` policy is an evaluation-only oracle
and is never exposed to planners, candidates, reducers, repair acts, Experience
Skill inputs, or replay evidence.

## Scientific invariants

- The game backend, human submissions, rules, SDK, seeds, timeouts, and role
  assignments are content-hashed before a run.
- Candidate policies may use observable-state conditionals, finite memory,
  state machines, graph algorithms, path prediction, and other interpretable
  code.
- Strategy size and the number of conditional branches are not optimization
  penalties.
- Candidate policies may not read opponent source, branch on opponent identity,
  branch on seed, memorize replay identifiers, or access hidden engine state.
- Fixed replay execution is a diagnostic screen only. Promotion requires a
  live match against the unchanged opponent package.
- Four candidates in a proposal cycle are siblings from one explicit parent;
  the lineage is not a four-ary tree.
- Framework-measured match and activation facts are authoritative. Model-authored
  summaries cannot overwrite them.
- Failure, invalidity, and negative causal findings remain durable research
  evidence.

## Architecture

The HL system consists of a game-neutral orchestration layer and a registered
game profile.

```text
HL controller
  -> GameProfile
       -> context assets
       -> candidate scaffold and smoke contract
       -> live match evaluator
       -> replay indexer and evidence builder
       -> policy atomizer and behavior comparison
       -> game metric adapter
  -> provider
  -> lineage and specialist archive
  -> experience ledger and Skill projection
  -> measurement and reporting
```

The controller owns API invocation, recovery, version snapshots, k-candidate
orchestration, logs, budgets, rollback, selection, Experience updates, and
curve production. A game profile owns only game semantics.

### GameProfile contract

Each profile implements the following capabilities:

```python
class GameProfile(Protocol):
    game_id: str

    def build_context_bundle(self, run_root: Path) -> ContextBundle: ...
    def build_game_digest(self, bundle: ContextBundle) -> Mapping[str, Any]: ...
    def seed_candidate(self, destination: Path, origin: OriginSpec) -> None: ...
    def build_prompt_profile(self) -> PromptProfile: ...
    def verify_candidate(self, candidate: Path) -> SmokeResult: ...
    def build_evaluator(self, config: EvaluationConfig) -> CandidateEvaluator: ...
    def build_replay_evidence(self, matches: Sequence[MatchRecord]) -> ReplayEvidence: ...
    def compare_behavior(self, parent: Path, candidate: Path,
                         cases: Sequence[BehaviorCase]) -> BehaviorComparison: ...
    def metric_schema(self) -> MetricSchema: ...
```

Profiles are loaded from a registry by `game_id`. The CLI contains no
game-name conditional that selects Rollman-specific fields.

### PromptProfile contract

Game-specific terminology and candidate instructions are data, not controller
branches. A prompt profile supplies:

- role names and role-observation semantics;
- public policy input interface;
- legal output format;
- compile and smoke commands;
- replay inspection commands;
- candidate access allowlist;
- branch-diversity requirements;
- evidence field descriptions;
- game-specific prohibited information.

Planner branch roles are generated from replay evidence. They are not fixed to
named opponents or manually invented tactics. Four briefs must have distinct
mechanisms and falsifiers; threshold-only variants of one mechanism are
rejected before API candidate acts begin.

## Context package and token economy

Every run snapshots a content-addressed static package:

```text
context/
  manifest.json
  rules.md
  decision_space.yaml
  sdk_interface.md
  game_digest.json
  replay-skill/
  backend_manifest.json
  human_pool_manifest.json
```

`manifest.json` records SHA-256 hashes and source provenance. `game_digest.json`
is generated deterministically from the human-authored inputs and contains no
model-invented tactical categories.

The bootstrap act reads the complete static package. Proposal cycles receive
incremental packets containing:

- the selected parent and its relevant dependency closure;
- the compact game digest;
- the active Experience Skill projection;
- new bounded replay summaries and authorized causal windows;
- parent measurements and active research questions;
- one evidence-grounded branch brief;
- an exact smoke contract.

Long rules are not copied into every prompt. A candidate may inspect a precise
authoritative section when its change depends on that rule. Provider session
resumption is an optimization, not a reproducibility dependency: every
decisive input, raw response, tool audit, and output artifact is saved under
the run directory.

The planner may retain a continuous run-level research session. The four
sibling candidates receive identical frozen evidence packets plus distinct
branch briefs, preventing cross-branch contamination. A selected branch may
provide the lineage session for the following cycle. A run can be replayed
from files when provider session recovery is unavailable.

## Generic match record

All evaluators emit one common record:

```json
{
  "schema_version": "1.0",
  "game": "30_antwar2",
  "candidate": "v000123",
  "opponent": "rank01",
  "candidate_role": "P1",
  "seed": 1,
  "status": "complete",
  "result": "win",
  "points": 1.0,
  "candidate_score": 1.0,
  "opponent_score": 0.0,
  "dense_margin": 3.0,
  "terminal_metrics": {},
  "rounds": 418,
  "replay": "...",
  "trace": "...",
  "faults": []
}
```

`points` is win/draw/loss utility. `dense_margin` is a game-defined monotone
comparison used only after valid match points. AntWar2 defines it from public
terminal camp HP, successful breaches, and other frozen public measurements.
Infrastructure faults, missing entry points, crashes, protocol errors, and
timeouts are represented explicitly and never counted as strategy wins.

## AntWar2 profile

### Frozen backend and policy interface

The profile uses the official C++ backend and Python SDK line protocol. The
backend receives length-prefixed JSON from the local judger, while AI packages
receive public SDK state and submit operation bundles. Both P0 and P1 are
candidate roles.

The profile records:

- backend source hash and compiled binary hash;
- compiler identity and flags;
- SDK tree hash;
- each human package tree hash;
- match transport and timeout configuration;
- Python executable and dependency environment.

No rule or legality change is allowed to obtain platform compatibility.

### Atomic decision space

The manually auditable decision space contains protocol-level operations only:

```yaml
atomic_operations:
  - HOLD
  - BUILD_TOWER(x, y)
  - UPGRADE_TOWER(tower_id, target_type)
  - DOWNGRADE_TOWER(tower_id)
  - USE_LIGHTNING_STORM(target_cell)
  - USE_EMP_BLASTER(target_cell)
  - USE_DEFLECTOR(target_cell)
  - USE_EMERGENCY_EVASION(target_cell)
  - UPGRADE_GENERATION_SPEED
  - UPGRADE_GENERATED_ANT
```

This list contains no tactical hypothesis, strategy label, inferred objective,
or opponent category. The legal support at state `s` is derived exclusively
from the frozen SDK legality predicate.

An emitted bundle is atomized in accepted execution order. For atom `j`, the
support is computed using the public state plus atoms `0..j-1` as pending
operations. An empty accepted sequence contributes `HOLD`. Illegal proposed
operations are logged but do not enter the accepted behavioral sequence.

### Deterministic-policy information gain

Candidate policies normally choose deterministic actions. For a legal support
of size `n`, epsilon smoothing assigns `1-epsilon` plus the uniform residual to
the selected atom and distributes the residual across legal alternatives. The
same transformation is applied to the origin policy. KL is evaluated on a
frozen state-occupancy sample and averaged over valid atomic decisions.

The reported value is behavioral divergence from the phase origin. It does not
claim to measure the language model's epistemic uncertainty. Occupancy shift is
reported separately when enabled.

### Replay Skill

The AntWar2 Replay Skill generates three bounded layers:

1. `match_summary.json`: role, opponent, seed, validity, winner, rounds,
   terminal camps, breaches, and return codes.
2. `event_index.jsonl`: accepted operations and public state events, including
   tower construction, upgrade, downgrade, weapon use, ant death, breach,
   camp damage, production, and material coin changes.
3. `causal_windows/*.json`: bounded windows around indexed events with public
   before/action/immediate-after/delayed-after fields.

Natural-language summaries translate numeric protocol fields without assigning
unobservable intent. Each learning claim follows this structure:

```text
public condition
  -> opponent accepted operation
  -> immediate board effect
  -> delayed economy, route, production, or camp effect
  -> candidate response and falsifier
```

Replay-only wins are never promotable. Every hypothesis must be rerun against
the live unchanged opponent because that opponent may react to the modified
public state.

### Candidate smoke and activation measurement

Smoke verification checks:

- import and protocol startup;
- both player roles;
- legal empty and non-empty bundles;
- tower ID refresh and delta-state handling;
- operation serialization;
- the branch activation condition through the public policy entry point;
- at least one preservation case outside the activation condition.

Behavior comparison records accepted atomic sequences for parent and candidate
on frozen public states. A candidate with no changed accepted atom on its
claimed activation cases is invalid. Large policy changes are allowed and are
not penalized.

## Replay-grounded Experience Skill

`experience/ledger.jsonl` is an append-only framework-owned evidence store.
Every branch representative records:

```yaml
condition: observable public predicate
public_evidence: exact replay window references
opponent_operation: accepted atomic operation sequence
immediate_effect: public board effect
delayed_effect: public economy, route, production, or camp effect
candidate_response: implemented mechanism
role: P0 | P1
opponent: package identifier
seed: integer
parent_result: win | draw | loss
candidate_result: win | draw | loss
verdict: verified_good | verified_bad | mixed | inconclusive | invalid
failure_boundary: scope in which the claim does not hold
selected: boolean
```

`experience/SKILL.md` is a deterministic bounded projection containing:

1. verified effective conditions and mechanisms;
2. verified ineffective conditions and mechanisms;
3. mixed or role-sensitive findings;
4. exact replay-grounded observations;
5. active failure cases and research questions.

Candidate notes are provisional until joined with Framework evaluation facts.
Compression may deduplicate equivalent findings but must retain condition,
mechanism, role, verdict, measured effect, failure boundary, and evidence keys.
Every planner and candidate reads the projection.

## k=4 proposal cycle

Each proposal cycle performs:

1. Select one explicit search parent.
2. Build live failure evidence for the active opponent, role, and seed cases.
3. Ask one planner for four causal, mechanically distinct, falsifiable briefs.
4. Run four sibling coding acts from the same parent.
5. Reject invalid, zero-activation, or access-violating candidates.
6. Run a common live quick screen on the active failure and preservation cases.
7. Run shared validation for the best candidates.
8. Select the next frontier parent and retain other useful role specialists.
9. Append all measured branch outcomes to the Experience ledger.
10. Regenerate the compact Skill and research state.

Candidate code may grow substantially. A change is constrained by public
observability and empirical falsifiability, not by edit size. Full policy code
or a dependency-complete representation is available when a mechanism crosses
economic, production, targeting, and role branches; arbitrary narrow slices
cannot hide required dependencies.

## Selection, specialists, and rollback

The archive retains:

- the certified champion;
- the active search frontier;
- the best P0 specialist;
- the best P1 specialist;
- per-active-target specialists with exact evidence coverage.

Selection is lexicographic:

1. valid fault-free execution;
2. points on the active failure cases;
3. preservation of locked wins;
4. worst-role and worst-seed points;
5. aggregate points;
6. worst and mean dense margin;
7. behavioral novelty as a final research tie-break only.

A specialist may remain archived without becoming the common parent. When P0
and P1 leaders differ, the reducer may create a role-dispatched synthesis
candidate and must validate the merged policy on the full locked matrix.

Rollback targets the nearest retained checkpoint that satisfies the relevant
locked cases. It never defaults mechanically to the phase origin. Sustained
degradation can recover the champion, a frontier ancestor, or a verified role
specialist according to the failed coverage set.

## Positive-control protocol

### Phase 0: backend and historical oracle certification

1. Build the unmodified frozen backend and record all hashes.
2. Run strong-versus-strong reference matches to validate transport, both
   roles, replay output, terminal interpretation, and deterministic seeds.
3. Run `ifelse_v239` against every runnable human package in both roles on
   seed 7.
4. Run rank01 in both roles on seeds `1,2,3,5,7,11,19,43`.
5. Run the historical rank02 and rank05 matrices.
6. Classify every excluded package by reproducible packaging or runtime fault.

The certification report distinguishes completed strategy games from package
failures. The phrase “all human players” is reserved for all runnable frozen
submissions that complete valid matches.

### Phase 1: blind `ifelse_v22` positive control

The candidate environment contains `ifelse_v22`, frozen rules, SDK, decision
space, Replay Skill, and evidence produced by its own matches. It excludes:

- `ifelse_v23` through `ifelse_v239` source;
- later historical Skills and reports;
- human source code;
- oracle replay summaries that were not produced by the positive-control run.

Milestones are cumulative:

1. at least three of four candidates per rolling three-cycle window reach live
   evaluation;
2. at least one replay-derived mechanism activates and improves a comparable
   live match;
3. both roles improve over v22 on held-out seeds;
4. the run reaches the rank01/rank02/rank05 validation band;
5. the run challenges the complete runnable human pool.

Historical version numbers and mechanisms are not success inputs. They are
used only after measurement to compare learning efficiency and achieved
strength.

### Phase 2: model-bootstrap from scratch

The model creates an interpretable, legal origin from the frozen static
package. The complete Phase 1 harness, evaluator, Experience mechanism, and
selection protocol are reused without access to historical candidate code.
This phase demonstrates end-to-end reproducibility rather than continuation
from a human-curated weak policy.

## Stagnation control and API efficiency

The run has no fixed iteration ceiling, but paid exploration is evidence-gated.

- After three proposal cycles without frontier improvement, the Framework runs
  a stagnation audit before another paid cycle. The audit checks candidate
  validity, activation, replay evidence quality, selection comparability,
  role imbalance, and repeated hypotheses.
- The audit may change the evidence target, select a verified specialist or
  frontier checkpoint, request role synthesis, or increase replay coverage.
- It may change `k` through configuration for an ablation, but the default
  positive-control run remains `k=4`.
- After six cycles without any points or dense-margin improvement, candidate
  API calls pause until the diagnostic failure is resolved. This is a harness
  safety condition, not an experimental success condition.
- Transport and budget failures resume from persisted checkpoints without
  consuming an iteration index as a scientific rollout.

All provider usage is attributed by phase, act, branch, candidate, input,
cached/prefill tokens when available, output tokens, retries, and outcome.

## Measurement and reporting

The primary report contains three panels with integer proposal iteration on
the horizontal axis:

1. behavioral information gain from the phase origin;
2. population Elo against the frozen valid human pool;
3. live-match win rate.

An optional diagnostic panel reports worst-role/worst-seed performance. Win
rate includes aggregate, P0, and P1 series. Iterations with no promotion retain
the parent value and are marked `no promotion`; they are not omitted.

Elo is updated from valid live matches against stable player identifiers. It is
not inferred from one head-to-head score. Faulted matches do not update Elo.
Curves, point tables, match records, replay references, policy hashes, and run
configuration are emitted together.

## Success and stopping conditions

The positive control demonstrates a functioning learning harness only when:

- most proposed candidates execute and reach live evaluation;
- claimed mechanisms activate through the public policy entry point;
- at least one replay-derived causal change improves a comparable live match;
- effective and ineffective knowledge persists across cycles;
- both roles and held-out seeds improve over the phase origin;
- gains survive live adaptive opponents and locked regression cases.

The SOTA pursuit succeeds when one certified candidate defeats every runnable
frozen human submission in both roles under the configured certification seed
matrix, with no AI fault and complete reproducibility artifacts. Unrunnable
packages are reported separately and never treated as defeated opponents.

## Verification

Unit and integration tests cover:

- profile registration without game-name branches in the HL CLI;
- strict generic match-record validation;
- AntWar2 bundle atomization and legality support;
- deterministic epsilon-smoothed KL for deterministic policies;
- replay event indexing and bounded causal windows;
- live-only promotion enforcement;
- P0/P1 metric symmetry and specialist retention;
- role-synthesis candidate creation;
- nearest-valid-checkpoint rollback;
- Experience ledger fact ownership and deterministic Skill projection;
- oracle and opponent-source access isolation;
- k=4 sibling lineage and proposal recovery;
- integer iteration curve output including no-promotion cycles;
- Phase 0 reference matches and Phase 1 blind-context manifest checks.

An end-to-end fake-game fixture proves that a second game profile can use the
same controller without Rollman or AntWar2 field names. The AntWar2 positive
control then validates the real frozen backend, replay loop, role-aware
selection, and API-driven policy learning.
