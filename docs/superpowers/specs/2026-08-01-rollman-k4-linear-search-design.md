# Rollman K=4 Linear Search HL Harness Design

## 1. Objective

The harness runs a reproducible, API-funded heuristic-learning experiment for
`29_rollman`. A coding agent creates an interpretable Rollman policy from the
frozen game package, learns from valid matches against human-written Ghost
programs, and iterates until the policy passes every valid opponent in the
frozen certification pool.

The experiment has these fixed properties:

- the candidate role is Rollman (`role_id=0`);
- every proposal cycle produces exactly four candidate programs from one common
  search parent;
- exactly one candidate becomes the next search parent, so the active search is
  linear rather than a persistent tree;
- all candidate artifacts remain immutable and available for audit or rollback;
- policy size, branch count, and source-line count are not optimization
  penalties;
- the experiment is run by `AgentBenchFramework` and the configured coding-agent
  provider, without depending on a Codex App conversation;
- model credentials are read from `.env` or the configured environment variable
  and are never exposed to candidate code.

Ghost-role HL, a human Rollman opponent pool, and the 125-action Ghost joint
policy measurement are outside this experiment.

## 2. Reproducibility Boundary

The full loop is started by one framework command:

```bash
agentbench hl run \
  --config configs/hl/29_rollman-k4.yaml \
  --run-dir .agentbench/29_rollman/runs/RUN_ID
```

The framework owns proposal-cycle scheduling, coding-agent invocation, source
snapshots, matches, measurements, candidate selection, rollback, experience
updates, reporting, pause, and resume. The Codex CLI is a pinned coding-agent
runtime over the configured Responses-compatible API; it is not an external
human-supervised conversation.

Every run freezes and records:

- the resolved configuration with secrets redacted;
- repository commit and dirty-state manifest;
- Codex CLI version and invocation arguments;
- provider model, reasoning effort, base URL, and wire protocol;
- Python version and relevant package versions;
- game logic, SDK, human corpus, rules, decision-space, and Replay Skill hashes;
- opponent build artifacts and execution-validity audit;
- learning, selection, and hidden certification seed protocols;
- every exact coding-agent prompt, JSONL response, tool record, token usage,
  workspace diff, version hash, match, replay, trace, and selection decision.

Repository-relative paths are stored in portable configuration. Machine-local
source roots can be supplied through environment variables or an ignored local
configuration file. Absolute paths are resolved only at run time and are not
part of the portable experiment specification.

## 3. Frozen Game Context

Each game exposes an immutable game package:

```text
game-package/
├── rules.md
├── decision_space.yaml
├── replay-skill/
├── game_digest.json
└── manifest.json
```

`rules.md`, `decision_space.yaml`, and the Replay Skill remain the authoritative
human-reviewed inputs. `manifest.json` contains their hashes. The compact
`game_digest.json` is compiled deterministically by the framework from the
decision-space schema, rule-section index, Replay Skill front matter, and
research-isolation configuration. It contains no model-authored tactical facts.
The digest is validated against a schema and stored with the run. Exact score,
collision, item, or termination semantics continue to resolve to the
authoritative rule sections.

The bootstrap coding act reads the complete authoritative package and writes the
initial Rollman policy from the empty candidate scaffold. Later acts receive the
digest and package hash by default. They read authoritative sections on demand
when a hypothesis depends on a precise rule. Prompts do not require full
re-reading of every static file on every cycle.

## 4. Explicit Research-State Checkpoint

The outer loop does not rely on opaque provider conversation memory. Every
completed proposal cycle writes a bounded `research_state.json` containing:

- selected search-parent version and official champion version;
- active target and locked certified opponents;
- stable game-grounded findings;
- failed hypotheses with replay evidence and falsifiers;
- open questions;
- compact opponent-model findings;
- recent candidate comparisons;
- stagnation counters and exploration-debt state.

The full history remains in append-only events and provider artifacts. The
bounded research state is the only cross-cycle semantic context required to
reconstruct the next prompt.

Coding acts use stateless, fully logged inputs. Their static prefix is
byte-identical whenever the game-package hash and provider configuration are
unchanged, allowing provider prompt caching without making correctness depend on
cache availability. A configured token threshold triggers research-state
reduction before the next cycle.

## 5. Human Opponent Validity and Difficulty Order

Only valid opponent executions enter learning or certification. Infrastructure
failure, process startup failure, malformed output, or external timeout produces
an incomplete match and no score. A game-rule timeout counts as a result only
when the frozen backend explicitly reports it as a valid game outcome.

Because the registered human programs control Ghosts, they cannot play direct
head-to-head matches against one another. Difficulty is calibrated by running
every Ghost program against the same frozen panel of reference Rollman policies
on identical seeds. The ordering key is lexicographic:

1. valid capture/win rate against the reference panel;
2. mean Ghost-minus-Rollman score margin;
3. worst-case score margin;
4. stable opponent ID as a deterministic final tie-breaker.

The curriculum target is the weakest empirically calibrated opponent that the
official champion has not certified. The target changes only after hidden
certification passes.

## 6. One Proposal Cycle

`proposal_cycle` is the integer shown on the requested iteration-axis plots. A
cycle has the following deterministic stages.

### 6.1 Evidence collection

The search parent plays the active target on rotating learning seeds. The
framework produces bounded replay summaries, selected trace windows, match
outcomes, score decomposition, and primitive-action counterfactual diagnostics.

Counterfactual diagnostics operate only on the human-reviewed Rollman action
support `{0,1,2,3,4}`. They do not introduce tactical labels or a synthetic
hypothesis action space.

### 6.2 Hypothesis planner

One review-model act consumes the common evidence packet and produces four
structured, falsifiable, mechanism-distinct branch briefs. A brief contains:

- replay-grounded causal diagnosis;
- proposed mechanism;
- expected observable change;
- falsification condition;
- files allowed to change.

Parameter-only variants of one mechanism do not satisfy the diversity contract.

### 6.3 Four coding candidates

Four independent coding-agent acts start from the same immutable search-parent
snapshot. Each receives one branch brief, the same compact game context,
`research_state.json`, and the same bounded evidence packet. Each may use
if/else rules, tables, finite memory, state machines, graph search, planning, or
other interpretable program structures.

Policy growth is allowed. The prompt does not request source compression or
penalize additional rules. It prohibits opponent-source access, opponent-ID
hardcoding, seed hardcoding, replay-ID hardcoding, fixed replay-coordinate
hardcoding, and unguided parameter grid search.

### 6.4 Successive-halving evaluation

All four candidates receive at least one real match and therefore one feedback
trajectory.

1. Static interface validation, compilation, and candidate smoke test.
2. One common quick-screen seed for every valid candidate.
3. The best two candidates receive three additional rotating selection seeds.
4. The selected candidate is determined from all completed selection matches.
5. Hidden certification runs only when the selected candidate reaches the
   configured target-training threshold.
6. Locked-opponent regression checks and full-pool certification run only after
   target hidden certification passes or at an explicit audit interval.

The coding agent never receives hidden certification seeds or their replays.

### 6.5 Comparative reducer

One review-model act receives the four branch briefs, four code diffs, completed
measurements, and bounded feedback summaries. It updates `research_state.json`
with accepted findings, rejected hypotheses, unresolved conflicts, and a
recommended mechanism class for the next cycle. Framework-recorded match facts
override model-authored claims.

## 7. Linear Search Parent and Official Champion

The harness maintains two pointers:

- `official_champion`: the best version under frozen hidden certification;
- `search_parent`: the single version expanded by the next four-candidate
  proposal cycle.

There is no active search tree. Every cycle selects exactly one next
`search_parent`. Non-selected candidates remain immutable audit artifacts but do
not produce descendants.

Candidate selection is lexicographic and uses only valid completed matches:

1. target wins and draws on the common finalist seed set;
2. mean Rollman-minus-Ghosts score margin;
3. worst-case score margin;
4. level progression and valid completion count;
5. survival and capture count;
6. behavioral novelty from the search parent;
7. lower stable branch index as the final deterministic tie-breaker.

The official champion advances only after hidden target certification and
locked-opponent regression checks pass. The search parent may advance without a
win when it improves the earlier lexicographic diagnostics. This allows a
multi-cycle mechanism to develop instead of forcing every unsuccessful version
back to the origin.

`exploration_debt` limits proxy drift. A search-parent lineage may advance for at
most three cycles without improving target wins or score margin. When the limit
is reached, the framework selects the best historical non-dominated version for
the active target. If none exists, it returns to the official champion, not
unconditionally to `v0`.

Rollback therefore chooses the nearest verified safe ancestor or best historical
target specialist. All snapshots remain available for explicit manual rollback.

## 8. Initial Policy and Stop Conditions

The bootstrap coding act writes the initial interpretable Rollman algorithm from
the frozen rules, decision space, Replay Skill, and candidate interface. It does
not copy a benchmark baseline implementation.

The run has no iteration cap. It completes only when one official champion:

- passes every valid registered human Ghost opponent;
- passes the frozen hidden seed protocol for every opponent;
- has no incomplete certification cases;
- passes source-isolation and reproducibility audits.

The run pauses, rather than reports success or failure, on exhausted API credit,
provider outage, invalid opponent corpus, infrastructure failure, or explicit
operator request. Resume continues from append-only events and explicit
checkpoints without using a Codex App conversation.

## 9. Measurements and Plots

The action space for Rollman measurement is exactly `{0,1,2,3,4}`. Deterministic
HL policies use the approved epsilon measurement channel from
`information_gain.md`. The metric is behavioral policy information gain, not
epistemic entropy reduction.

The primary report uses integer `proposal_cycle` on the x-axis:

1. behavioral policy IG versus proposal cycle;
2. Elo versus proposal cycle;
3. frozen reporting-panel win rate versus proposal cycle;
4. Rollman-minus-Ghosts score margin versus proposal cycle.

At each cycle, the four `search_parent -> candidate` behavioral measurements
appear as light scatter at the same integer x-coordinate. The selected branch IG
is a solid line. The performance panels show the selected search parent as a
solid line and the official champion as a step line. Episode-level IG points and
aggregate confidence intervals remain available without collapsing the raw
trace.

Every selected search parent runs a fixed, hidden reporting panel containing all
valid opponents and one frozen seed per opponent. This produces one comparable
win-rate, score-margin, and Elo point for every integer proposal cycle. Reporting
panel replays and seeds are not exposed to coding acts. Multi-seed full
certification remains gated on target success.

Resource-efficiency reports additionally use cumulative coding-agent acts,
learning episodes, total tokens, uncached tokens, and wall time as x-axes. One
proposal cycle contains one planner act, four coding acts, and one reducer act;
these counts are never conflated.

Elo and frozen-pool win rate use only valid completed games. Incomplete cases are
missing values and are never silently converted to losses, wins, or zero.

## 10. Configuration Surface

The experiment is parameterized for ablation without code changes:

```yaml
run:
  role: rollman
  context_mode: checkpointed_stateless

iteration:
  candidates_per_cycle: 4
  planner_enabled: true
  reducer_enabled: true
  quick_screen_seeds: 1
  finalist_count: 2
  finalist_seeds: 3

selection:
  mode: linear_lexicographic
  exploration_debt_cycles: 3
  source_size_penalty: false

context:
  use_game_digest: true
  require_full_static_read_each_cycle: false
  research_state_max_bytes: 16384
  reduction_token_threshold: 250000

evaluation:
  target_mode: weakest_empirical_unpassed
  rotate_learning_seeds: true
  hide_certification_seeds: true
  reporting_panel_every_cycle: true
  reporting_seeds_per_opponent: 1
  full_pool_on_target_pass: true

measurement:
  epsilon: 0.05
  local_policy_kl: true
  occupancy_shift: true
```

Primary ablations vary `candidates_per_cycle`, planner/reducer presence,
counterfactual feedback, opponent distillation, and exploration debt. The
benchmark, opponent pool, hidden seed protocol, model, and total budget are held
fixed for comparisons.

## 11. Required Verification

Implementation is accepted only when automated tests demonstrate:

- four candidates use one identical parent and exactly one next parent is
  selected;
- no non-selected candidate is scheduled for descendants;
- a dense diagnostic improvement can advance the search parent without
  advancing the official champion;
- exploration debt selects a historical specialist or official champion rather
  than always selecting `v0`;
- static context is read fully at bootstrap and represented by a verified digest
  in later cycle prompts;
- every coding act is reconstructible from persisted explicit inputs;
- `.env` credentials are redacted from logs and unavailable to candidate code;
- invalid/TLE infrastructure matches cannot inflate wins, Elo, or certification;
- all four candidates contribute feedback to the reducer;
- plot x-values are integer proposal cycles and the four branch points share the
  same cycle value;
- IG uses the approved primitive action support and epsilon measurement channel;
- run pause/resume produces the same next-cycle inputs as uninterrupted
  execution.
