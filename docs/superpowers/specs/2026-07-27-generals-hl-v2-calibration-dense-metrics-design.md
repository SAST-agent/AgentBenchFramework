# Generals HL Round-2 Calibration and Dense Metrics Design

**Date:** 2026-07-27  
**Status:** Direction approved on 2026-07-27; written-spec review pending  
**Formal benchmark version:** `generals-hl-pilot-v1` (unchanged)  
**Calibration benchmark version:** `generals-hl-calibration-v1`  
**Parent run:** first complete `v0 → v1` Generals HL pilot

## 1. Goal

Run a scientifically comparable second Codex heuristic-learning act while
removing the calibration floor effect and adding dense trajectory diagnostics:

```text
frozen original v0
→ separate weak-opponent calibration
→ recover the first Codex-produced v1
→ v1 learning episodes and compact dense feedback
→ one new Codex act
→ immutable v2
→ same formal benchmark evaluation
→ calibration, dense traces, behavior change, budgets, logs, and CI data
```

The round must answer three separate questions:

1. Can the original v0 achieve a non-degenerate 20%–60% win rate against a
   frozen weak calibration opponent?
2. Does the second Codex act change behavior and dense game progress?
3. Does v2 improve on the unchanged formal human-opponent benchmark?

These questions use separate measurements. A calibration result or dense
diagnostic must never be substituted for the formal benchmark score.

## 2. Research Contract

`附件/information_gain.md` is the measurement authority for this design.
The implementation must preserve these boundaries:

- one complete game is one episode;
- one accepted player turn-level macro-action is one environment step;
- primitive commands are auxiliary counts, not environment steps;
- `coding_agent_act`, `episode`, `env_step`, and
  `game_agent_decision_step` remain distinct;
- raw/evo/gain comparisons use the same versioned formal benchmark cases;
- learning, evaluation, and total budgets are recorded separately;
- the main budget AUC uses learning-only budget;
- invalid or incomplete fixed-case evaluation produces a missing aggregate
  score, not a loss or zero;
- finalized events are append-only;
- missing data remains missing rather than defaulting to zero;
- deterministic action disagreement is a behavior-change measurement, not
  strict policy KL or epistemic information gain;
- occupancy shift is reported separately and is not added to action
  disagreement or dense diagnostics.

The new survival, territory, army, coin, and main-general-pressure values are
called **dense trajectory diagnostics**. They are not called information gain,
reward, policy KL, or benchmark score.

## 3. Considered Approaches

### Selected: separate weak calibration suite, no human policy intervention

Add a deterministic weak SDK opponent and tune its frozen difficulty using
calibration-development seeds. Keep the original v0 and first Codex v1
unchanged. Run the second Codex act from the exact v1 snapshot.

Advantages:

- removes the calibration floor without modifying the policy under study;
- preserves the clean `v0 → v1 → v2` Codex lineage;
- makes the calibration score auditable and reproducible;
- avoids attributing researcher-written rules to Codex.

### Rejected: manually strengthen v0 before the second act

This may put the formal score in a useful range, but it inserts an unmeasured
human policy update between Codex acts. It would require a distinct
researcher-authored version and would make the v0/v1/v2 attribution less
direct. It remains a future baseline-design experiment, not part of round 2.

### Rejected: include the weak opponent in the formal aggregate

This would mechanically raise raw/evo/gain and change the benchmark version.
It would prevent direct comparison with the completed first round and could
create a misleading performance claim.

## 4. Immutable Version Lineage

The completed first run remains unchanged. Round 2 creates a new run that
references it:

```text
parent_run_id      = <first complete Generals HL run>
parent_version_id  = v1
version_before     = v1
version_after      = v2
coding_agent_act   = 2
```

The round-2 controller must:

1. resolve the parent run and verify its completed status;
2. verify the recorded v1 content hash;
3. copy the exact v1 snapshot into a new run-owned workspace;
4. record the parent run, parent version, source manifest, and source hash;
5. never edit the parent run or committed v0 template;
6. snapshot the second Codex result as v2 even when no files changed.

The global research curve retains:

```text
raw_score    = formal score of original v0
evo_score_1  = formal score of first Codex v1
evo_score_2  = formal score of second Codex v2
gain_k       = evo_score_k - raw_score
```

Round 2 may re-evaluate v1 to obtain paired dense traces and verify
reproducibility. A repeated v1 result is not a new coding-agent act.

## 5. Dual-Track Evaluation

### 5.1 Formal benchmark

The formal benchmark remains exactly `generals-hl-pilot-v1`:

- the same high, medium, and low human algorithms;
- evaluation seeds `280101`, `280202`, and `280303`;
- both evaluated-agent seats;
- the same official engine hash and termination rules;
- 18 fixed cases per evaluated version.

The weak calibration opponent is absent from the formal `BenchmarkSpec`.
Formal aggregate score remains:

```text
(wins + 0.5 × draws) / 18
```

It is emitted only when all 18 cases have valid results. Per-tier and seat
results remain secondary formal metrics.

### 5.2 Calibration benchmark

The calibration benchmark is a separate versioned suite:

```text
benchmark_id = generals-hl-calibration-v1
target       = original v0
target range = 20%–60% held-out win rate
seats        = both
```

It has independent development and held-out test seed lists. Neither list
overlaps formal evaluation seeds or human-opponent learning seeds.

The first calibration version freezes:

```text
development seeds = 282101, 282202, 282303, 282404, 282505
held-out seeds     = 282601, 282702, 282803, 282904, 283005
seats              = 0, 1
```

Candidate weak policies are declared before held-out evaluation:

1. `passive`: legal end-turn behavior with minimal defensive movement;
2. `local-expander`: attacks adjacent capturable neutral resources;
3. `resource-greedy`: routes one stack toward the nearest resource but does
   not perform global opponent search. Its development-tuned ceiling may use
   the same bounded production and spare-stack rules as frozen v0.

Only development cases may select the candidate or tune fixed thresholds.
The selected opponent source, parameters, manifest, and content hash are then
frozen before its held-out cases are run.

The held-out calibration result is opened once. If it misses 20%–60%, that
candidate is recorded as a failed calibration version. Any redesign uses a
new calibration version and fresh unseen held-out seeds; failed results are
not overwritten.

Calibration test replays, results, and seeds are never included in the Codex
prompt. Calibration-development episodes may be used as learning feedback.

Implementation note (2026-07-27): the initially bounded versions of all three
candidates gave v0 a 100% development win rate. Without opening held-out
seeds, `resource-greedy` was promoted to a deterministic mirror-strength
ceiling: adjacent combat, non-owned-resource routing, affordable main
production upgrades, and one spare-stack move. The repeated development
scores were 100%, 100%, and 50%, so `resource-greedy` was frozen. The failed
development runs remain separate append-only run records.

### 5.3 Calibration reporting

Calibration exposes its own fields:

```text
calibration_benchmark_id
calibration_status
calibration_score
calibration_wins
calibration_losses
calibration_draws
calibration_per_seat
calibration_target_range
calibration_in_target_range
```

None of these fields feeds formal `benchmark_score`, `raw_score`, `evo_score`,
`gain`, or their AUC values.

## 6. Dense Trajectory Diagnostics

### 6.1 Sampling

Each match stores a round-aligned target-agent trace sampled at:

1. the initial official state;
2. every completed full round after player 1 acts and the official round
   update runs;
3. the terminal state, including termination during either player's turn.

Each sample contains:

```text
sample_index
official_round
sample_kind = initial | completed_round | terminal
evaluated_seat
target_alive
opponent_alive
```

This avoids counting two player turns as two territorial time points while
retaining the terminal state. Raw state references and canonical state IDs
remain available for audit.

### 6.2 Survival

Store:

```text
terminal_round
completed_rounds_survived
termination_type
terminated
truncated
outcome
```

`completed_rounds_survived` is the number of official full-round updates
observed before termination. It is interpreted jointly with outcome:
surviving longer in a loss may indicate improvement, while a faster win must
not be treated as regression.

### 6.3 Territory

For each player, owned territory is the number of traversable cells whose
official owner equals that player. Store:

```text
target_territory
opponent_territory
territory_margin = target_territory - opponent_territory
territory_share  = target_territory /
                   (target_territory + opponent_territory)
```

If the share denominator is zero, the share is missing rather than zero.

### 6.4 Army

Army is the sum of official army counts on cells owned by the corresponding
player. Store:

```text
target_army
opponent_army
army_margin = target_army - opponent_army
army_share  = target_army / (target_army + opponent_army)
```

If the share denominator is zero, the share is missing.

### 6.5 Coins

Use the official per-player coin vector:

```text
target_coins
opponent_coins
coin_margin = target_coins - opponent_coins
coin_share  = target_coins / (target_coins + opponent_coins)
```

If both players have zero coins, `coin_share` is missing.

### 6.6 Main-general pressure

Pressure uses shortest-path distance across traversable cells, not raw
Manhattan distance. For a main general, its pressure neighborhood contains
cells with graph distance at most two.

Movable attacking mass on an owned cell is:

```text
max(cell_army - 1, 0)
```

For pressure on the opponent main:

```text
target_attack_mass_near_opponent_main
opponent_defense_mass_near_opponent_main
pressure_for =
    target_attack_mass_near_opponent_main
    - opponent_defense_mass_near_opponent_main
```

Apply the symmetric calculation around the target main:

```text
pressure_against
net_main_pressure = pressure_for - pressure_against
```

Defense mass includes all owned army in the two-step neighborhood, including
the main cell. If a main no longer exists, alive/outcome fields are
authoritative and the corresponding pressure fields are missing.

### 6.7 Episode summaries

Every episode retains the complete ordered dense trace. Derived summaries may
include:

- terminal value;
- minimum and maximum;
- unweighted time average;
- trapezoidal AUC over official-round coordinates.

An episode-level derived value is missing if a required sample is missing.
The calculator does not interpolate across missing samples.

Episode summaries remain per episode. Reports may add episode-balanced
mean/median and intervals, while decision- or duration-balanced summaries
must be labeled separately.

No weighted composite of territory, army, coins, survival, and pressure is
introduced in round 2.

## 7. Behavior Measurements

For `v1 → v2`, use only target-agent decision states from learning
trajectories:

- save canonical decision state IDs;
- run v1 and v2 on the same probe states;
- save the raw `action_disagreement_trace` for each episode;
- derive episode-balanced and decision-balanced summaries;
- save normalized occupancy histograms and `occupancy_shift` separately.

Because the current deterministic HL policy does not expose a normalized
distribution over the complete legal macro-action set:

```text
local_policy_kl_trace = missing
policy_kl_status =
  complete_macro_action_distribution_unavailable
```

The report must not display action disagreement as KL or epistemic
information gain.

## 8. Second Codex Learning Act

The second act starts from the verified v1 snapshot. Its learning material
may contain:

- official rules and replay interpretation;
- current v1 strategy documentation and tests;
- the first `v0 → v1` diff;
- new v1 learning episodes against low and medium human opponents;
- calibration-development episodes;
- per-episode dense traces and deterministic compact summaries;
- selected critical decision windows from losses.

New human-opponent learning episodes use seeds `283101`, `283202`, and
`283303`, from both seats, against only the low and medium opponents. They
remain disjoint from the formal and calibration seed sets.

It must not contain:

- formal evaluation seeds, results, or replays;
- calibration-test seeds, results, or replays;
- high-tier opponent source or evaluation feedback;
- arbitrary dense-score weights;
- opponent source code.

Full learning replays remain run artifacts. Prompt construction uses a
deterministic compact representation with a frozen `262144`-byte UTF-8 limit
and records:

```text
prompt_selection_policy
included_episode_ids
omitted_episode_ids
prompt_bytes
prompt_estimated_tokens
prompt_truncated
```

Truncation occurs only at whole-record boundaries. The exact prompt and
provider raw JSONL remain retained.

The controller performs exactly one new non-interactive Codex invocation.
There is no accept/reject gate, rollback, best-version substitution, or
unlogged second attempt.

## 9. Events, Budgets, and Artifacts

New fields are additive and preserve schema compatibility. Suggested
finalized event types are:

```text
calibration_spec
calibration_result
dense_trajectory
dense_episode_summary
lineage_import
```

Every event includes the public schema envelope:

```text
schema_version
event_id
event_type
run_id
created_at
```

The fact source remains `events.jsonl`. Reports and `summary.json` are
derived. Full raw artifacts include:

```text
benchmark/formal-spec.json
benchmark/calibration-spec.json
benchmark/calibration-opponent-manifest.json
matches/{version}/{case_id}/replay.jsonl
matches/{version}/{case_id}/dense-trace.jsonl
provider/codex-act-2.raw.jsonl
provider/codex-act-2.prompt.md
versions/v1/manifest.json
versions/v2/manifest.json
versions/v1-to-v2.patch
quality.json
```

Budget ledgers separately retain calibration construction, learning,
evaluation, and total:

```text
coding_agent_acts
episodes
env_steps
game_agent_decision_steps
primitive_commands
prompt_tokens
completion_tokens
total_tokens
time_s
```

Candidate selection against v0 counts as calibration-construction budget and
is excluded from coding-agent learning AUC. After the opponent is frozen,
calibration-development episodes actually included in the Codex feedback
count as learning. Calibration-test and formal benchmark games count as
evaluation. Total budget includes every executed game. Provider-unknown token
fields remain unknown.

## 10. Framework and CI Presentation

The Framework summary and report expose three non-interchangeable panels:

1. **Formal performance:** raw/evo/gain and formal budget AUC;
2. **Calibration:** score, target band, seat split, and calibration version;
3. **Dense diagnostics:** per-episode traces and paired v1/v2 differences.

The quality panel continues to show malformed events, unknown event types,
duplicate IDs, missing fields, incomplete evaluation, and missing metrics.
The CI must preserve missing and incomplete states without interpolation.

Formal and calibration series use distinct names, colors, legends, and
benchmark IDs. A calibration score must never appear on the formal
benchmark-score curve.

## 11. Testing and Verification

Implementation is test-driven. Required coverage includes:

- deterministic weak-opponent legal action and protocol tests;
- calibration development/test separation and frozen-hash tests;
- proof that calibration cases are absent from the formal `BenchmarkSpec`;
- dense metric unit tests on small synthetic official states;
- round-aligned sampling and terminal-state tests for both seats;
- graph-distance pressure tests with obstacles;
- per-episode trace, AUC, missing denominator, and missing-main tests;
- incomplete evaluation produces missing aggregate score;
- calibration-construction/learning/evaluation/total budget classification;
- prompt leakage and whole-record truncation tests;
- v1 parent hash and lineage import tests;
- fake-provider `v1 → v2` pipeline test;
- CI/report tests that formal and calibration series remain separate;
- official-engine live smoke tests;
- final full Framework, baseline, data-contract, and report verification.

The real run additionally verifies:

- original v0 held-out calibration score is 20%–60%;
- all formal cases for each reported version are valid;
- exactly one second Codex act is recorded;
- v2 is snapshotted regardless of score direction;
- full provider, replay, dense, budget, diff, and quality artifacts exist;
- event-quality diagnostics contain no silent schema corruption.

## 12. Failure Semantics

- A failed calibration candidate is retained and versioned.
- A held-out miss does not silently trigger tuning on those seeds.
- An invalid fixed formal case makes the formal aggregate incomplete.
- A provider failure still records the act and readable workspace snapshot.
- An unreadable post-act workspace leaves `version_after` missing.
- A dense metric calculation failure marks that metric missing and emits a
  quality diagnostic; it does not fabricate zero.
- A report failure does not mutate the run fact source.
- No run, version, replay, event, or failed attempt is silently overwritten.

## 13. Non-Goals

Round 2 does not:

- add the weak opponent to the formal score;
- claim dense diagnostics are reward or information gain;
- implement a complete legal macro-action distribution or strict HL KL;
- manually strengthen the policy between Codex acts;
- perform multiple Codex acts or automatic prompt retries;
- train an RL policy;
- alter historical opponent source;
- claim statistical significance from this single second act.
