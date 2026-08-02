# Rollman HL Scoped Repair Protocol

## 1. Objective

The protocol searches for an interpretable Rollman policy that defeats every frozen human Ghost opponent. Each proposal cycle generates four mechanism-level siblings, applies one feedback-driven repair to the two most promising siblings, preserves every immutable version, and advances only through frozen match evidence.

The protocol does not create a four-ary search tree. Each selected branch contains at most one linear repair descendant. The parent, initial sibling, and repaired descendant remain independently addressable for audit and rollback.

## 2. Experiment provenance

The scoped-repair phase uses `origin.mode: imported_version` with these frozen inputs:

- source run: `run-20260802-gpt55-k4-interface-v4`
- source version: `v000037`
- source role: Rollman
- active curriculum target: `rank15`
- locked human opponents: the fourteen opponents certified by the source run

The phase writes its own run directory, configuration snapshot, source audit, events, versions, matches, measurements, research state, and reports. The aggregate report maps the imported origin to global HL iteration 11; phase iteration `n` maps to global iteration `11 + n`.

## 3. Ablation parameters

The iteration configuration exposes these frozen parameters:

```yaml
iteration:
  candidates_per_cycle: 4
  planner_enabled: true
  reducer_enabled: true
  scope_contract_required: true
  repair_enabled: true
  repair_top_k: 2
  repair_rounds: 1
  quick_screen_seeds: 1
  finalist_count: 2
  finalist_seeds: 3
```

Validation rules:

- `repair_top_k` is between 1 and `candidates_per_cycle` when repair is enabled.
- `repair_rounds` is non-negative and equals one for this experiment.
- repair requires planner mode and staged evaluation.
- `scope_contract_required` is independently ablatable.
- `repair_enabled`, `repair_top_k`, and `repair_rounds` require no code changes for ablation.

## 4. Branch scope contract

Each planner branch brief contains exactly these fields:

- `branch_index`
- `diagnosis`
- `mechanism`
- `activation_condition`
- `preservation_contract`
- `expected_change`
- `falsifier`

`activation_condition` defines a general predicate over observable game state. It may use relative geometry, board topology, skill state, score state, finite memory, and rule-defined events. It may not use seed, replay identity, absolute replay coordinates, human identity, or source-code knowledge.

`preservation_contract` names the existing decision path that retains control outside the activation condition. A candidate may grow arbitrarily large inside the scoped mechanism, but it may not globally replace scorers, weights, predictors, or action selection when the branch evidence supports only a bounded condition.

The planner prompt requires four distinct activation conditions and rejects threshold-only or weight-only variants. The candidate prompt requires one gated integration point and an explicit falsifier. Source size and `if/else` count are not selection penalties.

## 5. Proposal-cycle state machine

One cycle executes these phases:

1. **Planner** — produces four validated branch briefs from the imported parent, research state, bounded replay summaries, and game digest.
2. **Initial candidates** — four isolated coding acts start from the same immutable parent and implement one branch brief each.
3. **Quick screen** — each completed initial candidate plays the same frozen quick-screen seed against the active target.
4. **Repair selection** — the framework ranks completed initial candidates with the existing lexicographic diagnostics and selects exactly `repair_top_k` branches.
5. **Feedback packet** — each selected branch receives a bounded packet containing:
   - parent and candidate version identifiers;
   - branch scope contract;
   - same-seed parent and candidate scores, margins, and results;
   - parent and candidate replay summaries;
   - candidate failure and runtime diagnostics;
   - immutable paths to the parent and candidate policy snapshots.
6. **Repair act** — one fresh coding act starts from the initial candidate. It must explain the regression or missed improvement, then narrow the activation condition, fix the integration, or retain the initial implementation. It may not switch to another branch mechanism.
7. **Repair quick screen** — each completed repaired descendant plays the same frozen quick-screen seed.
8. **Branch representative selection** — the framework compares the initial candidate and repaired descendant of each selected branch. The stronger immutable version represents that branch; an unsuccessful repair cannot erase its initial candidate.
9. **Finalists** — the two strongest branch representatives play the remaining fixed finalist seeds. Their combined evaluations use the existing frozen selection diagnostics.
10. **Parent comparison** — the strongest finalist advances only when the existing linear successor rule prefers it to the imported parent. Otherwise the lineage head remains the parent.
11. **Reducer** — receives all initial, repair, representative, finalist, timeout, and rollback facts and updates the bounded research state.
12. **Measurement and report** — the selected lineage head produces policy KL/information gain, occupancy shift, Elo, full-pool win rate, target-gate score, and score-margin records at an integer global iteration.

## 6. Repair constraints

The repair prompt follows these rules:

- read the candidate and parent replay summaries before code changes;
- distinguish a wrong diagnosis from an overly broad activation condition and a faulty integration;
- preserve the initial candidate as an immutable alternative;
- perform no grid search, parameter enumeration, seed memorization, opponent-identity checks, or unrelated refactoring;
- use at most two bounded trace windows;
- run one compile check and one NumPy-backed SDK object smoke test;
- write a structured experience update with empirical evidence and a falsifier;
- stop after the first successful verification.

## 7. Events and artifacts

The event log records:

- `repairs_selected`
- `repair_started`
- `repair_completed`
- `branch_representative_selected`

Every repair act receives a checkpoint and raw provider JSONL. Every repair version receives an immutable version snapshot and evaluation events. The proposal directory stores `repair_input-bNN.json` for each repaired branch. Events contain no API key or hidden provider credential.

The branch report contains one row per initial and repaired version with:

- iteration and global iteration;
- branch index and stage (`initial` or `repair-1`);
- parent version and version identifier;
- selected representative flag;
- provider status;
- target win rate and mean score margin;
- local policy KL when measured.

## 8. Failure and resume behavior

- Provider 429 with zero usage follows the existing cooldown and retry policy.
- A timed-out or failed initial candidate cannot enter repair selection.
- A failed repair leaves the initial candidate eligible as the branch representative.
- Fewer than two completed initial candidates reduce the repair count to the number available.
- Proposal-cycle checkpoints and events identify completed initial and repair stages. Resume reuses persisted successful provider output and immutable evaluations rather than issuing duplicate billable acts.
- Reducer failure does not invalidate completed matches or selected lineage state.

## 9. Aggregate curves

The aggregate report renders exactly four primary panels with integer global HL iteration on the x-axis:

1. mean local policy KL / information gain;
2. Rollman Elo;
3. win rate against the full frozen human pool;
4. mean Rollman-minus-Ghosts score margin against the full frozen human pool.

Phase-local CSV files remain unchanged and auditable. The aggregate CSV adds `source_run`, `phase_iteration`, and `global_iteration`.

## 10. Verification requirements

Automated tests prove:

- strict parsing and validation of the two scope-contract fields;
- prompt inclusion of activation and preservation requirements;
- four initial siblings start from one immutable parent;
- only the configured top-k completed branches receive one repair act;
- repaired versions descend linearly from their initial candidates;
- initial versions survive failed or regressing repairs;
- finalists use immutable representative versions;
- the parent remains selected when every representative regresses;
- successful repair output is reusable after interruption without another provider call;
- repair events, reports, token totals, and global iteration mapping are deterministic;
- the fake end-to-end run and the full repository test suite pass.

## 11. Acceptance criteria

The protocol is ready for the Rollman experiment when:

- configuration validation, proposal schema, prompts, controller, resume, report, and fake end-to-end tests pass;
- a dry-run exposes all repair ablation parameters without a credential;
- an imported-version run proves provenance from `v000037`;
- one fake cycle produces four initial candidates, two repair descendants, two representatives, two finalists, one selected lineage head, and one integer aggregate curve point;
- API credentials remain confined to provider subprocesses and absent from all artifacts.
