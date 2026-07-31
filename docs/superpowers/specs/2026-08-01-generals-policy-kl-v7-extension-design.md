# Generals Controlled Policy-KL v7 Extension Design

## Goal

Produce one new, self-contained, append-only Generals controlled-reference
policy-KL measurement run covering `v0` through `v7`, then regenerate the
English publication three-panel figure with the additional `v6→v7`
transition.

The extension must preserve direct comparability with the completed v1
measurement. It therefore reuses the exact same 12 controlled reference
states and exact canonical support counts after cryptographic verification.
It does not mutate the completed v1 run, recollect states, recount the action
space, or treat the invalid v8 attempts as a retained policy version.

## Scientific Contract

The measurement remains `controlled_reference_policy_kl`: deterministic
policy-change KL on a frozen reference domain. It is not trajectory KL,
occupancy KL, benchmark score, reward, or epistemic information gain.

The frozen domain remains:

- strongest human opponent: `advanced-rank02-robinliu-v18`;
- collector seeds: `289101`, `289202`, and `289303`;
- evaluated seats: `0` and `1`;
- controlled decisions: `2` and `10`;
- exactly 12 reference states;
- macro-action-space specification and canonicalization from the completed v1
  measurement;
- epsilon sensitivity set: `0.001`, `0.01`, `0.05`, and `0.1`;
- primary epsilon: `0.01`;
- strict uniform `U_s` smoothing over each exact canonical support.

The retained history is ordered `v0` through `v7`. The v7 source is frozen to
run `20260730_1739_680b1632` with content hash
`c1eb1e4eae5fb1afa393e15370d744743fe0af206cc05a329fa36bd62ca3c4c4`.
No v8 entry is permitted because no valid retained v8 policy exists.

## Chosen Architecture

### 1. Frozen v2 reference manifest

Add a new assets manifest rather than editing
`policy-kl-reference-v1.toml`. The v2 manifest contains:

- measurement ID `generals-policy-kl-reference-v2`;
- the unchanged collector, state-domain, epsilon, and primary-epsilon fields;
- the immutable v0-v7 history, including the approved v7 run and source hash;
- the completed v1 measurement run ID as the reuse source;
- one canonical source-run tree hash computed from every regular file's
  relative path and SHA-256 digest in sorted order.

The assets loader validates the complete manifest byte contract, history
order, exact approved run IDs and hashes, disjoint/frozen seeds, and the
source-run identity. The v1 loader and manifest remain supported unchanged.

### 2. Verified import into a new run

Expose `generals extend-policy-kl-v7`, a dedicated command that accepts the v2
manifest and the explicit completed v1 source-run directory. It always starts
a new Framework measurement run. `generals recover-policy-kl-v7` can resume
only that new run if it stops before completion.

Before any v7 probe, the pipeline verifies that the source run:

- is complete and has the v1 measurement ID;
- has zero malformed, invalid, duplicate, missing-ID, and unknown events;
- uses the expected action-space and reference-state specifications;
- contains exactly 12 unique reference snapshots with matching state IDs;
- contains exactly 12 complete exact-count records with positive support;
- contains complete canonical actions for v0-v6;
- contains the six ordered v0→v6 transitions and all 288 per-state,
  four-epsilon KL facts;
- matches the hashes frozen by the v2 manifest.

Source verification uses the canonical tree-hash algorithm above; filesystem
metadata is excluded. The pipeline computes the tree hash before import and
again after finalization. Any difference proves that the supposedly read-only
source changed during the extension and prevents a complete result.

After verification, the pipeline materializes the reused artifacts inside the
new run. Copying makes the run self-contained; provenance receipts record the
source run, relative path, source digest, copied digest, semantic role, and
reuse status. The completed source run is never opened for writing.

The imported set is fixed rather than implementation-defined: both source
benchmark specifications, all 12 reference snapshots, all 12 exact-count records, the
v0-v6 policy manifests and source trees, all 84 v0-v6 state-action probe
records, the 288 rows in `measurement/per-state-kl.jsonl`, the old transition
summary, `events.jsonl`, `quality.json`, and `summary.json`. The benchmark,
The action-space specification, state, count, and policy artifacts occupy
their established paths in the new run. A v2 reference-state specification is
generated from the verified v1 coordinates and state IDs; both exact v1
benchmark specifications are also preserved under `provenance/source-run/`.
The new 336-row per-state file is generated from the 288 verified source
facts followed by the 48 new facts. The exact old per-state file, source-run
metadata, and old transition summary are preserved under
`provenance/source-run/`, so they cannot overwrite the new run's event,
quality, summary, per-state, or seven-transition artifacts. SQLite caches,
prepared opponents, and collector replays are not imported because the
completed state and count receipts are the frozen scientific inputs; their
integrity is still covered by the whole-source tree hash.

All 396 imported scientific events keep their original measurement fields and
values: 12 reference selections, 12 support counts, 84 v0-v6 actions, and 288
KL facts. New Framework event IDs and the new run ID identify their
materialization in the extension run. Each carries its original
`source_event_id`, `source_run_id`, and `reuse_status = "verified_reuse"`.
The 12 v7 action events and 48 v6→v7 facts have no source event and carry
`reuse_status = "new"`.

The new event types used for source verification and artifact reuse are added
to the Framework event contract, so their presence cannot be hidden as
unknown-event warnings.

### 3. Probe only v7

The existing immutable historical-policy resolver verifies and copies the v7
source from its retained HL run. On each of the 12 imported states, the
subprocess probe executes v7 twice. A usable observation requires:

- both invocations to finish successfully;
- identical raw actions across repeats;
- successful canonicalization under the frozen action-space spec;
- one nonempty legal canonical macro.

The v6 canonical actions are reused from the verified source run. No v0-v6
policy is executed again.

### 4. Compute v6→v7 and finalize one complete run

For each reference state and each frozen epsilon, compute
`D_KL(pi_v6 || pi_v7)` using the verified exact support size and strict uniform
`U_s` smoothing. This produces 48 new per-state KL facts.

The new run exposes one coherent projection:

- seven ordered transitions from v0→v1 through v6→v7;
- 336 per-state KL facts (`7 × 12 × 4`);
- 12 reference-state facts;
- 12 exact support-count facts;
- canonical action facts sufficient to audit every transition;
- coverage and epsilon sensitivity for every transition;
- explicit reuse and source-hash receipts;
- a clean event-quality report.

The six v0→v6 aggregates must be numerically identical to the completed v1
run. The only new numerical result is v6→v7.

## Artifact Layout

The extension run keeps the established layout and adds a provenance area:

```text
benchmark/
  action-space-spec.json
  reference-state-spec.json
reference/
  states/
action-space/
  counts/
policies/
  v0/ ... v7/
measurement/
  per-state-kl.jsonl
  transition-summary.json
provenance/
  source-run-receipt.json
  imported-artifacts.jsonl
  source-run/
    action-space-spec.json
    events.jsonl
    per-state-kl.jsonl
    quality.json
    reference-state-spec.json
    summary.json
    transition-summary.json
events.jsonl
quality.json
summary.json
```

The summary records the new measurement ID, source measurement run ID,
source receipt digest, reuse mode, policy history, seven transitions, and
event quality.

## Failure and Recovery Semantics

The extension fails closed. It must not silently recollect, recount, interpolate,
or substitute an artifact when verification fails.

- A source digest, specification, coordinate, event, or aggregate mismatch
  stops the run with an explicit provenance-validation failure.
- A missing, nondeterministic, failed, or illegal v7 action leaves the affected
  v6→v7 aggregate `null`, reports exact coverage, and finalizes as incomplete
  policy measurement.
- A process interruption leaves an auditable incomplete run that can be
  resumed append-only after revalidating all previously materialized files.
- Existing complete v1 artifacts and figures remain recoverable from git and
  the original run directory.

## Figure Update

Generalize the publication-figure loader using explicit measurement-ID
dispatch while preserving strict order and full coverage validation. A v1 run
must contain exactly the original six transitions; a v2 run must contain
exactly seven. The loader never infers the contract from list length. This
keeps the old figure reproducible, while the updated figure reads only the new
complete v2 run.

The output remains English-only and preserves the established visual design:

1. controlled-reference policy KL over iterations at epsilon `0.01`;
2. epsilon sensitivity for all four epsilon values;
3. exact canonical support size for the unchanged 12 reference states.

Panels (a) and (b) add `v6→v7`. Panel (c) is numerically unchanged. The figure
is written as editable SVG and 300-DPI PNG. v8 is neither plotted nor shown as
a zero-valued or missing transition.

## Verification Strategy

Implementation follows test-driven development. Required coverage includes:

1. v2 asset-loader acceptance and rejection of any domain, source-run, history,
   or v7 hash change;
2. rejection of incomplete, mutated, structurally inconsistent, or
   scientifically incomplete source runs;
3. proof that only v7 is newly probed and exactly 12 v7 observations are
   produced;
4. byte/digest verification for every copied artifact;
5. equality of all imported v0→v6 facts and aggregates with v1;
6. correct v6→v7 four-epsilon computation under exact uniform `U_s`;
7. 12/12 coverage, 336 per-state rows, ordered seven-transition summary, and
   clean event quality in the successful fake end-to-end run;
8. null-preserving incomplete behavior for a failed v7 state;
9. append-only recovery of an interrupted extension run;
10. continued acceptance of valid six-transition v1 runs, rejection of
    six-transition or malformed v2 runs, and rendering of `v6→v7` in both SVG
    and PNG;
11. focused Framework and assets suites, followed by their full test suites;
12. one real v2 extension run, artifact/hash audit, figure rendering, and
    visual inspection.

## Acceptance Criteria

The work is complete when:

- the old completed v1 measurement run is byte-for-byte unchanged;
- a new complete v0-v7 measurement run exists and names the verified source
  run;
- v7 source provenance matches its retained HL manifest;
- only v7 was newly executed against the 12 controlled states;
- all six old transitions exactly match v1 and v6→v7 has 12/12 coverage at all
  four epsilons;
- the new run contains 336 auditable per-state KL facts and no event-quality
  defects;
- the English SVG and 300-DPI PNG show v0→v7 with unchanged support data;
- the result note reports the v6→v7 value, sensitivity, provenance, limits,
  and reproduction command without calling the metric epistemic information
  gain.
