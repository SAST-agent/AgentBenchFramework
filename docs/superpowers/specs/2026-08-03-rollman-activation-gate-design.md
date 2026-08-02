# Rollman Candidate Activation Gate Design

## Objective

Prevent paid matches for candidate programs that do not change any Rollman decision on the current parent's learning trajectories, while preserving the existing K=4 linear search, rollback, evaluation, and certification rules.

## Design

The controller accepts an optional game-specific activation probe. After a candidate snapshot is created and before quick-screen matches start, the probe replays the parent's ordered visible states through the parent and candidate policies. It returns the total decisions, changed actions, changed fraction, and bounded per-episode evidence.

A successful probe with zero changed actions makes the candidate ineligible for game evaluation. The candidate remains an immutable historical version and receives a structured `candidate_activation_measured` event. Probe failure is fail-closed: the candidate is not evaluated, because silently paying for an unverified candidate would defeat the budget guard. Bootstrap and non-staged evaluators remain unchanged.

Candidates with at least one changed action continue into the existing quick screen. The gate does not judge whether a change is strategically good; real matches retain that responsibility. This keeps deterministic action divergence separate from reward, information gain, Elo, and win rate.

## Repair Context

Repair packets retain exact replay and trace paths but also embed bounded replay-summary text. Repair prompts instruct the coding model to use the packet's embedded evidence directly. This removes the need for directory traversal or path-discovery scripts and keeps the provider access guard strict.

Candidate acts receive the same bounded-context treatment through one per-branch `candidate_input` artifact. It embeds the compact game digest, research state, Experience Skill, replay summaries, measurements, and shared opponent distillation while retaining exact trace paths for at most two authorized windows. The first coding-agent call reads this single artifact instead of issuing separate reads for every static input.

## Data and Events

`candidate_activation_measured` records the iteration, act, branch, parent and candidate version IDs, status, decision count, changed action count, changed fraction, and per-episode summaries. A zero-change rejection uses the machine-readable error `no_parent_trace_action_change`.

## Verification

Tests prove that zero-change candidates cause no evaluator call, changed candidates do reach quick screening, probe failures fail closed, activation events are persisted, and repair packets contain bounded inline summaries. Existing controller, Rollman measurement, provider isolation, reporting, and full regression suites must remain green.
