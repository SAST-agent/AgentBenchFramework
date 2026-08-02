# Rollman Candidate Activation Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Skip paid Rollman matches for candidates that do not change an action on the current parent's learning traces and make repair evidence directly readable without directory traversal.

**Architecture:** Add a pluggable pre-evaluation callback to the game-neutral controller and implement it with the existing Rollman policy probe. Persist structured activation evidence and fail closed on zero-change or probe failure. Inline bounded summary text in the existing repair packet.

**Tech Stack:** Python 3.11, pytest, existing VersionStore, HLEventWriter, Rollman policy probe.

## Global Constraints

- Preserve K=4 linear search; do not build a tree.
- Do not change rollback, curriculum, IG, Elo, win-rate, or certification semantics.
- A deterministic action is compared directly; no invented tactic hypothesis space or model probability is used.
- Do not weaken provider filesystem isolation.

---

### Task 1: Controller activation boundary

**Files:**
- Modify: `src/agentbench_frame/hl/controller.py`
- Test: `tests/hl/test_controller.py`

**Interfaces:**
- Consumes: `activation_probe(new_version, parent_version, parent_evaluation)`.
- Produces: structured `candidate_activation_measured` events and pre-evaluation rejection.

- [x] Write tests where a zero-change result prevents `quick_screen` and a changed result permits it.
- [x] Run the focused tests and confirm they fail for the missing callback boundary.
- [x] Add the optional callback, fail-closed error handling, event emission, and `CandidateResult.activation` metadata.
- [x] Run the focused controller tests and confirm they pass.

### Task 2: Rollman action-change probe

**Files:**
- Modify: `src/agentbench_frame/games/rollman/research_measurement.py`
- Modify: `src/agentbench_frame/hl/cli.py`
- Test: `tests/rollman/test_research_measurement.py`

**Interfaces:**
- Consumes: parent evaluation traces and two immutable versions.
- Produces: `status`, `decision_count`, `changed_action_count`, `changed_fraction`, and `episodes`.

- [x] Write a failing test using a fake probe runner with one changed action.
- [x] Add `measure_activation` using ordered trace states, quick-screen seeds, exact action comparison, and cached parent probes.
- [x] Wire the callback into the Rollman controller construction.
- [x] Run focused measurement and CLI/controller tests.

### Task 3: Bounded repair evidence

**Files:**
- Modify: `src/agentbench_frame/hl/repair.py`
- Modify: `src/agentbench_frame/hl/context.py`
- Test: `tests/hl/test_repair.py`
- Test: `tests/hl/test_context.py`

**Interfaces:**
- Consumes: trusted summary paths returned by `summary_resolver`.
- Produces: `summary_text` capped at 12,000 characters per match.

- [x] Write a failing packet test asserting inline text and truncation.
- [x] Implement bounded UTF-8 summary loading while retaining exact paths.
- [x] Update the repair prompt to forbid first-call summary discovery.
- [x] Run focused repair/context tests.

### Task 4: Regression and experiment handoff

**Files:**
- Verify: `tests/`

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: a verified framework commit ready for the next paid proposal cycle.

- [x] Run all tests with the required sandbox permissions.
- [x] Generate the current run report and verify integer iteration axes and trustworthy metrics.
- [x] Commit the implementation without API credentials or run artifacts.

### Task 5: Single-read candidate input

**Files:**
- Modify: `src/agentbench_frame/hl/proposal.py`
- Modify: `src/agentbench_frame/hl/context.py`
- Modify: `src/agentbench_frame/hl/cli.py`
- Test: `tests/hl/test_proposal.py`
- Test: `tests/hl/test_context.py`

**Interfaces:**
- Consumes: compact context files, replay summary paths, branch brief, and measurements.
- Produces: one bounded `candidate_input-bNN.json` artifact referenced by the candidate prompt.

- [x] Write failing tests for bounded inline context and the one-read prompt contract.
- [x] Implement deterministic packet creation and CLI wiring.
- [x] Run focused prompt/proposal/controller tests.
- [x] Run the complete regression suite.
