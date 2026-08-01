# Rollman K=4 HL Experiment Runbook

## Experiment Contract

- Candidate role: Rollman (`role_id=0`).
- Primitive decision support: `{0,1,2,3,4}` from the reviewed decision-space YAML.
- Origin: one model-authored interpretable policy generated from the frozen game package.
- Proposal cycle: one planner act, four mechanism-distinct sibling policies from one parent, staged evaluation, and one comparative reducer act.
- Search topology: one active linear successor; no persistent four-way tree.
- Policy form: interpretable Python including if/else rules, graph search, planning, state machines, tables, and finite memory. Source size is not penalized.
- Curriculum: freeze an empirical hardest-to-easiest Ghost order from origin certification, then target the easiest opponent not yet certified.
- Completion: one official champion passes every valid human Ghost under the frozen certification protocol with no incomplete match.

## Local Configuration

The API credential stays in the ignored repository-root `.env`:

```dotenv
AGENTBENCH_API_KEY=...
```

The machine-local repository root is supplied at launch:

```bash
export AGENTBENCH_SAST_ROOT=/absolute/path/to/SAST
```

No experiment prompt, event, version, replay, report, or frozen configuration contains the credential.

## Validation

```bash
python -m agentbench_frame.hl.cli validate \
  --config configs/hl/29_rollman-k4.yaml

python -m agentbench_frame.hl.cli audit \
  --config configs/hl/29_rollman-k4.yaml

python -m agentbench_frame.hl.cli prepare-opponents \
  --config configs/hl/29_rollman-k4.yaml
```

## Start and Resume

```bash
python -m agentbench_frame.hl.cli run \
  --config configs/hl/29_rollman-k4.yaml \
  --run-dir .agentbench/29_rollman/runs/RUN_ID
```

The run is open-ended. To execute a bounded number of proposal cycles for an
operator checkpoint, add `--acts N`. Continue the exact run with:

```bash
python -m agentbench_frame.hl.cli resume \
  --config configs/hl/29_rollman-k4.yaml \
  --run-dir .agentbench/29_rollman/runs/RUN_ID \
  --acts N
```

`--acts` counts proposal cycles at the outer-loop interface. Each K=4 cycle
contains six provider acts: planner, four coding candidates, and reducer.

## Context and Token Control

The bootstrap act reads the complete authoritative rules, decision space, and
Replay Skill. Proposal cycles use:

- deterministic `context/game_digest.json` and the immutable package hash;
- bounded `research_state.json`;
- bounded replay summaries and at most two targeted trace investigations;
- one structured branch brief per coding candidate.

An act reads an authoritative rule section only when its hypothesis depends on
precise semantics. Correctness does not rely on hidden provider conversation
memory. Each prompt and model JSONL response is stored in the run.

## Selection and Rollback

Every valid candidate is ranked lexicographically by target points, mean score
margin, worst score margin, level progress, survival, captures, behavioral
novelty, and stable branch index. A candidate becomes the next search parent
only when it improves this ordering over the current parent. Otherwise the
current parent remains active; the harness does not jump back to the origin.

The official champion is a separate pointer. It advances only after target
certification and locked-opponent regression checks pass. Every version remains
immutable and can be checked out explicitly.

## Primary Outputs

```bash
python -m agentbench_frame.hl.cli report \
  --run-dir .agentbench/29_rollman/runs/RUN_ID
```

The primary plot contains four panels over integer `proposal_cycle`:

1. behavioral information gain: epsilon-channel local policy KL over Rollman actions;
2. Rollman Elo on the fixed reporting panel;
3. full human-pool win rate on the fixed reporting panel;
4. mean Rollman-minus-Ghosts score margin on the fixed reporting panel.

`branches.csv` stores the four rollout candidates at the same integer cycle.
`curves.csv` stores the active linear successor. `matches.csv`,
`curriculum.csv`, `events.jsonl`, provider JSONL, versions, replays, traces,
measurements, source audit, opponent order, and context hashes provide the
complete audit trail.

## Pause Conditions

Provider failure, exhausted credit, incomplete opponent execution, malformed
game output, incomplete reporting panel, or measurement failure pauses the run
without declaring scientific success. Resume uses the event log, immutable
versions, and explicit research state.
