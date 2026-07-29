# Generals HL v6 production protocol

Date: 2026-07-29

Status: frozen protocol, preflight-hardened after a zero-act Task 8 attempt;
no retry has started

## Scope and canonical defaults

This is the approved legacy `v5 -> v6` macro-planner pilot. The authoritative
canonical-default list is
`AgentBench/backend_sources/corpus/28_generals/README.md`; this operational
mirror must remain consistent with it:

1. initialize editable policies from scratch;
2. learn each iteration from replays against the strongest frozen human;
3. keep code explainable without limiting it to if/else;
4. use compression and refactoring instead of indefinite rule stacking;
5. maintain an experience Skill;
6. retain immutable historical versions for audited rollback.

V3-v6 are a legacy exception only to the from-scratch clause because their
continuation lineage was approved before this canonical default. They do not
waive the other five defaults or redefine future experiments.

## Frozen inputs and exact invocation

Run from:

```text
/home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl
```

The production invocation for Task 8 is:

```bash
.venv/bin/python -m agentbench_frame.cli generals iterate-v6 \
  --agentbench-root /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets \
  --manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml \
  --learning-manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/v6-strongest-learning-v1.toml \
  --replay-skill backend_sources/corpus/28_generals/skills/replay-analysis-v2/SKILL.md \
  --parent-run /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-hl/20260729_0818_e6bcb9b3 \
  --expected-parent-hash facd39c8a0c064823da68f3dd3eba4a832c08815f75369407112ca087f1ecd9e \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data \
  --codex-executable codex \
  --provider-timeout 1800
```

The command must verify the parent source/manifest and expected hash before
running any learning game or invoking Codex. It must freeze and hash the v2
skill bytes placed in the provider prompt. The human-authored v1 remains
unchanged; the Task 7 validation hashes are:

- v1 entry:
  `f6f3a6eadfd092d476e73c034bce9452b1c548d80d3ce0d4ebefdc9865799df4`;
- v2 entry:
  `0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48`;
- v2 schema reference:
  `5ed24116a502de5a8d31d013f9e5ddc14e1686c20eb1d64d42e1858a98180711`.

### Preflight hash correction

The selected parent v5 hash is
`facd39c8a0c064823da68f3dd3eba4a832c08815f75369407112ca087f1ecd9e`.
It is independently present in `versions/v5/manifest.json`,
`coding_agent_act.snapshot_content_hash`, and `version.manifest_hash`. An
earlier draft carried `17356a...`, which belongs to the historical v1 source;
it is superseded and must not be used for this v6 lineage.

### Role-scoped static-context preflight

The exact production static bundle is pinned as follows:

- v5 `strategy.py`:
  `18b4560ef80d802d06c88acbfe1e790a07085a9a46a8fec0020550b81ffbe975`;
- v5 `EXPERIENCE.md`:
  `c9a6d800fcca6f8091bc6221e35b61655715860f5129937ceb898f9cbaf6fc0d`;
- official rules:
  `9138bb16a27714b9be2403a28bac3f81d2de5b667a7eec1dbb3198887f516950`;
- replay-analysis-v2 entry:
  `0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48`.

After manifest/hash verification and before creating or calling the learning
evaluator, the pipeline applies one public role-aware static validator. V5
strategy, official rules, and the replay Skill remain under the strict
historical/formal/validation denylist. Only the exact V5 experience role may
retain declared round-5 learning replay/state-ID citations for seeds
286101/286202/286303. That role still rejects formal seeds, validation seeds,
current-v6 seeds, undeclared historical seeds, arbitrary six-digit seeds, and
whole-word formal/validation material. Final prompt construction reuses the
same public validator. The dynamic learning action profile remains checked
after learning but before any provider act.

## Split isolation

Only the following six v5 games are learning evidence:

- opponent: `advanced-rank02-robinliu-v18` (high/strongest);
- seeds: `287101`, `287202`, `287303`;
- seats: `0`, `1`.

The single provider prompt may contain the verified v5 source and experience,
official rule summary, frozen replay-analysis-v2 entry, the six learning
critical windows/dense summaries, and the learning action profile. It must not
contain any validation or formal case, replay, state, outcome, score, dense
trace, critical window, or action profile.

The inherited V5 experience is not new round-6 candidate evidence. Its
declared round-5 learning replay/state-ID citations are permitted only in that
role and do not expand the exact round-6 evidence set below.

Validation is the 12-case high/medium matrix using seeds
`288101/288202/288303` and both seats. Formal evaluation is the unchanged
18-case high/medium/low pilot matrix. Both are **post-act only**: create them
only after the provider act completes and the v6 source, manifest, source hash,
prompt bytes, and prompt hash are frozen. Validation never gates saving v6 or
running formal evaluation; every runnable v6 receives all 18 formal cases.

Task 7 replay-skill baseline and forward tests are Skill validation. They
count as neither a v6 provider act nor v6 provider/gameplay budget.

### Attempt-1 preflight postmortem

Attempt 1 is retained immutably at run `20260729_1555_c23d1075`. It completed
six learning episodes, made zero provider acts, and failed before creating a
v6 source, provider output directory, or score. Because the old static bundle
was ordered strategy, experience, rules, then Skill, the first rejected
`286101` came from the exact V5 `EXPERIENCE.md`: a context-agnostic gate had
incorrectly treated its legitimate round-5 learning citation as candidate
round-6 evidence.

The old replay Skill was independently contaminated by a real forward-test
opponent/match identity and whole-word held-out phase material. The assets
cleanup remains required and is retained at commit `5d085bb`; it was not the
source of the first reported `286101`. The failed run was not modified, and
neither `iterate-v6` nor `recover-v6` was run during either preflight fix. No
retry has started.

## Recovery semantics

Recovery always creates a new run linked to the failed run; it never mutates a
finalized run.

```bash
.venv/bin/python -m agentbench_frame.cli generals recover-v6 \
  --agentbench-root /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets \
  --manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml \
  --learning-manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/v6-strongest-learning-v1.toml \
  --replay-skill backend_sources/corpus/28_generals/skills/replay-analysis-v2/SKILL.md \
  --parent-run /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-hl/20260729_0818_e6bcb9b3 \
  --expected-parent-hash facd39c8a0c064823da68f3dd3eba4a832c08815f75369407112ca087f1ecd9e \
  --failed-run /absolute/path/to/FAILED_V6_RUN \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data \
  --codex-executable codex \
  --provider-timeout 1800
```

- A `prompt_incomplete` zero-act run may reuse only verified learning inputs;
  recovery rebuilds and rehashes the prompt.
- A provider failure retry is a new visible act and increments cumulative act,
  token/tool/time, and learning-budget accounting.
- A post-act recovery verifies the saved runnable v6 source/manifest/prompt
  lineage, performs zero new provider acts, and reruns the complete validation
  and formal matrices in the new recovery run.
- Any parent, v6, prompt, learning-ID, or terminal-status mismatch is fatal and
  remains visible.

## Expected artifacts

Before the act:

- `run.toml`, learning/validation/formal specs, events, summary, and quality;
- six learning match directories with metadata, official/normalized replay,
  dense trace/summary, protocol/process/stderr bytes;
- copied v5 source and manifest plus lineage metadata;
- frozen v2 skill receipt and hash.

At and immediately after the act:

- prompt markdown/JSON/manifest/hash;
- raw provider JSONL, stderr, token/tool/time records, and feedback receipt;
- provider-mutated workspace, candidate test log, scope audit;
- frozen v6 source/manifest/hash and `v5-to-v6.patch`.

Post-act only:

- 12 validation and 18 formal match artifact families;
- learning/validation/formal action profiles;
- score, gain, AUC/decision-space/information-gain, local and cumulative budget
  records;
- final `summary.json`, `quality.json`, and event audit.

## Audit checklist

- [ ] Both repositories are clean at their recorded Task 7 commits.
- [ ] Parent source and manifest match the explicitly reconciled expected hash.
- [ ] The exact four-file static bundle matches the pinned hashes and passes
      role-scoped validation before any learning evaluator or gameplay.
- [ ] V2 skill hash matches the prompt receipt; v1 hash remains unchanged.
- [ ] Exactly 6 learning, 12 validation, and 18 formal cases are saved.
- [ ] Learning, validation, and formal seed sets are disjoint.
- [ ] One normal-run provider act is visible; every retry is separately counted.
- [ ] Prompt manifest contains only the six learning episodes.
- [ ] V6 scope audit allows only declared strategy/docs/tests/policy paths and
      preserves `main.py`.
- [ ] Candidate tests pass before validation/formal gameplay.
- [ ] Validation failure cannot suppress formal evaluation for runnable v6.
- [ ] Every match has the required replay/protocol/process artifact family.
- [ ] Prompt/source/skill hashes recompute; action profiles and budgets exist.
- [ ] `quality.json` and events expose every malformed, missing, or invalid state.

## Interpretation limits

The formal success checks are medium at least `2/6` and low exactly `6/6`;
high at least `1/6` is an additional breakthrough. Every runnable result is
published even when it misses them.

Score changes, dense metrics, action histograms, end-only counts, upgrade
counts, and neutral-plain ratios are descriptive evidence. They do not prove
that macro planning caused a score change. Deterministic action disagreement
is not policy KL, and diagnostics are not epistemic information gain. Missing
or invalid values remain missing; no gap is converted to zero or interpolated.
