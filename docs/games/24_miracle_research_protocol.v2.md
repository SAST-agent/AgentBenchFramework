# 24_miracle research protocol v2

Target-contract authority:
[24_miracle KL contract authority v2](24_miracle_kl_contract_authority.v2.md).
The migrated contract is `KL(new||old)`, natural logarithm, symmetric uniform
measurement smoothing with fixed main `epsilon=0.01`, actual new-policy
occupancy, ordered local trace, and arithmetic-mean episode information gain in
`nats / decision`. The distinct optional sum uses `nats / episode`. There is no
KL acceptance threshold.

The research/measurement contract has protocol version
`24-miracle-research-v2`. The separately versioned frozen 72-case evaluation
schema has benchmark version `24m-frozen-v1`. The manifest emits both values:
changing measurement semantics requires a protocol-version change, while
changing the frozen population or case identities requires a benchmark-version
change.

`tools/miracle_research_manifest.py` emits deterministic canonical UTF-8 JSON
(sorted keys, compact separators, one trailing LF). A future Results research
run must copy those exact bytes to `research_manifest.json` in the run
directory and record their SHA-256, `protocol_version`, and
`benchmark_version` in `summary.json`. Results validates the submitted file;
it does not recreate the Framework seed algorithm.

The run-local file and summary reference are integrity evidence, not approval.
Results maintains a separate, version-controlled allowlist from benchmark
version to approved canonical manifest digests. A run cannot add its own digest
to that allowlist, and run data must never update it automatically. The
production allowlist is currently empty.

The current generated manifest remains `BLOCKED_NOT_AUTHORITATIVE` with
`authoritative_ready=false`, non-empty blockers, and unqualified rank02,
rank10, and rank16 audits. Canonical serialization does not change that status.

An authoritative approval requires this ordered process:

1. Implement and review the real old/new-policy binding and Judge seed
   integration, then qualify rank02, rank10, and rank16.
2. Generate a new canonical manifest whose internal status is
   `AUTHORITATIVE_READY` and whose blockers and audits support that claim.
3. Human-review the relevant source code, asset hashes, random controls, case
   identities, seed bundles, and exact canonical manifest bytes.
4. Through a separate reviewed Results code/configuration change, register the
   canonical digest in the approved allowlist for its benchmark version.
5. Only after registration may Results accept a `complete` run, and only when
   its remaining schema, case, seed, and KL evidence also passes.

The canonical blocked digest changes with this protocol migration. It is for
blocked reporting only and must not be placed in the complete allowlist. Fake
qualified manifests used by tests are likewise test-only and must never be
registered as production approvals.

Status: **pre-registered, blocked for authoritative execution**.

This protocol replaces neither the historical Plan A run nor its evidence.  It
defines the next no-backpropagation heuristic-learning experiment and its final
frozen evaluation.

## Population

- train (10): rank01, rank03, rank04, rank06, rank07, rank08, rank11, rank12,
  rank14, rank15
- validation (3): rank05, rank09, rank13
- test (3): rank02, rank10, rank16

Test opponent source, behavior, replay, intermediate scores, and artifacts must
not be exposed to the coding agent during learning or validation.

## Frozen test matrix

One case is identified by:

```text
opponent version × evaluated-agent camp × map type × day time × repeat
```

The final matrix contains:

```text
3 test opponents × 2 camps × 2 maps × 2 day states × 3 repeats = 72 cases
```

`research_protocol.py` deterministically derives a seed bundle for every case.
The logic seed is selected so that the Judge's first two `random.randint(0, 1)`
draws equal the frozen map and day values.  An authoritative run must also prove
that the evaluated agent and opponent either consume their assigned seeds or
are deterministic.  An opaque unseeded process is not eligible for the frozen
test matrix.

Static audit on 2026-07-27 found that the current real match bridge still
starts the Judge as `python main.py`; it does not invoke
`seeded_python_entry.py`. More importantly, the compiled rank02 policy seeds
`rand()` from `clock()+time(0)`, while rank10 terminates search according to the
process CPU clock. The Python launcher cannot control either C++ source.
Rank16 contains no active RNG call found by the audit, but its deterministic
qualification remains open because its retained compiler/runtime risk has not
been closed. Consequently, the 72-case manifest is a preregistration only; it
is not yet an executable authoritative frozen matrix.

The complete JSON-ready case list can be inspected without starting a runtime:

```text
python tools/miracle_research_manifest.py
```

## Heuristic learning boundary

HL optimization in this experiment performs no backpropagation or gradient
updates.  Rule editing, scoring, path search, planning, symbolic reasoning, and
controlled random heuristics are permitted.  Every coding-agent act produces a
retained version; lower-performing versions are not silently rolled back.

### Versioned iteration protocol

The frozen benchmark case identity remains unchanged. The separate
iteration mechanism has version `24-miracle-iteration-v4`; its canonical
description is produced by `iteration_protocol_manifest()` and records the
version and SHA-256 of the checked-in `24m-minimal-bootstrap-v1` template.

The fake-only lifecycle is explicit and ordered: `PLANNED`, `MATCH_READY`,
`BASELINE_STORED`,
`REPLAY_CAPTURED_UNAPPROVED`, `REPLAY_APPROVED`, `LEARNING_READY`,
`CANDIDATE_STORED`, `EVALUATION_READY`, and finally `COMPLETED` or
`INCOMPLETE`. A canonical `MatchPlanManifest` is independently approved before
a runner factory can be opened. It binds the champion, human replay-reading
Skill, evaluated policy and source digest, bootstrap, protocol/research
identities, and every planned case role, camp/map/day/repeat value and seed
bundle. It deliberately contains no future replay.

After MatchPlan approval, the fixed bootstrap is materialized as
`strategy-v0`, written once to the immutable store, loaded back, and checked
byte-for-byte and by source SHA against the plan before the match runner
factory is called. The factory receives the stored version, source SHA, store
identity, and frozen cases. A missing, changed, or differently approved
baseline prevents the factory from opening; saving v0 after a match cannot
satisfy this gate.

After a fake match stage, captured artifacts form a separate canonical
`ReplayEvidenceManifest`. It references the approved match-plan digest and
must cover every planned case exactly once. Artifact path and bytes, case,
role, seeds, champion, human Skill, and evaluated policy must still match the
plan. Missing, extra, duplicate, or drifted cases fail closed. Replay evidence
receives a second independent approval before any learning/controller factory
may read it. Both production approval sets are empty.

New research defaults to a from-scratch v0 created from that template. The
template contains only the observation/action adapter, complete legal-action
handling, and a minimal legal fallback. It contains no copied if-else tactics
or opponent knowledge. The previous v0/v1 smoke entry is available only as
the explicit `--legacy-smoke` compatibility mode; its observed 0.5 to 0.5
score was not an improvement and is not authoritative evidence.

Before any runner, session, or log is created, normal preflight requires an
explicit frozen human-champion descriptor with logical ID, version, artifact
SHA-256, provenance, qualification evidence, and compatible protocol and
benchmark versions. There is no rank04, rank09, filename-based, or other bot
fallback. The full canonical descriptor digest, rather than its self-declared
qualification field or artifact digest alone, must equal the single current
digest in the version-controlled production approval boundary. That value is
currently `None`, so production execution remains blocked.

Human replay reading and Agent-authored experience are separate schemas. A
canonical human replay-reading Skill must receive independent digest approval;
the experience Skill can only reference that digest and cannot replace it with
a `human_authored=true` flag. The approved human replay-Skill, match-plan, and
replay-evidence sets are also currently empty. Tests exercise future success only by
temporarily monkeypatching these module constants; no fake digest is production
approval state, and no CLI, run file, manifest, or ordinary Python function can
override an approval table.

The default CLI is read-only and has separate `plan`, `replay`, and `learning`
preflight commands. It never starts a Judge, opponent, provider, match, or
session. With the empty production approvals each command returns a fatal
blocked result without creating output. The opt-in `--legacy-smoke` route
remains deprecated and non-authoritative. Only evidence whose frozen role is
`train` can enter a prompt, change plan, or experience Skill; renaming
validation evidence cannot change its role.

One HL/IG decision is exactly one legal atomic Judge command. Multi-command
macro-actions and illegal commands are outside ActionSupport. This iteration
protocol now validates the v2 formal profile: fixed epsilon `0.01`,
`KL(new || old)`, natural logarithm, symmetric uniform smoothing, actual
new-policy rollout, ordered decision records, and the arithmetic-mean episode
IG. Generic research KL remains separately available but cannot acquire the
formal profile or populate formal IG fields.

Every strategy version is a canonical immutable UTF-8 JSON record containing
the complete source snapshot, entrypoint, dependency list, source SHA-256,
structured change plan, interpretability category, complexity metrics, and
read-only probability-query capability. Allowed interpretable forms include
rule tables, finite-state machines, deterministic heuristic planners, scoring
functions, and structured combinations. Network dependencies, unregistered
external state, symlinks, and opaque binary models fail closed.

Dependencies are individually registered against a small module allowlist;
`python-standard-library` is not an authorization. Dynamic import, `eval`/
`exec`/`compile`, unregistered file reads, process or system launch, and unsafe
modules fail static validation. Strategy, Skill, version, iteration, and replay
logical IDs are path-safe, and every resolved read/store target must remain
inside its declared root. This static validation is not a runtime sandbox and
does not authorize an authoritative run; a separately audited sandbox remains
required before real execution.

Change plans distinguish `add`, `replace`, `merge`, `delete`, and `simplify`.
Complexity records effective nodes, source lines, duplicate or shadowed rules,
and maximum decision depth and is deterministically computed from the complete
source snapshot. Candidate-reported values must match. Before any store path is
created, every non-v0 version requires independently validated train evidence,
loads its exact parent, validates its real structured operations, compares
computed metrics, and applies the growth gate. Complexity growth additionally
requires an explicit reason.
Validation replay, validation score, and final test conclusions are never
included in the modification payload. `strategy-v0` has no parent and must
exactly materialize the fixed bootstrap as `main.py`.

The Miracle experience Skill uses schema
`24-miracle-experience-skill-v2`, separate from both the generic `generals`
replay Skill and the human replay-reading Skill. It distinguishes frozen facts,
training observations, supported heuristics, failed/revoked heuristics,
unverified hypotheses, approved train-evidence digests, strategy references,
case identities, authors, and an immutable version SHA. `experience-v0` is the
fixed empty baseline: it records no observations, heuristics, hypotheses, or
training evidence and binds the approved human reader plus the same complete
iteration identity as the bootstrap. Every later Skill requires an existing
same-identity parent and non-empty case/evidence references drawn only from the
currently approved train replay.
Candidate readiness binds a same-generation pair such as `strategy-v1` and
`experience-v1`, including both canonical SHA-256 values and their common
parents and control identity. The Experience Skill must reference that exact
candidate strategy and the approved training evidence. A strategy without its
corresponding non-v0 Skill is not a candidate bundle.
Each round reads the last approved version and may append a candidate version;
in-place replacement is forbidden. The required human-authored Miracle replay
reader has not been supplied and remains explicitly
`HUMAN_AUTHORED_CONTENT_REQUIRED`. Agent-generated text cannot satisfy it.

Rollback is append-only and the target must be an ancestor of the source; a
sibling or unrelated branch cannot be selected. It first revalidates the source and target version
bytes/SHA, bootstrap, champion, human replay Skill, match plan, replay-evidence
manifest, artifact bytes, and all current approval tables, then writes a new rollback
iteration record containing source, target, reason, and operator. It never
deletes or changes later versions. Missing, tampered, or incompatible inputs
fail before a rollback record or runtime resource is created.

Candidate evaluation is a separate lifecycle, not reuse of the initial v0
MatchPlan. A canonical `CandidateEvaluationPlan` binds the baseline and
candidate policy versions/source digests, Experience Skill, champion, human
replay-reading Skill, training provenance, research/iteration identities,
purpose, and frozen evaluation case/seed bundles. Its production approval set
is empty. Fake-only tests may temporarily approve a digest with pytest
monkeypatch; that state is restored after the test.

Evaluation produces a separate canonical `EvaluationEvidenceManifest` with
exactly one baseline/candidate replay-result pair for every planned evaluation
case. It binds artifact bytes, outcomes and scores, policy and Skill identity,
strict trajectory KL evidence, information gain, and failure reasons. Missing,
extra, duplicate, seed-drifted, policy-drifted, or tampered evidence fails
closed. Evaluation cases and artifacts never enter the training prompt,
change plan, or Experience Skill.

There is no boolean completion API. `IterationAcceptanceManifest` derives its
status from the independently approved evaluation plan and evidence: all cases
must be complete, and the validator requires the exact formal profile,
epsilon `0.01`, `KL(new || old)`, new-policy rollout and full ActionSupport
distributions. Every local value, mean and sum is recomputed. IG must be
present, score gain must be positive, and blockers must be empty. The
acceptance manifest itself
also needs independent digest approval. Complete fake evidence is reported as
`fake_only=true`; all three production approval sets are empty, so production
`COMPLETED` remains unreachable. Its Results summary mapping retains raw/evo/
gain, IG, incomplete reasons, protocol/research/iteration identities, policy,
Skill, training/evaluation plan, evidence, KL-evidence, and acceptance SHA
fields without changing Results validation.

All mutable factories and stores perform their own stage preflight or fully
revalidate an internally issued context immediately before side effects. A
publicly constructed or forged dataclass is not authorization. Contexts bind
the original manifest and asset paths, canonical digests, and approval state;
post-preflight replacement of the champion, human Skill, match plan, replay
manifest, replay artifact, or bootstrap therefore fails with zero new writes.

Current status is deliberately split as follows:

- mechanism implemented: yes;
- fake-only verified: yes, by temporary fixtures only;
- human input required: champion asset and human-authored replay Skill;
- authoritative execution blocked: yes;
- authoritative completed: no.

The final HL checklist cannot be marked complete until a separately approved
real experiment retains complete replays, auditable strategy changes, old and
new immutable versions, at least one valid score improvement, complete real
KL/IG, and score-iteration and IG-iteration curves.

## Policy information gain

- measurement profile: `24_miracle_policy_information_gain_v2`
- epsilon: `0.01`
- direction: `new||old`
- smoothing: symmetric uniform mixture on the complete current support
- rollout: actual new-policy rollout
- primary episode aggregate: arithmetic mean, `nats / decision`
- optional local-KL sum: `nats / episode`
- decision: one atomic Judge command (`init`, `move`, `attack`, `summon`, `use`,
  `endround`, or `surrender`)
- decision-change rate: not collected

The game adapter must provide the complete legal atomic-command support and
strict new/old distributions on the same visible state.  A deterministic HL
policy reports a true one-hot distribution.  Random behavior must report its
actual categorical distribution and must not be disguised as one-hot.
Framework applies the same epsilon channel to both distributions only for
measurement; it does not change the new policy action executed by the Judge.

If support completeness or either distribution is unavailable, the episode is
`incomplete`; trace-derived information gain and sum remain missing rather than
zero. Uploaded local or episode summaries are checked against values recomputed
from the ordered distributions and cannot override them. `occupancy_shift`
remains separate and is never added to information gain.

If a selector depends on internal memory, authoritative context is `z=(s,m)`.
Current evidence records visible-state identity but not canonical `m`, so a
stateful real policy has
`internal_memory_evidence=not_collected`. The production iteration preflight
therefore derives an incomplete formal IG until an H08 integration binds
`m` to the policy/context identity. Tests can inject the separately named
test-only bound state; uploaded run data cannot override this control-plane
state.

`Run.log_policy_kl_trace()` remains readable for old events, including records
with `epsilon=None`, but writes `measurement_profile=legacy_policy_kl_trace`,
`information_gain=null`, and `information_gain_status=unverified`. Only a rich
v2 payload whose frozen decision distributions and all derived fields reverify
can populate formal IG in tracking and reports.

The authoritative observation frame is six ASCII decimal length bytes followed
by UTF-8 JSON. `decode_ai_observation()` validates that frame and
`enumerate_legal_commands()` derives finite legal support from the decoded
dynamic state plus fixed map geometry. The authoritative Judge accepts
WindBlessing at any cube coordinate without an in-map check; that support is
unbounded, so the enumerator deliberately fails closed for such a state. The
frozen if-else configuration selects InfernoFlame unless explicitly overridden.

`IsolatedDeterministicPolicy` is a fake-testable deep-copy boundary for a pure
deterministic selector. It is not yet a binding to the frozen
`ifelse_bot/main.py`: that program's `AiClient` methods write a command to
stdout and then consume the next observation. A separately reviewed adapter is
still required to intercept exactly one command from an isolated copy without
submitting it or advancing either old/new policy session. Until that binding is
implemented and verified, real-policy trajectory KL remains incomplete.

The Judge preserves the submitted `creatures` order in the player's capacity
list and exposes that order in later observations.  Init support therefore
contains all `4 * P(7, 3) = 840` ordered legal card selections; permutations are
distinct canonical actions.

## Results integration boundary

Framework's Results pipeline checks load the repository named by
`AGENTBENCH_RESULTS`.  In a two-worktree integration environment they exercise
that exact aggregate/report implementation.  In an independent Framework
checkout without the environment variable and Results files, these optional
cross-repository checks skip with an explicit dependency reason; zero skips is
not an independent-checkout guarantee.

## Execution boundary

This document and its tests authorize no Judge, opponent, Provider, real match,
matrix, or authoritative session.  Runtime execution requires separate approval
after the adapter and Results CI pass fake-only acceptance.
