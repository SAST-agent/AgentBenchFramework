# Generals Complete Macro-Action Space and Strict Policy-KL Design

**Date:** 2026-07-30
**Status:** Approved design; implementation has not started

## 1. Objective

Complete the two missing Generals research measurements without changing any
historical gameplay result:

1. define a normalized, complete, state-conditioned legal macro-action space
   \(A(s)\); and
2. calculate strict uniform-epsilon policy KL for every adjacent historical HL
   version from v0 through v6 on one frozen reference-state distribution.

The implementation must follow `附件/information_gain.md`: both policies use
the same legal support, deterministic HL is mixed with the uniform
distribution \(U_s(a)=1/|A(s)|\), natural logarithms produce nats, missing data
remain missing, and policy KL is behavioral change rather than epistemic
information gain.

The existing v0-v6 runs remain immutable. Their current `null` KL values are
historically correct because those runs did not contain the required action
support or legality state. New values live in a separate, provenance-linked
historical measurement run.

## 2. Scope and non-goals

This design covers:

- the official Generals Python engine;
- a complete canonical behavioral macro-action grammar;
- exact support cardinality on a controlled real-replay state set;
- deterministic HL-to-HL KL for v0-v6;
- append-only measurement artifacts, summary fields, and CI presentation; and
- conformance, differential, regression, and end-to-end tests.

This design does not:

- change v0-v6 policy source or scores;
- run another Codex policy-improvement act;
- claim full-episode or trajectory KL;
- estimate support sizes by sampling or truncation;
- assign semantic distance to two different macro-actions;
- call policy KL epistemic information gain; or
- produce a Generals RL KL value before an RL policy exposes a distribution on
  the same complete canonical support.

## 3. Frozen historical policy lineage

The measurement run imports these exact successful snapshots:

| Version | Source run | Content hash |
|---|---|---|
| v0 | `20260726_1631_f56f789f` | `bfab30cdaaa0ad8e0ac6ed1b9ab047417c621cc22dd22fd95b0946c3af4bd5fa` |
| v1 | `20260726_1631_f56f789f` | `17356a28682378c4877f7da7aeb8ac90b8891e2685012dec8cdec990cb282e79` |
| v2 | `20260727_0639_89578eec` | `1e8a2ce4fa38b4171f6f8d77d42b80d48438f5213b08d543714d8241e292c565` |
| v3 | `20260728_1122_57f647d5` | `a9f27eb2a02ba452e9363022d97ffa5556c0f3a02eef120061e73e93c68ce815` |
| v4 | `20260729_0818_e6bcb9b3` | `5c12e7c92843cbc18b742ab7be71c684cfa0a0aac8e5dbb5a24fc7b905fe959f` |
| v5 | `20260729_0818_e6bcb9b3` | `facd39c8a0c064823da68f3dd3eba4a832c08815f75369407112ca087f1ecd9e` |
| v6 | `20260729_1653_af8eda26` | `974050ee1a3d4b4c4f96e61f5af39b4e52e2cfbc50f146f9e8b4ac96c4ac798b` |

Every imported source tree is copied into the new run, verified against its
manifest, and then treated as read-only. Duplicate and failed historical runs
are not alternative versions.

## 4. Measurement state

The current normalized replay is insufficient for complete legality because
it omits the player's remaining army-movement budget
`GameState.rest_move_step` and each general's remaining movement
`General.rest_move`.

A new `generals-measurement-state-v1` snapshot contains all state used by the
official legality functions, including:

- `GameState.round`, `coin`, `active_super_weapon`,
  `super_weapon_unlocked`, `super_weapon_cd`, `tech_level`,
  `rest_move_step`, `next_generals_id`, and `winner`;
- board-cell position, terrain, ownership, army, general placement, and active
  weapon references;
- complete general type, owner, position, levels, skill cooldowns, skill
  durations, and remaining movement;
- each active weapon's type, player, cooldown, remaining duration, and
  position; and
- the current actor and termination state carried by the harness.

Replay output paths and changed-cell bookkeeping are excluded because they do
not affect legality or transition semantics. The implementation audits every
field read by the official command-1 through command-7 path and locks the
allowlist in a schema contract test. It never guesses a missing field from an
old replay.

Serialization and reconstruction must round-trip:

```text
official GameState
→ measurement snapshot
→ reconstructed official GameState
→ measurement snapshot
```

The first and final semantic snapshots must have the same content hash. This
round-trip is required before either action counting or historical policy
evaluation.

The existing normalized `state_id` algorithm and old replay files do not
change. Measurement snapshots receive a separate
`measurement_state_schema`, `measurement_state_id`, and content hash.

## 5. Canonical action-space abstraction

Framework gains an optional, backward-compatible abstraction:

```python
class CanonicalActionSpace(Protocol):
    spec_id: str

    def canonicalize(self, state, action): ...
    def contains(self, state, action) -> bool: ...
    def cardinality(self, state) -> int: ...
    def iter_actions(self, state): ...  # optional
```

Existing games with a materializable action list continue to use
`get_legal_actions()` and `get_action_distribution()`. Generals uses
`GeneralsMacroActionSpaceV1`; no existing external interface is removed or
renamed.

The action-space specification records and hashes:

- official engine source identity;
- measurement-state schema;
- primitive command grammar;
- parameter domains;
- canonicalization rules;
- terminal semantics;
- exact-count algorithm version; and
- ordering and serialization rules.

The resulting `spec_id` is stored with every state, count, KL record, and
aggregate. Results with different `spec_id` values cannot be combined.

## 6. Complete canonical Generals macro-actions

A canonical macro-action is a sequence

\[
(c_1,\ldots,c_n,[8])
\]

where every primitive \(c_i\) is legal after applying the preceding legal
prefix to a cloned official state. `[8]` is available at every nonterminal
prefix. There is no artificial `MAX_PRIMITIVES` measurement bound.

The finite syntactic candidate generator covers:

1. army movement from every board source, in every direction, for every
   distinct executable positive army amount;
2. every general and every board destination for general movement;
3. every general and production, defense, and mobility upgrades;
4. every general skill and its complete canonical target domain;
5. all four technologies;
6. all four super-weapons and their complete canonical source/target domains;
7. every board position for sub-general recruitment; and
8. end-macro.

Every candidate is executed against an isolated clone through the official
command logic. Only official success enters the legal primitive set \(L(s)\).
The candidate generator is not itself treated as the authority.

Canonicalization collapses raw protocol encodings that have the same official
behavior:

- an oversized army request is represented by the actual distinct executable
  amount;
- ignored surplus parameters on skills are removed;
- parameter lists use their canonical official shapes;
- commands after the first end marker are removed;
- if a primitive ends the game, an unreachable submitted suffix is removed
  and the canonical macro receives one conceptual final `[8]`; and
- zero-effect, malformed, illegal, or nonterminating submitted forms are not
  members of \(A(s)\).

This is a complete support over normalized behavioral macro-actions, not over
redundant raw integer strings.

## 7. Exact support cardinality

Let \(T(s,c)\) be the cloned official transition for legal primitive \(c\).
The number of legal suffix macros is:

\[
N(s)
=
1
+
\sum_{c\in L(s)}
\begin{cases}
1, & T(s,c)\text{ is terminal},\\
N(T(s,c)), & \text{otherwise}.
\end{cases}
\]

The leading one is the immediate `[8]` macro. Different first commands remain
different macro-actions even if they reach the same successor state.

The counter:

- uses arbitrary-precision integers;
- memoizes by action-space spec, actor, and complete measurement-state hash;
- persists resumable on-disk checkpoints;
- detects recursion cycles and fails the specification if a canonical
  successful command can create an unbounded same-turn cycle;
- records expanded states, legal edges, cache hits, wall time, and peak
  resource usage; and
- returns only an exact integer or an explicit incomplete status.

No partial count, bound, sample estimate, or length-truncated count may enter a
KL calculation. Cardinalities are serialized as decimal strings so JSON and
JavaScript cannot lose integer precision. CI may additionally show a derived
`log10_support_size`.

## 8. Frozen reference-state distribution

Reference states are independent of every compared HL version. The collector
is official strongest-human self-play:

- player 0: `advanced-rank02-robinliu-v18`;
- player 1: `advanced-rank02-robinliu-v18`;
- seeds: `289101`, `289202`, and `289303`;
- for each seed and each seat, select that seat's second and tenth decision
  pre-state.

This gives exactly:

\[
3\text{ seeds}\times2\text{ seats}\times2\text{ decisions}=12
\]

primary reference states. Selection is frozen before historical policies run
and cannot depend on score, action disagreement, support size, or KL.

The collector run retains official replay, normalized replay, full measurement
snapshots, both process/protocol/stderr logs, termination metadata, engine
hash, opponent hash, and selection receipt. An invalid collector game makes
the primary measurement run incomplete.

Additional middle- or late-game exact-count attempts may be saved as
exploratory artifacts. They do not change the 12-state primary distribution
and do not enter the main aggregate.

## 9. Historical policy evaluation

For every one of the 12 states, v0-v6 run in isolated fresh subprocesses. The
measurement harness reconstructs the official SDK `GameState`, imports the
frozen `main.agent`, and invokes that unchanged entrypoint with the frozen
round and seat. Each historical `state_view.py` therefore decides exactly
which official fields its policy can observe; the harness does not pass the
measurement JSON directly to `strategy.py` or expose new keys to old policy
code.

Each policy is run twice to verify deterministic output. Source and runtime
audits verify whether it has cross-decision mutable memory. The controlled
measurement context is explicitly \(z=(s,m_0)\), where \(m_0\) is the reset
process state. A nondeterministic, stateful-without-reconstructible-memory,
crashed, timed-out, malformed, or illegal policy result is retained as first
hand evidence and yields a missing KL for affected pairs; the historical
policy is never edited to make the metric complete.

Raw output, canonical macro, legality result, latency, process status, policy
hash, state ID, and action-space spec ID are retained.

## 10. Strict uniform-epsilon HL policy KL

For \(N=|A(s)|\), deterministic action \(f(s)\), and fixed epsilon:

\[
\widetilde\pi(a\mid s)
=
(1-\epsilon)\mathbf 1[a=f(s)]
+
\frac{\epsilon}{N}.
\]

The primary epsilon is `0.01`. Sensitivity values are:

```text
0.001, 0.01, 0.05, 0.1
```

All policy types must use these same values when they are compared on this
action-space specification.

For an adjacent pair \(v_{k-1}\rightarrow v_k\), define:

\[
q=1-\epsilon+\frac{\epsilon}{N},
\qquad
r=\frac{\epsilon}{N}.
\]

If the canonical macros are equal, KL is zero. If they differ:

\[
D_{KL}
\left(
\widetilde\pi_{v_k}
\middle\|
\widetilde\pi_{v_{k-1}}
\right)
=
(q-r)\ln\frac{q}{r}.
\]

The calculation uses natural logarithms. To avoid underflow for large \(N\),
it evaluates the algebraically equivalent expression with an
arbitrary-precision decimal context of at least 80 significant digits and
saves both a decimal nats string and a finite display float.

The main per-transition aggregate is the equal-weight mean over all 12
reference states at epsilon `0.01`. It exists only when all 12 values exist.
Sensitivity aggregates follow the same completeness rule.

This metric is named `controlled_reference_policy_kl`. It is not
`episode_policy_kl`, `trajectory_kl`, or epistemic information gain. For
deterministic policies under the same uniform mixture, any two unequal macros
at a fixed state receive the same KL; the metric adds exact action-space scale
to disagreement but does not encode semantic action distance.

## 11. Events, artifacts, and summary

The new measurement run is append-only and contains:

```text
benchmark/
  action-space-spec.json
  reference-state-spec.json
  historical-policy-lineage.json
reference/
  matches/...
  states/<measurement_state_id>.json
action-space/
  counts/<measurement_state_id>.json
  cache/...
policies/
  <version>/<measurement_state_id>.json
measurement/
  per-state-kl.jsonl
  transition-summary.json
events.jsonl
summary.json
quality.json
```

New finalized events include:

- `reference_state_selected`;
- `action_space_count`;
- `historical_policy_action`; and
- `controlled_reference_policy_kl`.

Every event retains common schema fields and provenance IDs. Count events
store decimal `support_size`; KL events store direction, epsilon, decimal
nats, display nats, canonical action hashes, equality, coverage, and status.

`summary.json` derives six adjacent transition points, the primary
KL-iteration series, sensitivity series, coverage, support statistics,
resource totals, and explicit missing statuses. Original v0-v6 summaries are
never modified.

## 12. Failure and recovery semantics

- Invalid reference replay: run incomplete; no primary aggregate.
- Missing measurement-state field: affected state incomplete.
- Exact counter timeout, memory stop, or interruption: save checkpoint and
  incomplete status; never publish a partial count.
- Counter cycle: fail the action-space specification and publish no KL.
- Historical policy process failure or illegal macro: save raw evidence and
  mark affected values missing.
- Missing primary state in a transition: aggregate is `null`; no subset mean.
- Unknown token, time, or resource field: preserve `null`, never zero-fill.
- Recovery creates finalized continuation events without rewriting finalized
  facts.

Operational time and memory guards protect the host but are not scientific
truncation parameters. An incomplete exact count may be resumed with more
resources under the same `spec_id`.

## 13. CI presentation

The research report adds:

- controlled-reference KL versus policy iteration for v0-v6;
- primary epsilon `0.01`;
- a four-epsilon sensitivity table;
- per-transition coverage;
- support cardinality or `log10 |A(s)|` per reference state;
- action disagreement beside, not renamed as, KL; and
- visible incomplete and unavailable statuses.

The chart title and caption state that this is early controlled real-state
policy behavior change. The report must not present it as a full-episode
information-gain curve or as epistemic learning.

RL remains visibly unavailable until an RL adapter supplies a probability
distribution on this exact `spec_id`. Count-only HL support does not justify
an RL KL claim.

## 14. Verification strategy

Tests include:

1. official differential legality for every command family 1-8;
2. complete parameter-domain and stable-order contracts;
3. request sentinel, ignored parameter, terminal suffix, and malformed-action
   canonicalization;
4. explicit enumeration versus memoized count on small controlled states;
5. legal-prefix property tests against cloned official execution;
6. recurrence, arbitrary-precision serialization, cycle detection, and
   checkpoint recovery;
7. closed-form KL versus explicitly materialized distributions on small
   supports;
8. zero for equal actions and finite nonnegative nats for unequal actions;
9. epsilon and high-precision numerical tests;
10. measurement-state, action-space spec, policy, and reference selection hash
    stability;
11. v0-v6 process isolation, determinism, legality, and source immutability;
12. event, summary, missing-value, quality, and CI contracts;
13. a fake end-to-end measurement run; and
14. the real 12-state historical run followed by fresh artifact audit.

Both Framework and Generals-assets full test suites run before completion is
claimed.

## 15. Acceptance criteria

The ideal complete numerical result requires:

- one frozen, valid 12-state strongest-human self-play reference set;
- exact action-space counts for 12/12 primary states;
- legal deterministic outputs from v0-v6 on all 12 states;
- six adjacent transitions with 72/72 primary per-state values;
- 288/288 sensitivity values;
- six complete primary KL aggregates;
- saved raw counts, policy actions, provenance, budgets, and logs;
- zero malformed, duplicate, unknown, or missing-required-field events;
- correct CI curves, sensitivity, coverage, and missing-state display; and
- all targeted and full regression tests passing.

If a frozen historical policy is genuinely invalid on a frozen reference
state, the implementation may still be technically complete, but the affected
scientific aggregate must remain `null`. The result report must distinguish
that policy fact from an infrastructure or action-space implementation
failure.
