# Generals controlled-reference policy KL through v7

## Status and provenance

- Status: `complete`
- Measurement ID: `generals-policy-kl-reference-v2`
- Run ID: `20260731_1741_a4aa1f8c`
- Source measurement: `generals-policy-kl-reference-v1`
- Source run: `20260730_1126_8d123b55`
- Reuse mode: `verified_v1_domain_probe_only_v7`
- Framework commit recorded by the run: `00437d0`
- Wall time: 8.28 seconds
- KL direction and unit: `new || old`, natural logarithm, nats
- Primary smoothing: `epsilon = 0.01`

The v2 run did not overwrite or recompute the v1 measurement. It verified the
frozen v1 domain and facts, materialized them into a new run, and added only
the v7 policy probe and v6→v7 KL records. The canonical source-tree hash was
identical before and after extension:

`6203161e1056628d46bb974ed2ffaaf1b2d6a67581e80b57f0d92afa2bb41225`

All first-six transition objects in the v2 summary are exactly equal to the
corresponding v1 summary objects. All 396 reused scientific events retain a
non-null `source_event_id`; 162 copied artifacts have individual SHA-256
receipts.

## Frozen measurement domain

The run reuses the same 12 strongest-human self-play states and exact
canonical support set as v1:

- seeds: `289101`, `289202`, `289303`;
- seats: `0`, `1`;
- seat-local decisions: `2`, `10`;
- exact support coverage: 12/12;
- exact support range: 7–24,596; and
- action-space `spec_id`:
  `a383f61cba2b284623c0b377eaddee4ef7522b8e8adb53e0ea3efb7bc9bc329e`.

No support count, reference state, v0–v6 action, or v0→v6 KL value was
recomputed.

## v7 probe integrity

The retained v7 policy source comes from run `20260730_1739_680b1632` with
content hash:

`c1eb1e4eae5fb1afa393e15370d744743fe0af206cc05a329fa36bd62ca3c4c4`

All 12 v7 probes completed. Each action was reproduced by two fresh
subprocess probes, and all 12 pairs were deterministic. Every selected macro
was valid under the frozen canonical action-space specification.

The v6 and v7 canonical actions are equal state-by-state on all 12 frozen
states. Therefore the measured v6→v7 disagreement rate and KL are zero. This
is a real observed equality on this domain, not missing data, interpolation,
or fallback reuse of v6 actions.

## v0→v7 controlled-reference KL curve

| Transition | Mean KL at ε=0.01 (nats/state) | Action disagreement | Coverage |
|---|---:|---:|---:|
| v0→v1 | 0.000000 | 0.0% | 12/12 |
| v1→v2 | 0.000000 | 0.0% | 12/12 |
| v2→v3 | 5.395765 | 50.0% | 12/12 |
| v3→v4 | 8.929730 | 91.7% | 12/12 |
| v4→v5 | 8.929730 | 91.7% | 12/12 |
| v5→v6 | 6.504663 | 66.7% | 12/12 |
| v6→v7 | 0.000000 | 0.0% | 12/12 |

## Epsilon sensitivity

| Transition | ε=0.001 | ε=0.01 | ε=0.05 | ε=0.1 |
|---|---:|---:|---:|---:|
| v0→v1 | 0.000 | 0.000 | 0.000 | 0.000 |
| v1→v2 | 0.000 | 0.000 | 0.000 | 0.000 |
| v2→v3 | 6.599 | 5.396 | 4.394 | 3.827 |
| v3→v4 | 11.127 | 8.930 | 7.134 | 6.144 |
| v4→v5 | 11.127 | 8.930 | 7.134 | 6.144 |
| v5→v6 | 8.103 | 6.505 | 5.198 | 4.478 |
| v6→v7 | 0.000 | 0.000 | 0.000 | 0.000 |

Coverage is 12/12 for every transition and epsilon. The zero v6→v7 result is
epsilon-independent because the two deterministic policies select the same
canonical macro at every measured state.

## Event and artifact audit

- `policy_kl_source_verified`: 1
- `policy_kl_artifact_reused`: 162
- `reference_state_selected`: 12
- `action_space_count`: 12
- `historical_policy_action`: 96 (84 reused, 12 new v7)
- `controlled_reference_policy_kl`: 336 (288 reused, 48 new v6→v7)
- Per-state KL rows: 336 = 7 transitions × 12 states × 4 epsilons
- Total events: 619
- Malformed, invalid, unknown, or duplicate events: 0
- Missing event IDs or run IDs: 0
- Quality warnings: 0

## Interpretation boundary

This result measures deterministic policy change on 12 controlled early-game
states. A v6→v7 value of zero means only that the two policies chose the same
canonical macros on this frozen domain. It does not establish global source
equivalence, identical behavior later in games, or equal benchmark strength.
The KL curve must be read beside formal benchmark and dense trajectory
metrics; it is not itself a performance or epistemic information-gain score.

## Reproduction

From the Framework worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m agentbench_frame.cli \
  generals extend-policy-kl-v7 \
  --agentbench-root /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets \
  --manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml \
  --reference-manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/policy-kl-reference-v2.toml \
  --source-run /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-policy-kl/20260730_1126_8d123b55 \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data
```

```bash
MPLCONFIGDIR=/tmp/agentbench-matplotlib .venv/bin/python \
  scripts/plot_generals_controlled_policy_kl.py \
  --run-dir agentbench_data/runs/28_generals/generals-policy-kl/20260731_1741_a4aa1f8c \
  --output-prefix docs/experiments/figures/generals-controlled-policy-kl-three-panel
```

The generated paper figure is available as both PNG and SVG under
`docs/experiments/figures/`.
