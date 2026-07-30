# Generals controlled-reference policy KL result

## Status and run identity

- Status: `complete`
- Measurement ID: `generals-policy-kl-reference-v1`
- Run ID: `20260730_1126_8d123b55`
- Run directory:
  `agentbench_data/runs/28_generals/generals-policy-kl/20260730_1126_8d123b55`
- Framework commit recorded by the run: `2558e4c`
- Wall time: 195.76 seconds (`wall_hours = 0.05`)
- Primary smoothing: `epsilon = 0.01`
- KL direction and unit: `new || old`, natural logarithm, nats

The earlier directory `20260730_1119_072ba7dd` is an unfinalized
performance-diagnostic attempt that was manually interrupted before any
scientific result was emitted. It is not included in the curve below.

## Frozen reference-state provenance

The reference domain contains exactly 12 pre-decision states collected from
strongest-human `advanced-rank02-robinliu-v18` self-play:

- seeds: `289101`, `289202`, `289303`;
- seats: `0`, `1`; and
- seat-local decision numbers: `2`, `10`.

The saved `benchmark/reference-state-spec.json` contains the 12 ordered state
IDs. All 12 corresponding lossless `generals-measurement-state-v1` snapshots
are present under `reference/states/`.

## Action-space specification and exact-count completeness

- Schema: `generals-canonical-macro-action-space-v1`
- Canonicalization:
  `generals-behavioral-canonicalization-v1`
- Official engine hash:
  `4ca898365a6fe9a0cca39c3c8be0e49f449334814b106882c0b78b3aabdb1d97`
- Action-space `spec_id`:
  `a383f61cba2b284623c0b377eaddee4ef7522b8e8adb53e0ea3efb7bc9bc329e`

All command families 1–8 are represented. Candidate-domain pruning removes
only requests that are impossible from the serialized official state
(for example, a locked weapon or an opponent-owned source); every remaining
candidate is still accepted or rejected by a cloned official engine
transition.

All 12 exact canonical support counts completed. No sample, truncation,
lower bound, interpolation, or approximate count entered the metric.

| Seed | Seat | Decision | Exact `|A(s)|` | Count time (s) | Expanded states |
|---:|---:|---:|---:|---:|---:|
| 289101 | 0 | 2 | 13 | 0.405 | 13 |
| 289101 | 0 | 10 | 6,231 | 26.600 | 730 |
| 289101 | 1 | 2 | 7 | 0.192 | 7 |
| 289101 | 1 | 10 | 1,050 | 8.453 | 204 |
| 289202 | 0 | 2 | 13 | 0.421 | 13 |
| 289202 | 0 | 10 | 2,619 | 16.678 | 392 |
| 289202 | 1 | 2 | 7 | 0.165 | 7 |
| 289202 | 1 | 10 | 7,020 | 40.050 | 959 |
| 289303 | 0 | 2 | 10 | 0.286 | 10 |
| 289303 | 0 | 10 | 10,977 | 25.329 | 594 |
| 289303 | 1 | 2 | 11 | 0.363 | 11 |
| 289303 | 1 | 10 | 24,596 | 55.087 | 1,324 |

The observed support range is 7–24,596. Decision-10 states have much larger
supports than decision-2 states in this controlled set.

## Historical policy integrity and probe coverage

The frozen v0–v6 sources were copied only after their workspace manifests and
content hashes matched the approved history:

| Version | Source run | Content hash |
|---|---|---|
| v0 | `20260726_1631_f56f789f` | `bfab30cdaaa0ad8e0ac6ed1b9ab047417c621cc22dd22fd95b0946c3af4bd5fa` |
| v1 | `20260726_1631_f56f789f` | `17356a28682378c4877f7da7aeb8ac90b8891e2685012dec8cdec990cb282e79` |
| v2 | `20260727_0639_89578eec` | `1e8a2ce4fa38b4171f6f8d77d42b80d48438f5213b08d543714d8241e292c565` |
| v3 | `20260728_1122_57f647d5` | `a9f27eb2a02ba452e9363022d97ffa5556c0f3a02eef120061e73e93c68ce815` |
| v4 | `20260729_0818_e6bcb9b3` | `5c12e7c92843cbc18b742ab7be71c684cfa0a0aac8e5dbb5a24fc7b905fe959f` |
| v5 | `20260729_0818_e6bcb9b3` | `facd39c8a0c064823da68f3dd3eba4a832c08815f75369407112ca087f1ecd9e` |
| v6 | `20260729_1653_af8eda26` | `974050ee1a3d4b4c4f96e61f5af39b4e52e2cfbc50f146f9e8b4ac96c4ac798b` |

All 84 version-state probes completed, were deterministic across two fresh
subprocesses, and produced an official-legal canonical macro. Coverage is
therefore 84/84.

## v0→v6 controlled-reference KL curve

| Transition | Mean KL at ε=0.01 (nats/state) | Action disagreement | Coverage |
|---|---:|---:|---:|
| v0→v1 | 0.000000 | 0.0% | 12/12 |
| v1→v2 | 0.000000 | 0.0% | 12/12 |
| v2→v3 | 5.395765 | 50.0% | 12/12 |
| v3→v4 | 8.929730 | 91.7% | 12/12 |
| v4→v5 | 8.929730 | 91.7% | 12/12 |
| v5→v6 | 6.504663 | 66.7% | 12/12 |

On these states, v0, v1, and v2 choose identical canonical macros. This does
not establish global policy identity; it establishes zero measured change on
the frozen reference domain.

v5 exactly matches v3 on all 12 controlled states. Consequently, v3→v4 and
v4→v5 change the same 11 states and have the same support-weighted KL. This is
direct behavioral evidence that the retained v5 rolled these probes back from
v4 to the v3 behavior, not evidence that v4 and v5 are similar.

## Epsilon sensitivity

| Transition | ε=0.001 | ε=0.01 | ε=0.05 | ε=0.1 |
|---|---:|---:|---:|---:|
| v0→v1 | 0.000 | 0.000 | 0.000 | 0.000 |
| v1→v2 | 0.000 | 0.000 | 0.000 | 0.000 |
| v2→v3 | 6.599 | 5.396 | 4.394 | 3.827 |
| v3→v4 | 11.127 | 8.930 | 7.134 | 6.144 |
| v4→v5 | 11.127 | 8.930 | 7.134 | 6.144 |
| v5→v6 | 8.103 | 6.505 | 5.198 | 4.478 |

Coverage is 12/12 at every epsilon. The expected magnitude decreases as
uniform smoothing increases, while zero/nonzero transition conclusions and
the ordering of the nonzero transitions remain stable.

## Resource and cache budget

- Exact-count elapsed time summed over the 12 roots: 174.03 seconds.
- Exact completed SQLite rows: 4,264.
- Expanded states: 4,264.
- Legal edges: 11,948.
- Cache hits: 7,696.
- SQLite cache size after completion: 1,376,256 bytes.
- Saved successful official primitive transition records: 12,027.
- Operational guards: 3,600 seconds and 5,000,000 expanded states; neither
  guard fired.

The generic Framework budget remains zero because this is a measurement run,
not a learning or benchmark episode run. The exact-count resource receipt
above comes from the dedicated first-hand count events.

## Event and artifact quality

- `reference_state_selected`: 12
- `action_space_count`: 12
- `historical_policy_action`: 84
- `controlled_reference_policy_kl`: 288
- Total events: 396
- Primary ε records: 72/72
- Four-ε records: 288/288
- Malformed events: 0
- Invalid events: 0
- Unknown event types: 0
- Duplicate event IDs: 0
- Missing event IDs/run IDs: 0
- Quality warnings: 0

`agentbench data check` reported `16 valid, 0 invalid` for the complete local
data directory. The static report was built successfully and keeps this
metric in a separate “Controlled-reference policy KL” panel rather than the
epistemic information-gain panel.

## Scientific limitations

1. This is deterministic policy-change KL on 12 early controlled real states.
   It is not trajectory KL, occupancy-weighted KL, reward, benchmark score, or
   epistemic information gain.
2. The states cover only three seeds, two seats, and decisions 2 and 10 from
   one strongest-human self-play collector. They do not represent the full
   state distribution.
3. Uniform `U_s` makes a disagreement larger on states with larger exact
   canonical support. The result therefore combines disagreement incidence
   with support scale by design.
4. A zero value means identical selected macros on this frozen domain only.
   It does not prove source equivalence or identical behavior elsewhere.
5. These KL values do not imply benchmark improvement. They must be read
   beside the independently reported formal benchmark and dense trajectory
   metrics.
6. RL comparison remains unavailable until an RL policy exposes a
   distribution over this exact action-space `spec_id`.

## Reproduction commands

From the Framework worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m agentbench_frame.cli \
  generals measure-policy-kl \
  --agentbench-root /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets \
  --manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml \
  --reference-manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/policy-kl-reference-v1.toml \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data \
  --count-wall-time 3600 \
  --count-max-states 5000000
```

```bash
.venv/bin/python -m agentbench_frame.cli data check \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data

.venv/bin/python -m agentbench_frame.cli report \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data \
  --output-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/_site
```
