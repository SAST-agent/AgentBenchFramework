# Generals pilot v3 rescue result

## Scope

This is the explicitly labeled legacy-pilot `v2 → v3` rescue. It is not a new
canonical from-scratch experiment. The immutable parent was:

- run: `20260727_0639_89578eec`
- version: `v2`
- content hash:
  `1e8a2ce4fa38b4171f6f8d77d42b80d48438f5213b08d543714d8241e292c565`

The successful child run is `20260728_1122_57f647d5`. Its v3 content hash is
`a9f27eb2a02ba452e9363022d97ffa5556c0f3a02eef120061e73e93c68ce815`.

## Frozen protocol

- Learn against the strongest frozen human opponent on three learning-only
  seeds and both seats: 6 v2 games.
- Give Codex a 15,718-byte compact prompt containing official rules, replay
  guidance, six redacted learning summaries, and observable v2 action classes.
- Permit deterministic, explainable architecture changes and strategy
  compression. Require an updated `EXPERIENCE.md`.
- Re-run the same 6 learning cases with v3.
- Open the formal 18-case high/medium/low benchmark only if every behavior-gate
  condition passes.

Prompt SHA-256:
`de47e9b2ce313c83c0d3955075d24133676b5d8f8030074bbe088120dabe04b7`.

## Gate result

All seven conditions passed: provider completion, protected-file integrity,
strategy tests, nonzero behavior change, observed non-main movement, 6/6 valid
v3 validation games, and at least one improved paired dense metric.

On the 1,072 frozen v2 probe states:

| Observable action class | v2 | v3 |
| --- | ---: | ---: |
| Main-army move | 93.3% | 17.4% |
| Non-main-army move | 0.2% | 64.5% |
| General upgrade | 0.0% | 11.7% |
| End only | 6.5% | 6.5% |

Action disagreement was `82.93%`.

Paired v3-minus-v2 learning deltas:

| Dense metric | Mean delta |
| --- | ---: |
| Completed rounds survived | +3.83 |
| Terminal territory share | +0.2553 |
| Terminal army margin | +311 |
| Terminal coin share | -0.0880 |
| Terminal net-main pressure | missing |

The strongest-human learning result remained `0/6`; dense improvement therefore
does not imply a strongest-opponent win.

## Formal result

The formal benchmark improved from `0/18` to `7/18`:

- score: `38.89%`
- gain from v0: `+38.89` percentage points
- high tier: `0/6`
- medium tier: `1/6`
- low tier: `6/6`
- evaluated as seat 0: `4/9`
- evaluated as seat 1: `3/9`
- valid cases: `18/18`

This is a real removal of the original floor against the low tier and a first
medium-tier win. It is not yet competitive with the strongest opponent.

## What changed

Codex replaced the main-first BFS chain with a deterministic global stack
planner. It enumerates safe captures from all owned movable stacks, can buy a
known production upgrade, and compares stable BFS routes across all stacks.
The act changed only:

- `strategy.py`
- `STRATEGY.md`
- `EXPERIENCE.md`
- `tests/test_strategy.py`

The v3 snapshot passes `15` strategy tests.

## Measurement integrity

- The historical provider failure remains a missing score at global act 2.
- The score curve is `[0, 0, missing, 0, 0.3889]`.
- Full-curve AUC is unavailable; the report does not bridge the missing point.
- The audited cumulative learning budget is 4 coding-agent acts, 46 episodes,
  27,664 environment steps, 10,485 game-agent decisions, 41,977 primitive
  commands, and 582.50 seconds.
- Cumulative tokens remain missing because the failed act exposed no token
  count. The successful v3 act itself used 248,987 input and 5,779 output
  tokens.
- The budget repair is recoverable through
  `summary.pre-budget-audit.json` and documented by `budget-audit.json`.
- Event quality is 134/134 valid, with no malformed, unknown, duplicate, or
  missing-ID events.

## Next iteration

The next v4 should retain the global-stack planner and target the remaining
high/medium gap: explicit main-threat reserves, safer all-but-one movement,
combat attrition, and urgent defense versus production. The formal low-tier
cases should remain held out while those changes are learned and gated on a new
strongest-human learning seed set.
