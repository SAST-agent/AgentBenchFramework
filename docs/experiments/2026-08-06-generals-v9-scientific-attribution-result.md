# Generals v9 Scientific Attribution — Partial Result

Date: 2026-08-06

Status: **blocked before v9 generation by the external Codex account usage
limit**. The paired v8 scientific attribution is complete and auditable. No
runnable v9 policy, v9 validation/formal score, expanded-24 KL run, or v9 paper
figure is claimed.

## Frozen authorities

- v7 policy parent: run `20260730_1739_680b1632`, source hash
  `c1eb1e4eae5fb1afa393e15370d744743fe0af206cc05a329fa36bd62ca3c4c4`.
- v8 chronological predecessor: run `20260806_1126_8e1b6795`, source hash
  `3f69b81a1b4831a980084f8419b39292fa34feac43cb6b88aefe26da3898f35a`.
- Completed attribution run: `20260806_1639_e4c73432`.
- Attribution report SHA-256:
  `f134e91092adab1452fb3a006268c4a5a1b0fb518fef136b3f94973f3b8c88b5`.
- Attribution evidence SHA-256:
  `b5ddc9c2024c924b248d0510e38dd8bc6185d9fba66a393d659f579080571711`.

## Attribution validity

- 4 policy cells in frozen order A/B/C/D.
- 12 identical seed/seat pairs per cell; 48/48 valid games.
- 48 diagnostic measurement states.
- 192 same-state policy probes: four policies per state, each internally
  double-probed for determinism.
- 0 coding-agent acts and no formal benchmark access in the attribution run.
- 503/503 valid events, with zero malformed records, unknown event types,
  duplicate IDs, missing IDs, or warnings.

The cells were:

| Cell | Policy | Intervention |
|---|---|---|
| A | v7 control | none |
| B | v7 | large-stack priority only |
| C | v7 | contact before economy only |
| D | v8 | both interventions |

## Paired factorial estimates

Effects are `B-A` (large stack), `C-A` (contact), and `D-B-C+A`
(interaction). Intervals are the frozen paired-bootstrap intervals. Missing
values remain missing.

| Metric | Large stack | Contact | Interaction | Complete pairs |
|---|---:|---:|---:|---:|
| outcome score | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 12 |
| rounds survived | 32.000 [-18.917, 81.167] | 12.333 [-1.167, 37.500] | -34.833 [-71.000, -1.250] | 12 |
| territory margin | 2.583 [-15.333, 19.250] | -3.250 [-9.167, 1.333] | 2.167 [-14.167, 20.000] | 12 |
| army margin | -875.667 [-2390.583, 577.583] | -469.667 [-1061.917, -87.833] | 1289.167 [452.750, 2274.833] | 12 |
| coin margin | -1229.750 [-3439.333, 1004.083] | -143.333 [-948.583, 363.500] | 1171.417 [-395.083, 2915.083] | 12 |
| net main pressure | missing | missing | missing | 0 |

## Supported conclusion

All four cells lost all 12 strongest-human attribution games, so binary outcome
is still at the floor and cannot identify either v8 intervention. The dense
measurements do show non-additive behavior: contact-before-economy alone has a
negative army-margin effect whose interval excludes zero, while the combined
interaction has a positive army-margin effect whose interval excludes zero.
The same interaction reduces rounds survived. Large-stack priority alone has
wide intervals containing zero. These are diagnostic effects on the selected
states and cases, not evidence of a formal score improvement.

The retained champion therefore remains v7. No v9 champion decision was
opened.

## Provider blocker and act accounting

The first provider attempt (`20260806_1644_d5161186`) failed before Codex could
initialize its local state database under the outer filesystem sandbox. Its raw
JSONL is empty, token/tool counts are absent, and the workspace remained exactly
v7. It is retained as infrastructure evidence.

The sandbox-exempt attempt (`20260806_1852_a6db0256`) successfully started a
Codex thread but immediately returned the account usage-limit error before any
model output, token usage, tool call, or source edit. Its candidate hash is
still the v7 hash and the run is correctly marked `invalid_version`. The
provider reported that credits become available again at **2026-08-11 10:15
Asia/Shanghai**.

No further provider retry was made. Treat both records as failed provider
preflights, not as completed heuristic-learning acts. The scientific v9 act,
validation/formal evaluation, exact dual-domain KL, and final figures must be
continued only after provider capacity is restored.

## Required continuation

1. Re-run `iterate-v9` once from the same v7, v8, attribution report, replay
   Skill, and frozen hashes after provider capacity is restored.
2. If and only if the resulting v9 is runnable, require all 12 validation and
   all 18 formal games.
3. Run `measure-policy-kl-expanded` against the verified legacy v3 KL run and
   runnable v9 hash. Do not accept approximate support counts.
4. Generate the three English PNG/SVG figures from the completed authorities.
5. Replace this partial status with the v9 score, dual-domain KL, final
   champion decision, and artifact hashes.
