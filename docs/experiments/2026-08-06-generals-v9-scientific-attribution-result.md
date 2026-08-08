# Generals v9 Scientific Attribution — Terminal Result

Date: 2026-08-08

Status: **the single permitted v9 provider act completed, but the frozen
candidate failed closed before candidate tests and evaluation**. The paired v8
scientific attribution and the v9 provider receipt are complete and auditable.
No runnable v9 policy, v9 validation/formal score, expanded-24 KL run, or v9
paper figure is claimed.

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
- Terminal v9 act run: `20260808_1224_bca2ad8f`.
- Frozen changed-candidate hash:
  `cf7af250fee8a61961e5fd94d20fb1eac9cf6a747b6438ee24f8d4e051ff512b`.

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

The retained champion therefore remains v7. The generated v9 candidate did not
reach the runnable gate, so no champion decision was opened.

## Provider act and candidate gate

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

Both records remain failed provider preflights rather than completed
heuristic-learning acts.

The authorized retry `20260808_1224_bca2ad8f` completed one provider act from
the same frozen authorities. The exact receipt records:

- prompt tokens: `3,146,682`;
- completion tokens: `23,713`;
- total tokens: `3,170,395`;
- provider tool calls: `108`;
- provider elapsed time: `651.0511644259968` seconds;
- changed files: `strategy.py`, `tests/test_strategy.py`, `STRATEGY.md`, and
  `EXPERIENCE.md`; and
- frozen candidate hash:
  `cf7af250fee8a61961e5fd94d20fb1eac9cf6a747b6438ee24f8d4e051ff512b`.

The candidate added a compact phase scheduler, contact-before-economy ordering,
a higher contact coin reserve, late-contact coin preservation, frozen-source
skipping, and bounded projection of a safe large main stack through owned
route cells. The provider's own assertion runner reported that its strategy
tests passed. That is provider evidence only, not the Framework candidate-test
receipt.

The Framework version event was fail-closed with all of these checks true:

- `provider_completed`;
- `scope_valid`;
- `manifest_valid`;
- `required_files_valid`;
- `main_unchanged`; and
- `runtime_source_valid`.

`strategy_documents_valid` was false, so candidate tests and policy probes were
correctly skipped and `runnable` remained false. The exact gate mismatch was a
whitespace-sensitive inherited check. It accepts a contiguous `fallback`,
`verified prefix`, or `safe prefix` token in `STRATEGY.md`; the candidate wrote
`verified` and `prefix` across a Markdown line break. The document conveyed the
required verified-prefix rule semantically, but the frozen implementation gate
does not normalize whitespace before matching it.

This is recorded as an infrastructure false negative, not retroactively
converted into a runnable candidate. The predeclared protocol permits no second
v9 provider act and allows recovery only for an already hash-verified runnable
candidate. Recovery also may not upgrade a failed candidate. Therefore the
frozen run remains `invalid_version`, with no validation or formal games.

## Terminal decision

1. The one scientific v9 act is complete and immutable.
2. Because v9 is not runnable, no validation or formal score is reported.
3. `measure-policy-kl-expanded` is not run because its frozen contract requires
   a runnable v9 source and hash.
4. The v9 paper figures are not generated because the required v9 and expanded
   KL authorities do not exist.
5. v7 remains the champion at `12/18`; v8 remains the runnable regression at
   `7/18`; v9 remains an invalid historical candidate with no score.

The whitespace-normalization defect should be fixed and regression-tested for
future rounds, but such a Framework change must not alter this v9 result or
authorize post-hoc evaluation of its failed candidate.
