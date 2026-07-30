# Generals Policy-KL Paper Figure Design

## Goal

Generate one publication-ready, English-only, horizontal three-panel figure
from the finalized Generals controlled-reference measurement run
`20260730_1126_8d123b55`.

The figure must be reproducible from saved artifacts and must not manually
duplicate scientific values in the plotting code.

## Source artifacts

- `summary.json` supplies the six transition aggregates and all four epsilon
  sensitivity series.
- `events.jsonl` supplies the 12 exact `action_space_count` records, including
  seed, seat, decision number, support size, and status.
- A transition or state whose saved status/coverage is incomplete remains
  visibly missing; the plot does not interpolate it.

## Layout

Use one landscape canvas with three panels:

### Panel A — Controlled-reference Policy KL over Iterations

- X-axis: `v0→v1` through `v5→v6`.
- Y-axis: mean KL in nats/state at primary `epsilon = 0.01`.
- One dark-blue line with circular markers.
- Annotate every complete point with its displayed value.
- A missing aggregate creates a visible gap.

### Panel B — Epsilon Sensitivity

- Same transition x-axis as Panel A.
- Y-axis: mean KL in nats/state.
- Four lines for `epsilon = 0.001`, `0.01`, `0.05`, and `0.1`.
- Use a colorblind-safe palette, distinct markers, and an unobtrusive legend.
- Missing values break only the affected series.

### Panel C — Exact Canonical Support Size

- Twelve bars ordered by decision number, then seed, then seat.
- X labels use compact `d{decision} · s{seed suffix} · p{seat}` notation.
- Y-axis is logarithmic because observed exact supports span 7–24,596.
- Decision 2 and decision 10 use two related orange tones.
- Annotate each complete bar with its exact decimal support size.
- Missing or incomplete states use an outlined placeholder rather than a
  fabricated value.

## Visual style

- White background and light grey dashed gridlines.
- Dark neutral text, restrained spines, and generous whitespace.
- English-only labels and titles.
- Panel labels `(a)`, `(b)`, `(c)`.
- No gradients, shadows, decorative icons, or 3D effects.
- Match the supplied reference image's clean academic presentation without
  copying its underlying chart semantics.

## Outputs

- Vector: `docs/experiments/figures/generals-controlled-policy-kl-three-panel.svg`
- Raster:
  `docs/experiments/figures/generals-controlled-policy-kl-three-panel.png`
  at 300 DPI.
- Reproduction script:
  `scripts/plot_generals_controlled_policy_kl.py`

The script accepts explicit `--run-dir` and `--output-prefix` arguments,
validates the metric name, reference count, transition order, epsilon set,
coverage, and count event uniqueness, then writes both formats.

## Verification

- Run the plotting script against the finalized run.
- Confirm both files are non-empty and the PNG is readable at the intended
  dimensions.
- Parse the SVG and check all panel titles, six transitions, four epsilon
  labels, and 12 state bars are present.
- Compare plotted numerical labels to `summary.json` and `events.jsonl`.
- Run existing Framework tests to ensure the figure tooling does not alter
  measurement or report behavior.
