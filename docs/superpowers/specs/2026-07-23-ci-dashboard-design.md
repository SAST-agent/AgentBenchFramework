# Research CI Dashboard Design

## Goal

Make the static CI report a research instrument first: a reader should be able
to see performance, learning-only budget, information-change traces, and the
raw-data quality status without opening a log file.

## Page structure

1. A compact masthead identifies the benchmark report, generation time, and
   source-data status.
2. The first metric row shows benchmark score, gain over raw, learning-only
   coding-agent acts, and the count of complete runs.
3. The main performance panel plots benchmark score against coding-agent act
   and shows `AUC_coding_agent_act` beside the chart. Missing scores remain
   gaps; the report never fills an incomplete evaluation.
4. The secondary panels show per-episode policy-change IG and a budget ledger.
   IG is the episode mean of the raw local KL trace and is labelled in
   `nats / episode`; the raw trace remains available in `data.json`.
5. A run table exposes benchmark status, raw/evo/gain, learning-only budget,
   and the run identifier. Existing Elo fields remain visible in detail pages.

## Visual system

- Ink/navy background with warm paper-like panels, mint for valid improvement,
  amber for incomplete/unknown data, and red only for failed data.
- Inline SVG symbols use `currentColor`, 20px viewboxes, and a consistent
  1.7px stroke. This keeps icons offline, accessible, and themeable.
- Responsive CSS uses one-column layout below 760px and preserves readable
  tables by allowing horizontal scrolling.

## Compatibility

- `summary.json` remains the derived snapshot and `events.jsonl` remains the
  source of truth when present.
- Old Elo/win-rate-only runs continue to render with explicit missing research
  values.
- Unknown event types are ignored. New research fields are additive.
- The CI builder writes the derived research context to `data.json` so later
  visualizations can be added without changing raw files.
- Occupancy observations are reported as a separate evidence count; they are
  never added to policy KL or presented as a second interchangeable IG score.
