# Results, metrics, and failure semantics

## Authority and projection

The complete authority is:

```text
DotoResults/runs/23_doto/<agent>/<run-id>/
```

Do not hand-edit it. Validate with `doto-results validate <run-dir>`. Aggregate
and report with `doto-results aggregate <root>` and `doto-results build-report`.

The unchanged AgentBenchResults repository receives exactly five derived files:
`run.toml`, `summary.json`, `score_curve.json`, `ig_curve.json`, and
`doto_results_ref.json`. `doto-results check-projection <path>` verifies the file
set, finite values, hashes, identity, and byte equality with the authority.

## Training selection

Compare only complete training versions. A formal iteration has 30 declared
tasks: 15 policies, seed 11, both candidate seats. A score is formal only when
all 30 terminate normally. Keep incomplete/null points visible.

For one human policy, `defeated=true` only when both seats terminate normally
and their candidate-oriented mean score difference is strictly positive. A
single win, aggregate win rate, or one seat never establishes defeat.

Score and IG curve points must share iteration, version, and explicit parent.
Use strict KL status as its own evidence: zero, infinite with numeric null and
reason, or missing with reason. Do not rank candidates by a fabricated IG scalar.

## Failures and retries

- Build failure removes stale executable and is closed as failure.
- Formal timeout, crash, protocol error, corrupt replay, or missing cell makes
  the score null and remains attached to the iteration.
- Completed cells are immutable; interruption resumes missing cells only.
- The final candidate is fixed before hidden work. Final failure still seals the
  Run so selective retry cannot improve the reported outcome.
- `projection.json` may record pending, exported, or failed export attempts and
  is excluded from the sealed-content hash. Other post-seal edits invalidate
  authority.

## Publication checks

Require DotoResults validation, aggregation, report generation, projection
validation, and discovery by the unchanged AgentBenchResults aggregator. Scan
both targets for credentials, hidden source fragments, absolute sealed paths,
`NaN`, and `Infinity`. Exclude binaries, caches, locks, and unredacted sealed
metadata from publication.
