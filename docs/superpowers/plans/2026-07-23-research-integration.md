# Research integration implementation plan

## Goal

Keep first-hand run data stable while adding transparent Codex/Claude Code
provider adapters, multi-axis budget curves, occupancy-shift observations,
data-quality diagnostics, and a local research report. No concrete benchmark
test set is bundled.

## Tasks

- [x] Add failing contract tests for provider JSONL parsing, process invocation,
  multi-axis AUC, occupancy references, data quality, and local report fields.
- [x] Implement provider adapters and connect each provider call to one
  coding-agent act, usage ledger entry, workspace snapshot, and raw event.
- [x] Implement named AUC axes and occupancy-shift derivation without mixing
  policy change with state-distribution change.
- [x] Implement forward-compatible data-quality diagnostics for framework and
  CI report loading.
- [x] Synchronize the local report with research metrics and per-case evidence;
  extend the CI report while retaining old input compatibility.
- [ ] Run regression and synthetic-fixture verification, then commit and push
  framework and CI changes separately.

## Verification

- `PYTHONPATH=src python -m unittest discover -s tests -v`
- `python -m unittest discover -s tests -v` in the CI worktree
- compile checks and `git diff --check`
