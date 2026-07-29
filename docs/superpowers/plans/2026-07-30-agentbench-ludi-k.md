# AgentBench Ludi K-Complexity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible AB-Ludi/1 calculator and frozen Kolmogorov-complexity upper bounds for all ten public AgentBench game-logic corpora.

**Architecture:** A versioned path/hash manifest binds the authoritative logic source for each game to immutable blobs at one pinned Git commit. A lossless binary ludeme-tree encoder frames exact source modules, a zlib profile identified by runtime and behavior fingerprint supplies the executable description upper bound, and renderers produce auditable JSON and Markdown artifacts.

**Tech Stack:** Python 3.11 standard library, `pytest`, argparse, zlib, SHA-256.

## Global Constraints

- Core runtime remains zero-dependency.
- The metric is named and documented as an AB-Ludi/1 conditional upper bound, never exact \(K\).
- The source boundary contains exactly the ten public games at the pinned AgentBench commit.
- Every included file path and SHA-256 digest is retained in the JSON artifact.
- Input drift fails closed; files are never silently added or dropped.
- Production behavior is implemented test-first.

---

### Task 1: Frozen AgentBench Source Catalog

**Files:**
- Create: `src/agentbench_frame/research/__init__.py`
- Create: `src/agentbench_frame/research/agentbench_catalog.py`
- Create: `src/agentbench_frame/research/agentbench_ludi_v1_manifest.json`
- Create: `tests/research/__init__.py`
- Create: `tests/research/test_agentbench_catalog.py`

**Interfaces:**
- Produces: `GameSourceSpec`, `AGENTBENCH_GAME_SPECS`, and `collect_game_sources(repo_root, spec) -> tuple[SourceModule, ...]`.
- Consumes: a local checkout whose corpus is at `backend_sources/corpus`.

- [ ] **Step 1: Write failing tests for the ten-game identity set and boundary filtering**

Create temporary fixture roots containing genuine `.py`, `.cpp`, `.json`,
`.g4`, and `.map` files alongside `jsoncpp`, `data`, `test_config`, generated
ANTLR, debug/legacy, docs, binary, and sample-AI files. Assert only authoritative
logic modules remain and canonical paths are sorted.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=src UV_CACHE_DIR=/tmp/agentbench-ludi-uv-cache \
  uv run --no-project --with pytest \
  pytest tests/research/test_agentbench_catalog.py -q
```

Expected: import failure because `agentbench_frame.research.agentbench_catalog`
does not exist.

- [ ] **Step 3: Implement the immutable catalog and fail-closed collector**

Define one source root per game, an exact 149-file path/SHA-256 manifest, the
suffix allowlist, path/name exclusions, explicit generated-file exclusions,
Git-blob loading, and unexpected/missing-game validation. Read by object ID from
the pinned commit, compare the selected Git tree against the manifest, and use
immutable dataclasses/tuples.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all catalog tests pass.

### Task 2: AB-Ludi/1 Lossless Encoder and Metric

**Files:**
- Create: `src/agentbench_frame/research/ludi_k.py`
- Create: `tests/research/test_ludi_k.py`

**Interfaces:**
- Consumes: `SourceModule` tuples from Task 1.
- Produces: `encode_ludi_description`, `decode_ludi_description`,
  `measure_game`, `measure_agentbench_repository`, `write_json_report`, and
  `write_markdown_report`.

- [ ] **Step 1: Write failing tests for framing, order invariance, hashing, compression metadata, and renderers**

Tests use arbitrary bytes including NUL and parentheses, reverse module order,
verify exact round-trip, inspect required report fields, and assert rendering
twice yields byte-identical output.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
PYTHONPATH=src UV_CACHE_DIR=/tmp/agentbench-ludi-uv-cache \
  uv run --no-project --with pytest pytest tests/research/test_ludi_k.py -q
```

Expected: import failure because `agentbench_frame.research.ludi_k` does not
exist.

- [ ] **Step 3: Implement minimal deterministic framing, measurement, provenance, and rendering**

Use unsigned 64-bit big-endian prefixes, `hashlib.sha256`, a fully specified
`zlib.compressobj`, a fixed compressor test-vector fingerprint, sorted
modules/results, pinned `git rev-parse HEAD` provenance, and stable UTF-8
JSON/Markdown output with a trailing newline.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all encoder/metric tests pass.

### Task 3: Framework CLI

**Files:**
- Modify: `src/agentbench_frame/cli.py`
- Create: `tests/research/test_ludi_k_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `measure_agentbench_repository` and report writers from Task 2.
- Produces: `agentbench complexity ludi` with three required path arguments.

- [ ] **Step 1: Write a failing CLI test**

Build a temporary ten-game fixture from the catalog, invoke
`main(["complexity", "ludi", ...])`, and assert JSON/Markdown files and a
ten-row ranking are produced.

- [ ] **Step 2: Run the CLI test and verify RED**

```bash
PYTHONPATH=src UV_CACHE_DIR=/tmp/agentbench-ludi-uv-cache \
  uv run --no-project --with pytest pytest tests/research/test_ludi_k_cli.py -q
```

Expected: argparse rejects `complexity`.

- [ ] **Step 3: Add the nested CLI and README usage**

Add a `_cmd_ludi_complexity` handler, nested `complexity ludi` parser, explicit
path options, compact ranking output, and a README research-metrics section
with interpretation warnings.

- [ ] **Step 4: Run the CLI test and verify GREEN**

Run the command from Step 2. Expected: CLI test passes.

### Task 4: Calculate and Freeze All Public Game Results

**Files:**
- Create: `docs/research/agentbench-ludi-k-v1.json`
- Create: `docs/research/agentbench-ludi-k-v1.md`
- Create: `tests/research/test_ludi_k_snapshot.py`

**Interfaces:**
- Consumes: local public source checkout `/tmp/AgentBench-source` at commit
  `b581bca3ba3d2d7d58a2f8c6bbddd060fc7fdc87`.
- Produces: committed audit artifact covering every selected file and all ten
  games.

- [ ] **Step 1: Write a failing snapshot-contract test**

Assert the committed JSON exists, declares `AB-LUDI/1`, records the expected
source commit, contains exactly the ten catalog IDs, and gives every game
nonzero module count, source/canonical/upper-bound bits, source SHA-256, and a
nonempty file manifest.

- [ ] **Step 2: Run the snapshot test and verify RED**

```bash
PYTHONPATH=src UV_CACHE_DIR=/tmp/agentbench-ludi-uv-cache \
  uv run --no-project --with pytest pytest tests/research/test_ludi_k_snapshot.py -q
```

Expected: failure because the report does not exist.

- [ ] **Step 3: Generate the reports with the framework CLI**

```bash
PYTHONPATH=src python -m agentbench_frame.cli complexity ludi \
  --agentbench-repo /tmp/AgentBench-source \
  --json-output docs/research/agentbench-ludi-k-v1.json \
  --markdown-output docs/research/agentbench-ludi-k-v1.md
```

- [ ] **Step 4: Run snapshot and independent regeneration checks**

Run the snapshot test, regenerate both reports into a temporary directory, and
compare their bytes with the committed artifacts. Expected: pass and exact
matches.

### Task 5: Full Verification and Publication

**Files:**
- Review all files changed by Tasks 1-4.

**Interfaces:**
- Produces: one pushed branch and one ready-for-review PR against
  `SAST-agent/AgentBenchFramework:main`.

- [ ] **Step 1: Run all tests outside the socket-restricted sandbox**

```bash
PYTHONPATH=src UV_CACHE_DIR=/tmp/agentbench-ludi-uv-cache \
  uv run --no-project --with pytest pytest -q
```

Expected: all tests and subtests pass.

- [ ] **Step 2: Audit coverage and repository state**

Run the calculator against `/tmp/AgentBench-source`, verify exact artifact
reproduction, inspect `git diff --check`, `git diff --stat`, and confirm no
source checkout or generated temporary file is staged.

- [ ] **Step 3: Commit and push**

Stage only the planned source, tests, documentation, and result artifacts.
Commit with `feat: add reproducible Ludi game complexity bounds`, then push
`codex/ludi-k-complexity` to `framework`.

- [ ] **Step 4: Open the pull request**

Create a ready PR against `main` summarizing the reference machine, ten-game
coverage, source commit, metric limitations, and test evidence.
