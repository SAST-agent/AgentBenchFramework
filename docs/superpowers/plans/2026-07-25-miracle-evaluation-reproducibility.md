# 24 Miracle Evaluation Reproducibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PR #3 runnable from its own checkout with explicit control-plane inputs, reproducible resume checks, and deterministic Python opponent identity verification.

**Architecture:** Keep `tools/miracle_matrix.py` as the game-specific CLI boundary, but move input loading and preflight ahead of session creation. Store protocol/roster hashes in the existing `MatrixRunner` manifest and validate them read-only before resume. Resolve opponent directories deterministically and verify Python entry-point hashes in addition to the existing C++ artifact checks.

**Tech Stack:** Python 3.11+, `argparse`, `pathlib`, `hashlib`, JSON, pytest, existing AgentBench tracking and Miracle matrix runner.

## Global Constraints

- Do not add the complete evaluation set, replay corpus, or large binaries.
- Do not define or calculate information gain, raw/evo/gain/AUC, or RL statistics in the game adapter.
- Preserve existing event and manifest fields; additions must be forward-compatible.
- Resume verification must be read-only and must not change the session on failure.
- Local `main` is never merged; changes stay on the independent PR #3 repair branch.
- CLI defaults must preserve existing invocation paths while allowing explicit path overrides.

---

### Task 1: Bundle control-plane metadata and add preflight loading

**Files:**
- Create: `docs/games/24_miracle_evaluation_protocol.v0.3.json`
- Create: `docs/games/24_miracle_roster_manifest.json`
- Modify: `tools/miracle_matrix.py:1-80`
- Create: `tests/miracle/test_matrix_cli_inputs.py`

**Interfaces:**
- Produces `PreflightError(RuntimeError)` for user-facing input failures.
- Produces `ControlInputs` with fields `protocol: dict`, `roster: dict`, and `hashes: dict[str, str]`.
- Produces `load_control_inputs(protocol_path: Path, roster_path: Path, expected_protocol_sha: str | None = None) -> ControlInputs`.
- Produces `parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace` with `--dry-run`, `--resume`, `--protocol`, `--roster`, and `--protocol-sha`.

- [ ] **Step 1: Restore the two small metadata files from the canonical PR #1 protocol.**

Use the v0.3 JSON and roster manifest already present in `origin/pr-1`; keep their metadata and frozen hashes unchanged, and add `runnable_sha256` only to the three Python strategy entries using the hashes of their current `main.py` entry points:

```text
rank04: 1dfe69738141f80f7d2fb5ce1a8321126141360d86d515caf66d3f4010d86f88
rank05: 4166a54d042e9e2e5b37cf172705dbf7617872ffb1184adb3ceb697bcc6b9707
rank07: eab339ef43fcc521ec8eb7bcb39859e35e8ea4681bb00661a8faf91c5d63816c
```

- [ ] **Step 2: Write failing tests for clear control-input failures and argument parsing.**

```python
def test_load_control_inputs_reports_missing_file(tmp_path):
    with pytest.raises(PreflightError, match="protocol file missing"):
        load_control_inputs(tmp_path / "missing.json", tmp_path / "roster.json")


def test_parse_args_accepts_explicit_control_paths(tmp_path):
    args = parse_args(["--dry-run", "--protocol", str(tmp_path / "p.json"),
                       "--roster", str(tmp_path / "r.json")])
    assert args.dry_run is True
    assert args.protocol == tmp_path / "p.json"
    assert args.roster == tmp_path / "r.json"
```

- [ ] **Step 3: Run the focused tests and confirm they fail for the missing interfaces.**

Run: `PYTHONPATH=src python -m pytest -q tests/miracle/test_matrix_cli_inputs.py`

Expected: FAIL because `PreflightError`, `ControlInputs`, `load_control_inputs`, and `parse_args` are not implemented.

- [ ] **Step 4: Implement minimal preflight loading and parser defaults.**

The loader must check existence, JSON decoding, object type, protocol SHA when supplied, and roster strategy ranks exactly equal to `1..16`; every failure raises `PreflightError` with the input label. The parser must preserve the current defaults and return `Path` objects for path options.

- [ ] **Step 5: Run the focused tests and commit the control-plane change.**

Run: `PYTHONPATH=src python -m pytest -q tests/miracle/test_matrix_cli_inputs.py`

Expected: PASS.

```bash
git add docs/games/24_miracle_evaluation_protocol.v0.3.json \
        docs/games/24_miracle_roster_manifest.json \
        tools/miracle_matrix.py tests/miracle/test_matrix_cli_inputs.py
git commit -m "fix(24_miracle): add control-plane preflight inputs"
```

### Task 2: Persist and verify control-input hashes in sessions

**Files:**
- Modify: `src/agentbench_frame/games/miracle/matrix_runner.py:104-230,440-466`
- Modify: `tools/miracle_matrix.py:100-140,165-205`
- Modify: `tests/miracle/test_matrix_runner.py:82-100`
- Modify: `tests/miracle/test_section2_strictness.py:341-433`

**Interfaces:**
- Extends `MatrixRunner.record_manifest(..., control_inputs: dict[str, dict[str, str]] | None = None) -> Path`.
- Extends `verify_session_for_resume(..., expected_control_inputs: dict[str, str] | None = None) -> tuple[bool, list[str]]`.
- Stores `manifest["control_inputs"]` as `{name: {"path": str, "sha256": str}}`.
- `_run_resume(r, resume_sid, *, control_inputs=None, ...)` remains callable with its existing two positional arguments for compatibility.

- [ ] **Step 1: Write failing tests for manifest persistence and mismatch rejection.**

```python
def test_manifest_records_control_input_hashes(tmp_path):
    r = _runner(tmp_path, lambda **k: None)
    r.prepare_session()
    r.record_manifest(
        opponent_hashes={i: "h" + str(i) for i in range(1, 17)},
        build_hashes={i: "b" + str(i) for i in range(1, 17)},
        ifelse_sha="IF", judge_sha="JD", code_hashes={},
        control_inputs={"protocol": {"path": "/p.json", "sha256": "p"},
                        "roster": {"path": "/r.json", "sha256": "r"}},
    )
    manifest = json.loads((r.session_dir / "manifest.json").read_text())
    assert manifest["control_inputs"]["protocol"]["sha256"] == "p"


def test_resume_rejects_control_input_hash_change(tmp_path):
    manifest = _valid_manifest(tmp_path)
    manifest["control_inputs"] = {
        "protocol": {"path": "/p.json", "sha256": "old"},
        "roster": {"path": "/r.json", "sha256": "same"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    ok, errors = verify_session_for_resume(
        tmp_path,
        expected_control_inputs={"protocol": "new", "roster": "same"},
    )
    assert not ok and any("control input hash mismatch" in error for error in errors)
```

- [ ] **Step 2: Run the focused tests and confirm the new assertions fail.**

Run: `PYTHONPATH=src python -m pytest -q tests/miracle/test_matrix_runner.py::test_manifest_records_control_input_hashes tests/miracle/test_section2_strictness.py::test_resume_rejects_control_input_hash_change`

Expected: FAIL because the manifest does not store or verify `control_inputs`.

- [ ] **Step 3: Implement forward-compatible manifest storage and read-only verification.**

Store `control_inputs` only when supplied so old callers remain valid. When `expected_control_inputs` is supplied, reject missing entries, unexpected entries, or mismatched SHA-256 values without writing any file.

- [ ] **Step 4: Run the focused and existing strictness tests.**

Run: `PYTHONPATH=src python -m pytest -q tests/miracle/test_matrix_runner.py tests/miracle/test_section2_strictness.py`

Expected: PASS.

- [ ] **Step 5: Commit the session identity change.**

```bash
git add src/agentbench_frame/games/miracle/matrix_runner.py \
        tools/miracle_matrix.py tests/miracle/test_matrix_runner.py \
        tests/miracle/test_section2_strictness.py
git commit -m "fix(24_miracle): persist control input identities"
```

### Task 3: Make opponent resolution deterministic and verify Python entries

**Files:**
- Modify: `tools/miracle_matrix.py:55-98`
- Create: `tests/miracle/test_matrix_identity.py`

**Interfaces:**
- Produces `resolve_unique_dir(root: Path, pattern: str) -> Path`.
- Produces `resolve_opponent_dir(rank: int, roster: dict, *, extracted_root: Path, precheck_root: Path, rank16_build_root: Path) -> Path`.
- Produces `verify_python_strategy_hashes(strategy: dict, extracted_root: Path, archives_root: Path | None = None) -> list[str]`.
- Extends `verify_hashes(v3, roster, *, extracted_root=EXTRACTED, archives_root=ARCHIVES, precheck_root=PRECHECK_9A, rank16_build_root=RANK16_BUILD) -> list[str]`.

- [ ] **Step 1: Write failing tests for zero/multiple Python directories and modified `main.py`.**

```python
def test_resolve_unique_dir_rejects_ambiguous_matches(tmp_path):
    (tmp_path / "rank04__a").mkdir()
    (tmp_path / "rank04__b").mkdir()
    with pytest.raises(RuntimeError, match="multiple opponent directories"):
        resolve_unique_dir(tmp_path, "rank04__*")


def test_verify_hashes_detects_modified_python_entry(tmp_path):
    python_dir = tmp_path / "rank04__fixture"
    python_dir.mkdir()
    (python_dir / "main.py").write_text("modified\n")
    roster = {"strategies": [{"rank": 4, "type": "python_script",
                               "entry": "main.py",
                               "runnable_sha256": "expected"}]}
    errors = verify_python_strategy_hashes(roster["strategies"][0], tmp_path)
    assert any("rank04" in error and "runnable sha" in error for error in errors)
```

- [ ] **Step 2: Run the identity tests and confirm they fail.**

Run: `PYTHONPATH=src python -m pytest -q tests/miracle/test_matrix_identity.py`

Expected: FAIL because the resolver accepts `next(glob)` semantics and Python entry hashes are not checked.

- [ ] **Step 3: Implement deterministic resolution and Python hash checks.**

Use sorted matches and require exactly one extracted directory. For Python strategies require the declared `entry`, require its file to exist, and compare its SHA-256 with `runnable_sha256`. Also verify the uniquely matched archive bytes against `archive_sha256` when `archives_root` is available; report missing or ambiguous archive matches as identity errors. Keep the existing C++ `main.exe` hash behavior unchanged.

- [ ] **Step 4: Run the identity and existing matrix tests.**

Run: `PYTHONPATH=src python -m pytest -q tests/miracle/test_matrix_identity.py tests/miracle/test_matrix_runner.py tests/miracle/test_pr1_expanded.py`

Expected: PASS.

- [ ] **Step 5: Commit the identity verification change.**

```bash
git add tools/miracle_matrix.py tests/miracle/test_matrix_identity.py
git commit -m "fix(24_miracle): verify deterministic opponent identities"
```

### Task 4: Integrate the CLI, preserve compatibility, and verify the complete flow

**Files:**
- Modify: `tools/miracle_matrix.py:100-215`
- Modify: `tests/miracle/test_section2_strictness.py:319-433`
- Create: `tests/miracle/test_matrix_cli_flow.py`

**Interfaces:**
- `main(argv: Sequence[str] | None = None) -> int` consumes parsed paths and modes.
- CLI path options are `--protocol`, `--roster`, `--session-root`, `--judge-dir`, `--ifelse-dir`, `--extracted-root`, `--archives-root`, `--precheck-root`, and `--rank16-build-root`.
- New sessions perform control-input preflight before `MatrixRunner.prepare_session()`.
- Resume passes current control-input hashes to `verify_session_for_resume()` before opening `matrix.full.log`.

- [ ] **Step 1: Write failing CLI flow tests.**

```python
def test_main_missing_protocol_returns_preflight_error_without_session(tmp_path, monkeypatch, capsys):
    rc = main(["--dry-run", "--protocol", str(tmp_path / "missing.json"),
               "--roster", str(tmp_path / "roster.json"),
               "--session-root", str(tmp_path / "sessions")])
    assert rc == 2
    assert "protocol file missing" in capsys.readouterr().err
    assert not (tmp_path / "sessions").exists()


def test_cli_resume_verifier_receives_control_hashes(tmp_path, monkeypatch):
    mm = _import_cli_module(monkeypatch)
    sid = "sid"
    session_dir = tmp_path / sid
    session_dir.mkdir()
    monkeypatch.setattr(mm, "SESSION_ROOT", tmp_path)
    observed = {}

    def fake_verify(_session_dir, **kwargs):
        observed.update(kwargs)
        return False, ["stop"]

    monkeypatch.setattr(mm, "verify_session_for_resume", fake_verify)

    class FakeRunner:
        def __init__(self):
            self.session_dir = session_dir

    rc = mm._run_resume(
        FakeRunner(), sid,
        control_inputs={"protocol": {"path": "/p.json", "sha256": "p"},
                        "roster": {"path": "/r.json", "sha256": "r"}},
    )
    assert rc == 2
    assert observed["expected_control_inputs"] == {"protocol": "p", "roster": "r"}
```

- [ ] **Step 2: Run the CLI flow tests and confirm they fail.**

Run: `PYTHONPATH=src python -m pytest -q tests/miracle/test_matrix_cli_flow.py tests/miracle/test_section2_strictness.py`

Expected: FAIL because `main()` only scans `sys.argv`, reads control files before validation, creates sessions before preflight, and does not pass control hashes to resume verification.

- [ ] **Step 3: Refactor `main()` to use parsed arguments and preflight before side effects.**

Pass explicit paths into the resolver closure and `MatrixRunner`. For new sessions, call `load_control_inputs()` before `prepare_session()`, then store protocol and roster paths/hashes in `record_manifest()`. For resume, load and validate control inputs before `verify_session_for_resume()`, pass their hashes, and keep the existing verify-before-write ordering. Convert `PreflightError` into a single stderr line and return code `2`.

- [ ] **Step 4: Run focused flow tests and the complete test suite.**

Run:

```bash
PYTHONPATH=src python -m pytest -q tests/miracle/test_matrix_cli_inputs.py \
    tests/miracle/test_matrix_identity.py tests/miracle/test_matrix_cli_flow.py \
    tests/miracle/test_matrix_runner.py tests/miracle/test_section2_strictness.py
PYTHONPATH=src python -m pytest -q
python -m compileall -q src tests tools vendor
git diff --check
```

Expected: all focused and full tests pass; compileall exits `0`; `git diff --check` has no output.

- [ ] **Step 5: Commit the integrated CLI change.**

```bash
git add tools/miracle_matrix.py tests/miracle/test_section2_strictness.py \
        tests/miracle/test_matrix_cli_flow.py
git commit -m "fix(24_miracle): make matrix CLI reproducible"
```

### Task 5: Final review and publish the PR branch

**Files:**
- Review only: all files changed by Tasks 1-4
- No changes: `/home/wkj/projects/AgentBenchFrame` local `main`

- [ ] **Step 1: Inspect the final diff and verify only the intended worktree changed.**

Run: `git status --short --branch && git diff --stat origin/pr-3...HEAD && git diff --check origin/pr-3...HEAD`

Expected: only the PR #3 repair branch has changes, with no untracked or unrelated files.

- [ ] **Step 2: Run the clean-checkout-equivalent verification.**

Run: `PYTHONPATH=src python -m pytest -q` from the PR worktree, followed by `python -m compileall -q src tests tools vendor`.

Expected: zero test failures and zero compilation errors.

- [ ] **Step 3: Push the repair branch to PR #3's head branch without updating local main.**

```bash
git push origin HEAD:gongheng/24-miracle-review1-framework
```

- [ ] **Step 4: Report exact verification counts and the PR update.**

Do not claim completion unless the fresh commands above show zero failures; report any environment-only skips separately.
