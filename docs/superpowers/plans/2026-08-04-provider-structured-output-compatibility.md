# Provider Structured Output Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit provider capability mode that preserves strict local JSON validation when a Responses API proxy rejects native output-schema requests.

**Architecture:** `ProviderConfig` owns the reproducible capability choice. `CodexSessionProvider.supports_structured_output` derives from that choice, allowing the controller's native-schema and workspace-file branches to remain authoritative. Provider preflight and invocation metadata expose the mode for auditability.

**Tech Stack:** Python 3.12, dataclasses, pytest, YAML, Codex CLI Responses API.

## Global Constraints

- `structured_output_mode` accepts only `native_schema` and `validated_file`.
- The default is `native_schema`.
- Mode selection is explicit and never changes automatically during an act.
- `validated_file` retains the existing strict planner and reducer artifact validators.
- AntWar2 `sub2api` runs use `validated_file`.
- Provider credentials remain environment-only and never enter frozen configuration or logs.

---

### Task 1: Provider Configuration and Command Capability

**Files:**
- Modify: `src/agentbench_frame/hl/config.py`
- Modify: `src/agentbench_frame/hl/provider.py`
- Test: `tests/hl/test_config.py`
- Test: `tests/hl/test_provider.py`

**Interfaces:**
- Produces: `ProviderConfig.structured_output_mode: str`
- Produces: `CodexSessionProvider.supports_structured_output: bool`
- Produces: provider preflight and invocation metadata key `structured_output_mode`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_structured_output_mode_is_strict_and_serializable():
    native = ProviderConfig()
    file_mode = ProviderConfig(structured_output_mode="validated_file")
    assert native.structured_output_mode == "native_schema"
    assert file_mode.structured_output_mode == "validated_file"
    with pytest.raises(ValueError, match="structured_output_mode"):
        ProviderConfig(structured_output_mode="auto")
```

- [ ] **Step 2: Write failing provider capability tests**

```python
def test_validated_file_mode_disables_native_schema_arguments(tmp_path):
    provider = CodexSessionProvider(
        _provider_config(structured_output_mode="validated_file"),
        run_root=tmp_path,
    )
    assert provider.supports_structured_output is False
    assert provider.preflight()["structured_output_mode"] == "validated_file"
```

The test executable fixture supplies the existing version/features responses so preflight remains offline.

- [ ] **Step 3: Run focused tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/hl/test_config.py tests/hl/test_provider.py -q`

Expected: failures for the missing config field, validation, capability property, and metadata.

- [ ] **Step 4: Implement the minimal provider capability**

```python
@dataclasses.dataclass(frozen=True)
class ProviderConfig:
    structured_output_mode: str = "native_schema"

    def __post_init__(self) -> None:
        if self.structured_output_mode not in {"native_schema", "validated_file"}:
            raise ValueError(
                "provider.structured_output_mode must be native_schema or validated_file"
            )
```

In `CodexSessionProvider.__init__`, set:

```python
self.supports_structured_output = (
    config.structured_output_mode == "native_schema"
)
```

Add `structured_output_mode` to preflight facts and to every `ProviderInvocation.metadata` result in completed and timeout paths.

- [ ] **Step 5: Run focused tests and confirm success**

Run: `.venv/bin/python -m pytest tests/hl/test_config.py tests/hl/test_provider.py -q`

Expected: all tests pass.

### Task 2: AntWar2 Compatibility Configuration and Controller Contract

**Files:**
- Modify: `configs/hl/30_antwar2-positive-control.yaml`
- Test: `tests/hl/test_local_config.py`
- Test: `tests/hl/test_controller.py`

**Interfaces:**
- Consumes: `ProviderConfig.structured_output_mode`
- Consumes: `CodexSessionProvider.supports_structured_output`
- Preserves: `load_branch_briefs(path, expected_count=4, known_code_symbols=symbols, required_entry_symbol=entry_symbol)`

- [ ] **Step 1: Write failing AntWar2 configuration test**

```python
def test_antwar2_proxy_uses_validated_file_structured_output(monkeypatch):
    config = LocalHLConfig.load("configs/hl/30_antwar2-positive-control.yaml")
    assert config.run.provider.structured_output_mode == "validated_file"
```

- [ ] **Step 2: Add controller regression coverage**

Use the existing fake non-structured provider fixture to assert that a valid workspace `.agentbench/branch_briefs.json` advances to candidate rollout, while a missing or malformed file raises before any candidate provider invocation.

- [ ] **Step 3: Run focused tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/hl/test_local_config.py tests/hl/test_controller.py -q`

Expected: AntWar2 mode assertion fails until YAML is configured; strict file-validation tests remain green or expose a regression.

- [ ] **Step 4: Configure AntWar2**

Add under `run.provider`:

```yaml
structured_output_mode: "validated_file"
```

- [ ] **Step 5: Run focused tests and confirm success**

Run: `.venv/bin/python -m pytest tests/hl/test_local_config.py tests/hl/test_controller.py -q`

Expected: all tests pass.

### Task 3: Full Verification and Live Planner Resume

**Files:**
- Verify: `src/agentbench_frame/hl/`
- Verify: `tests/hl/`
- Runtime artifact: `.agentbench/30_antwar2/runs/from-scratch-k4-net3-20260804/`

**Interfaces:**
- Consumes: frozen run config comparison on resume
- Produces: a valid `proposals/iter-000001/branch_briefs.json`

- [ ] **Step 1: Run the complete HL test suite**

Run: `.venv/bin/python -m pytest tests/hl -q`

Expected: all tests pass.

- [ ] **Step 2: Implement and verify the frozen-run compatibility requirement**

Normalize a frozen run that lacks the field to `native_schema`. Add `resume --allow-provider-compatibility-change`; it permits only the `structured_output_mode` difference, rejects any additional difference, leaves `run-config.json` immutable, and appends a `provider_compatibility_selected` event with `field`, `frozen_value`, `active_value`, and `reason`.

- [ ] **Step 3: Run one AntWar2 planner act**

Run the existing `resume --acts 1 --allow-provider-compatibility-change` command with `AGENTBENCH_SAST_ROOT` and `ANTWAR2_POSITIVE_CONTROL_ROOT` set. Confirm the Codex command omits `--output-schema`, writes `.agentbench/branch_briefs.json`, and the controller persists four valid briefs.

- [ ] **Step 4: Continue the k=4 cycle**

Track four candidate acts, smoke tests, learning matches, reducer selection, Experience Skill update, policy measurements, and reporting panel. Confirm that only a completed cycle adds integer iteration 1 to IG, Elo, win-rate, and dense-margin curves.

- [ ] **Step 5: Commit implementation**

```bash
git add src/agentbench_frame/hl/config.py src/agentbench_frame/hl/provider.py configs/hl/30_antwar2-positive-control.yaml tests/hl/test_config.py tests/hl/test_provider.py tests/hl/test_local_config.py tests/hl/test_controller.py
git commit -m "fix: support validated-file provider outputs"
```
