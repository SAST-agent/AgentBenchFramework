# Planner Diversity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a distillation-aware k=4 planner allocate candidates across opponent prediction, offensive progress, opponent-score denial, and a distinct novel mechanism.

**Architecture:** Extend only `IterationContext.build_planner_prompt`; keep the branch schema, candidate packets, provider calls, evaluators, rollback, and reporting unchanged. The extra contract is conditional on an existing shared distillation path, so ordinary non-distillation cycles retain their present behavior.

**Tech Stack:** Python 3.13, pytest, string-built Codex prompts.

## Global Constraints

- Reuse the existing content-addressed Ghost distillation artifact.
- Add no API calls and no new context files.
- Preserve the existing `branch_briefs.json` schema.
- Keep Rollman and Ghost role semantics asymmetric.
- Preserve the primitive Rollman decision space used for KL.
- Activate cached distillation after two stagnant cycles or an exploration debt of two.
- Preserve audit history when explicitly replanning an interrupted cycle.

---

### Task 1: Enforce distillation-aware branch diversity

**Files:**
- Modify: `src/agentbench_frame/hl/context.py`
- Test: `tests/hl/test_context.py`
- Verify: `tests/hl/test_context.py`, full `tests` suite

**Interfaces:**
- Consumes: `IterationContext.build_planner_prompt(..., previous_measurements: Mapping[str, Any], ...) -> str`
- Produces: the same prompt string with a conditional four-role exploration contract when `opponent_distillation_path` is present

- [ ] **Step 1: Write the failing prompt-contract test**

```python
def test_planner_distillation_assigns_offensive_and_predictive_branch_roles(tmp_path):
    ...
    assert "branch 0" in prompt
    assert "branch 1" in prompt
    assert "branch 2" in prompt
    assert "至少两支" in prompt
    assert "不能直接复制 Ghost 动作" in prompt
    assert "不得重复运行蒸馏脚本" in prompt
```

The fixture supplies an existing `opponent_distillation_path`. A companion assertion builds the prompt without that path and confirms the fixed distillation roles are absent.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.venv/bin/pytest tests/hl/test_context.py::test_planner_distillation_assigns_offensive_and_predictive_branch_roles -q`

Expected: FAIL because the prompt does not assign fixed branch roles or require offensive diversity.

- [ ] **Step 3: Add the minimal conditional prompt contract**

In the existing `if shared_distillation:` block, replace the generic “four different best responses” sentence with explicit branch responsibilities and the requirement that at least two branches optimize progress, score, finish, or opponent denial instead of primarily vetoing/retreating/waiting.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `.venv/bin/pytest tests/hl/test_context.py -q`

Expected: all context tests pass.

- [ ] **Step 5: Run the full regression suite**

Run: `.venv/bin/pytest -q`

Expected: all tests pass.

- [ ] **Step 6: Commit and resume the experiment**

```bash
git add docs/superpowers/specs/2026-08-03-planner-diversity-design.md docs/superpowers/plans/2026-08-03-planner-diversity.md tests/hl/test_context.py src/agentbench_frame/hl/context.py
git commit -m "feat: diversify distillation-aware HL planning"
```

Resume one atomic iteration from the existing run; verify the planner emits at least one predictive-distillation branch and at least one offensive branch before allowing candidate evaluation to continue.

### Task 2: Replan an interrupted cycle under the active prompt contract

**Files:**
- Modify: `src/agentbench_frame/hl/cli.py`
- Test: `tests/hl/test_cli.py`

**Interfaces:**
- Consumes: `agentbench hl resume --replan-pending`
- Produces: a resumed run that ignores pending planner, candidate, and repair recovery references without deleting audit events

- [ ] **Step 1: Make the two-cycle trigger and CLI flag tests fail**

Assert that `_opponent_distillation_required` returns `True` for stagnation/debt two and that `main()` forwards `replan_pending=True` from the resume subcommand.

- [ ] **Step 2: Implement the minimal threshold and recovery bypass**

Use a threshold of two in `_opponent_distillation_required`; add the resume-only flag and pass it to `_run_real`, where it suppresses all three pending recovery collections.

- [ ] **Step 3: Run focused and full regression tests**

Run: `.venv/bin/pytest tests/hl/test_cli.py tests/hl/test_context.py -q`

Run: `.venv/bin/pytest -q`

Expected: all tests pass before `resume --replan-pending --acts 1` is executed.
