# 全分支 PR 自动审查与 main 集成 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让所有长期 PR 目标分支使用统一的自动测试与 AI 审查，并建立从研发分支经 promotion PR 合并到 `main` 的保护流程。

**Architecture:** 删除 PR workflow 的目标分支过滤器，使 `framework-tests` 和 `ai-pr-review` 对所有目标分支生效；将相同 workflow 同步到 `main` 及其他长期接收 PR 的分支。`worktree/framework` 作为研发集成线，`main` 作为最终集成线，两者都使用同名 required checks，最终合并仍由人工执行。

**Tech Stack:** GitHub Actions, GitHub Branch Protection API, Python 3.11, pytest, `gh` CLI, standard-library workflow contract tests。

## Global Constraints

- 所有 PR 目标分支使用稳定 job 名称 `framework-tests` 和 `ai-pr-review`。
- AI job 只 checkout trusted base SHA 并读取 PR diff，不执行 PR head 代码。
- API key 只能来自 GitHub Secret；缺少配置、endpoint 错误、超时或非法响应必须 fail-closed。
- `main` 是最终集成分支；`worktree/framework` 是研发阶段中间集成分支。
- 不自动执行 merge；required checks 只负责阻止不合格 PR 合并。
- 只有包含 workflow 文件的目标分支才能触发该 PR workflow；长期目标分支必须同步同一份 workflow。
- 不为临时 feature 分支单独创建 branch protection rule。

---

### Task 1: Make the PR workflow target-branch agnostic

**Files:**
- Modify: `.github/workflows/pr-review.yml:3-8`
- Create: `tests/test_pr_review_workflow.py`

**Interfaces:**
- Consumes: existing `Framework PR checks` workflow and its job IDs.
- Produces: a `pull_request` workflow without a `branches` filter, retaining event types `opened`, `synchronize`, `reopened`, and `ready_for_review`; job names remain `framework-tests` and `ai-pr-review`.

- [ ] **Step 1: Write the failing workflow contract tests**

Create `tests/test_pr_review_workflow.py`:

```python
from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "pr-review.yml"


def test_pr_review_workflow_listens_to_all_pull_request_base_branches():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in text
    assert "    branches:" not in text
    assert "types: [opened, synchronize, reopened, ready_for_review]" in text


def test_pr_review_workflow_keeps_required_job_names():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "  framework-tests:" in text
    assert "  ai-pr-review:" in text
    assert "name: framework-tests" in text
    assert "name: ai-pr-review" in text
```

- [ ] **Step 2: Run the new tests and verify the current workflow fails the new contract**

Run:

```bash
UV_CACHE_DIR=/tmp/agentbench-pr-review-uv-cache \
PYTHONPATH=src uv run --isolated --no-project \
  --with pytest --with psutil --with jinja2 \
  python -m pytest -q tests/test_pr_review_workflow.py
```

Expected: the first test fails because the current workflow contains `branches: - worktree/framework`.

- [ ] **Step 3: Remove only the target branch filter**

Change the trigger in `.github/workflows/pr-review.yml` from:

```yaml
on:
  pull_request:
    branches:
      - worktree/framework
    types: [opened, synchronize, reopened, ready_for_review]
```

to:

```yaml
on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
```

Do not change permissions, checkout refs, diff bounds, model configuration, job names, or the fail-closed behavior.

- [ ] **Step 4: Run the workflow contract tests and syntax checks**

Run:

```bash
UV_CACHE_DIR=/tmp/agentbench-pr-review-uv-cache \
PYTHONPATH=src uv run --isolated --no-project \
  --with pytest --with psutil --with jinja2 \
  python -m pytest -q tests/test_pr_review_workflow.py
git diff --check
```

Expected: both workflow contract tests pass and `git diff --check` reports no output.

- [ ] **Step 5: Commit the workflow change**

```bash
git add .github/workflows/pr-review.yml tests/test_pr_review_workflow.py
git commit -m "ci: review pull requests for every target branch"
```

### Task 2: Synchronize integration documentation

**Files:**
- Modify: `docs/integration/pr-review-check.md`
- Reference: `docs/superpowers/specs/2026-07-26-all-branch-pr-review-design.md`

**Interfaces:**
- Consumes: the branch-agnostic workflow from Task 1 and the approved design.
- Produces: user-facing instructions that distinguish PR listening from required protection and describe the `worktree/framework` → `main` promotion flow.

- [ ] **Step 1: Update the workflow scope paragraph**

Replace the statement that the workflow only listens to `worktree/framework` with explicit wording that it listens to all PR target branches where the workflow file exists.

- [ ] **Step 2: Document branch roles and protection**

Add these exact rules to the branch-protection section:

```text
main: final integration branch; framework-tests and ai-pr-review are required.
worktree/framework: research integration branch; the same checks remain required.
Other long-lived PR target branches: synchronize the workflow and protect them if they accept PRs.
Temporary feature branches: no separate protection rule by default.
```

- [ ] **Step 3: Document bootstrap and existing-PR behavior**

Document that the workflow must first be merged into `main`; GitHub will not retroactively create checks for PR events that occurred before the workflow existed. Explain that PR #2 needs a new eligible event after bootstrap and PR #3 needs a synchronize, reopen, or new commit event.

- [ ] **Step 4: Verify documentation consistency**

Run:

```bash
rg -n "只监听|worktree/framework|main|required|promotion|bootstrap" \
  docs/integration/pr-review-check.md \
  docs/superpowers/specs/2026-07-26-all-branch-pr-review-design.md
git diff --check
```

Expected: no statement says that only `worktree/framework` is listened to; the documented required check names remain exactly `framework-tests` and `ai-pr-review`.

- [ ] **Step 5: Commit the documentation update**

```bash
git add docs/integration/pr-review-check.md
git commit -m "docs: explain all-branch PR review and promotion flow"
```

### Task 3: Apply branch protection and prepare the main bootstrap

**Files:**
- External state: GitHub branch protection for `main`, `worktree/framework`, and any confirmed long-lived PR target branch.
- External state: a bootstrap promotion PR carrying the workflow into `main`.
- Verify only: `.github/workflows/pr-review.yml` exists in the base revision of each protected target branch.

**Interfaces:**
- Consumes: job names from Task 1 and documentation from Task 2.
- Produces: branch protection required contexts `framework-tests` and `ai-pr-review`, strict checks, no force-push/delete, and a reviewable path for installing the workflow in `main`.

- [ ] **Step 1: Verify the live branch tips before changing protection**

```bash
git fetch origin main worktree/framework
gh api repos/SAST-agent/AgentBenchFramework/branches/main/protection
gh api repos/SAST-agent/AgentBenchFramework/branches/worktree%2Fframework/protection
```

Record whether each branch already has the required contexts. Do not delete or reset any branch.

- [ ] **Step 2: Create or update protection for `main`**

Use the GitHub Branch Protection API with this exact policy shape:

```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["framework-tests", "ai-pr-review"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
```

Before applying it, confirm the repository and branch are exactly `SAST-agent/AgentBenchFramework` and `main`.

- [ ] **Step 3: Verify and repair protection for `worktree/framework`**

Confirm the same required contexts and strict setting. Preserve existing protection fields unless they conflict with the approved design; do not weaken the existing rule.

- [ ] **Step 4: Identify other long-lived target branches**

Run:

```bash
gh api repos/SAST-agent/AgentBenchFramework/branches --paginate \
  --jq '.[].name'
```

Only branches that are intentionally used as PR base branches receive the workflow and matching protection. Do not protect remote feature branches merely because they exist.

- [ ] **Step 5: Synchronize the workflow into each confirmed long-lived target branch**

For every confirmed target branch other than `main` and `worktree/framework`, create a normal synchronization PR that carries the canonical `.github/workflows/pr-review.yml` and the matching integration documentation into that branch. Do not push directly to a protected branch. Verify the synchronization PR before merging it.

- [ ] **Step 6: Prepare the bootstrap promotion PR to `main`**

After Task 1 and Task 2 are present on the research integration line, create a promotion PR whose base is `main` and whose diff includes the workflow and documentation. Since `main` did not previously contain the workflow, manually inspect the workflow, permissions, checkout refs, and Secret usage before allowing the bootstrap PR to merge.

Do not merge the bootstrap PR automatically. Its purpose is to install the workflow on `main`; subsequent PRs to `main` must receive the normal required checks.

- [ ] **Step 7: Commit migration metadata if needed**

If branch names or protection instructions change during live verification, update `docs/integration/pr-review-check.md`, run `git diff --check`, and commit only that documentation correction.

### Task 4: Validate existing and future PR behavior

**Files:**
- Verify: `.github/workflows/pr-review.yml`
- Verify: `tests/test_pr_review_workflow.py`
- Verify: GitHub Actions runs and branch protection APIs

**Interfaces:**
- Consumes: the branch-agnostic workflow and configured protection from Tasks 1–3.
- Produces: evidence that target `main` and target `worktree/framework` both run the same required checks and that failures block merges.

- [ ] **Step 1: Run the full local regression suite**

```bash
git diff --check
python -m compileall -q tools tests
UV_CACHE_DIR=/tmp/agentbench-pr-review-uv-cache \
PYTHONPATH=src uv run --isolated --no-project \
  --with pytest --with psutil --with jinja2 \
  python -m pytest -q
```

Expected: all existing tests and the new workflow contract tests pass.

- [ ] **Step 2: Push the implementation branch and inspect its checks**

```bash
GIT_SSH_COMMAND='ssh -o ConnectTimeout=20 -o ServerAliveInterval=5 -o ServerAliveCountMax=3' \
  git push origin codex/pr-review-high-reasoning
gh pr checks 4 --repo SAST-agent/AgentBenchFramework
```

Expected: both `framework-tests` and `ai-pr-review` use the exact required job names.

- [ ] **Step 3: Verify the main bootstrap PR**

After the workflow is present in `main`, create a minimal test PR targeting `main` or use the first real promotion PR. Verify both jobs run and inspect that the AI job checks the base SHA and reads the diff without executing PR head code.

- [ ] **Step 4: Verify the research integration target**

Create or update a test PR targeting `worktree/framework` and verify the same two jobs run. Confirm that a failed `framework-tests` or `ai-pr-review` check is reported as a required-check failure by GitHub.

- [ ] **Step 5: Re-trigger existing PRs without synthetic content commits**

For PR #2 and PR #3, use a legitimate new commit, synchronize event, or reopen event when available. Do not add an empty commit solely to manufacture a check. Record the resulting check URLs and conclusions.

- [ ] **Step 6: Perform final repository verification**

```bash
gh api repos/SAST-agent/AgentBenchFramework/branches/main/protection
gh api repos/SAST-agent/AgentBenchFramework/branches/worktree%2Fframework/protection
PROMOTION_PR_NUMBER="$(gh pr list --repo SAST-agent/AgentBenchFramework \
  --base main --state open --json number --jq '.[0].number')"
gh pr checks "$PROMOTION_PR_NUMBER" --repo SAST-agent/AgentBenchFramework
git status --short --branch
```

Expected: both protected branches require `framework-tests` and `ai-pr-review`; the final implementation branch is clean; no Secret value appears in logs or committed files.

### Task 5: Final documentation and handoff

**Files:**
- Verify: `docs/integration/pr-review-check.md`
- Verify: `docs/superpowers/specs/2026-07-26-all-branch-pr-review-design.md`
- Verify: `docs/superpowers/plans/2026-07-26-all-branch-pr-review-integration.md`

**Interfaces:**
- Consumes: completed implementation and live GitHub verification.
- Produces: a concise handoff with branch coverage, check URLs, protection state, and any bootstrap or existing-PR action still pending.

- [ ] **Step 1: Check that documentation matches live configuration**

```bash
rg -n "only|只监听|main|worktree/framework|framework-tests|ai-pr-review" \
  docs/integration/pr-review-check.md \
  docs/superpowers/specs/2026-07-26-all-branch-pr-review-design.md
```

Remove or correct any sentence that claims a branch is protected or covered when the live GitHub configuration disagrees.

- [ ] **Step 2: Run the final diff and status checks**

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors and no uncommitted implementation changes.

- [ ] **Step 3: Commit documentation corrections if required**

```bash
git add docs/integration/pr-review-check.md \
  docs/superpowers/specs/2026-07-26-all-branch-pr-review-design.md \
  docs/superpowers/plans/2026-07-26-all-branch-pr-review-integration.md
git commit -m "docs: finalize all-branch PR review handoff"
```
