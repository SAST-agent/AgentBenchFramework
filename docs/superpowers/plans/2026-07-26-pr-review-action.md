# Framework PR 自动审查 Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为指向 `worktree/framework` 的 PR 增加确定性测试和外部模型审查两个阻塞合并的 GitHub Checks。

**Architecture:** `.github/workflows/pr-review.yml` 负责触发、权限、测试环境和 HTTP 调用边界；`tools/pr_review.py` 负责构造 Responses/Chat Completions 请求、提取模型文本、校验结构化审查结果和输出 annotations；`tests/test_pr_review.py` 覆盖两种协议与 fail-closed 行为。模型审查 job 只 checkout 基础分支，不执行 PR 代码，通过 GitHub API 获取 diff。

**Tech Stack:** GitHub Actions, Python 3.11, Python standard library (`json`, `urllib`, `argparse`), pytest。

## Global Constraints

- 两类检查均失败即阻止 PR 合并。
- endpoint 不可用、认证缺失、HTTP 错误、响应非法时 fail-closed。
- API key 只能来自 GitHub Secret `PR_REVIEW_API_KEY`，不得写入仓库或日志。
- 模型 job 不 checkout 或执行 PR head 代码，只 checkout trusted base revision 运行审查工具。
- 支持 `responses` 和 `chat_completions` 两种 API mode，统一使用同一审查结果契约。
- 只使用 `contents: read`、`pull-requests: read` 权限。
- 不安装 RL 的 torch；测试环境安装 `.[miracle,report]` 与 pytest。

---

### Task 1: Add the test-first response contract parser

**Files:**
- Create: `tests/test_pr_review.py`
- Create: `tools/pr_review.py`

**Interfaces:**
- Produces `extract_model_text(api_mode: str, response: dict) -> str`.
- Produces `parse_review_document(text: str) -> dict`.
- Produces `review_should_fail(review: dict) -> bool`.
- Produces `build_api_request(mode: str, model: str, instructions: str, payload: str) -> dict`.

- [ ] **Step 1: Write the failing tests**

Add tests for:

```python
def test_extracts_responses_output_text_and_validates_review():
    response = {"output": [{"type": "message", "content": [
        {"type": "output_text", "text": '{"decision":"fail","summary":"bad","findings":[]}'},
    ]}]}
    review = parse_review_document(extract_model_text("responses", response))
    assert review["decision"] == "fail"
    assert review_should_fail(review) is True


def test_extracts_chat_completion_content():
    response = {"choices": [{"message": {
        "content": '{"decision":"pass","summary":"ok","findings":[]}',
    }}]}
    review = parse_review_document(extract_model_text("chat_completions", response))
    assert review == {"decision": "pass", "summary": "ok", "findings": []}


def test_rejects_unknown_severity_and_malformed_json():
    with pytest.raises(ValueError):
        parse_review_document('{"decision":"pass","summary":"x","findings":[{"severity":"P9"}]}')
    with pytest.raises(ValueError):
        parse_review_document("not json")


def test_builds_protocol_specific_json_requests():
    responses = build_api_request("responses", "review-model", "system", "diff")
    chat = build_api_request("chat_completions", "review-model", "system", "diff")
    assert responses["model"] == "review-model" and "input" in responses
    assert chat["model"] == "review-model" and "messages" in chat
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
python -m pytest tests/test_pr_review.py -q
```

Expected: collection failure because `tools/pr_review.py` does not exist.

- [ ] **Step 3: Implement the minimal pure functions**

Implement response extraction for:

- Responses: top-level `output_text`, or `output[*].content[*].text`.
- Chat Completions: `choices[0].message.content`, accepting a string or text-part list.
- Optional Markdown JSON fences are stripped before JSON parsing.

Validate the exact contract: `decision` is `pass|fail`, `summary` is a string, `findings` is a list, every finding has a valid `severity` in `P0..P3`, a string `message`, and optional string `path`, positive integer `line`, and string `suggestion`. `review_should_fail` returns true for `decision == fail` or any P0/P1 finding.

Build Responses requests with `input` and JSON object text format; build Chat Completions requests with `messages` and JSON object response format.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run:

```bash
python -m pytest tests/test_pr_review.py -q
```

Expected: all focused parser tests pass.

- [ ] **Step 5: Commit the parser and tests**

```bash
git add tools/pr_review.py tests/test_pr_review.py
git commit -m "feat: add structured PR review protocol adapter"
```

### Task 2: Add the fail-closed HTTP CLI

**Files:**
- Modify: `tools/pr_review.py`
- Modify: `tests/test_pr_review.py`

**Interfaces:**
- CLI reads `PR_REVIEW_API_KEY`, `PR_REVIEW_ENDPOINT`, `PR_REVIEW_MODEL`, `PR_REVIEW_API_MODE`, and `PR_REVIEW_INPUT_JSON`.
- CLI emits a compact summary to stdout, GitHub annotations for valid findings, and exits `0` only when the review passes.

- [ ] **Step 1: Write failing CLI tests**

Add a local `http.server` fixture and invoke the CLI through `subprocess.run`:

```python
def test_cli_sends_auth_and_passes(tmp_path):
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["authorization"] = self.headers["Authorization"]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"output_text":
                '{"decision":"pass","summary":"ok","findings":[]}'}).encode())

    with serve(Handler) as url:
        env = {**os.environ, "PR_REVIEW_API_KEY": "secret-value",
               "PR_REVIEW_ENDPOINT": url, "PR_REVIEW_MODEL": "m",
               "PR_REVIEW_API_MODE": "responses",
               "PR_REVIEW_INPUT_JSON": str(_input_file(tmp_path))}
        result = subprocess.run([sys.executable, "tools/pr_review.py"],
                                env=env, text=True, capture_output=True)
    assert result.returncode == 0
    assert seen["authorization"] == "Bearer secret-value"
    assert "secret-value" not in result.stdout + result.stderr


@pytest.mark.parametrize("env_name", ["PR_REVIEW_API_KEY", "PR_REVIEW_ENDPOINT",
                                       "PR_REVIEW_MODEL"])
def test_cli_fails_when_required_configuration_is_missing(tmp_path, env_name):
    env = {**os.environ, "PR_REVIEW_API_KEY": "k", "PR_REVIEW_ENDPOINT": "http://127.0.0.1",
           "PR_REVIEW_MODEL": "m", "PR_REVIEW_INPUT_JSON": str(_input_file(tmp_path))}
    env.pop(env_name)
    result = subprocess.run([sys.executable, "tools/pr_review.py"],
                            env=env, text=True, capture_output=True)
    assert result.returncode != 0
```

Add equivalent parameterized cases for HTTP 500, malformed model JSON, `decision=fail`, and a valid response containing a P1 finding; each must assert a nonzero exit code.

- [ ] **Step 2: Run the CLI tests to verify they fail**

Run:

```bash
python -m pytest tests/test_pr_review.py -q
```

Expected: failures for the missing CLI entry point and HTTP behavior.

- [ ] **Step 3: Implement the CLI**

Use `urllib.request` with a bounded timeout, `Authorization: Bearer ...`, `Content-Type: application/json`, and no shell interpolation. Read the request payload from a file or stdin, cap the outgoing payload before the workflow writes it, write the raw response only to an explicitly provided temporary path when requested, and never print response headers or the key. On validation failure print a sanitized error and exit nonzero.

- [ ] **Step 4: Run the CLI tests to verify they pass**

Run:

```bash
python -m pytest tests/test_pr_review.py -q
```

Expected: all parser and CLI tests pass.

- [ ] **Step 5: Commit the CLI**

```bash
git add tools/pr_review.py tests/test_pr_review.py
git commit -m "feat: make PR review checks fail closed"
```

### Task 3: Add the GitHub Actions workflow and setup documentation

**Files:**
- Create: `.github/workflows/pr-review.yml`
- Create: `docs/integration/pr-review-check.md`
- Modify: `tests/test_pr_review.py`

**Interfaces:**
- Workflow invokes `tools/pr_review.py` from the trusted base checkout.
- Required configuration is documented with exact Secret/Variable names.

- [ ] **Step 1: Write workflow contract tests**

Add a dependency-free text contract test:

```python
def test_workflow_is_pr_scoped_and_read_only():
    text = Path(".github/workflows/pr-review.yml").read_text(encoding="utf-8")
    assert "pull_request:" in text
    assert "pull_request_target" not in text
    assert "contents: read" in text and "pull-requests: read" in text
    assert "framework-tests:" in text and "ai-pr-review:" in text
    for name in ("PR_REVIEW_API_KEY", "PR_REVIEW_ENDPOINT",
                 "PR_REVIEW_MODEL", "PR_REVIEW_API_MODE"):
        assert name in text
```

Add a documentation contract test asserting the setup guide names the same four variables and the required-check names.

- [ ] **Step 2: Run the workflow contract tests to verify they fail**

Run:

```bash
python -m pytest tests/test_pr_review.py -q
```

Expected: failure because the workflow and setup document do not exist.

- [ ] **Step 3: Implement the workflow**

Create a concurrency-safe workflow with:

1. `framework-tests`: checkout the PR merge ref, install Python 3.11, install `.[miracle,report]` plus pytest, run `python -m pytest -q`.
2. `ai-pr-review`: checkout `${{ github.event.pull_request.base.sha }}` only, call the GitHub API for the PR diff, build a bounded JSON payload using environment variables, invoke `tools/pr_review.py`, and fail if configuration or review fails.

Use `env:` for all PR-derived values; do not interpolate title, branch, body, or diff into shell source. Keep the API key scoped only to the review step. The job writes findings to `$GITHUB_STEP_SUMMARY` and uses `::error` annotations for valid paths and lines.

Document repository Secrets and Variables, branch protection required-check names, API mode selection, response schema, fork-PR behavior, and how to test with a local endpoint.

- [ ] **Step 4: Run workflow and contract tests to verify they pass**

Run:

```bash
python -m pytest tests/test_pr_review.py -q
python -m compileall -q tools/pr_review.py
git diff --check
```

Expected: all tests pass and no whitespace errors are reported.

- [ ] **Step 5: Commit the workflow and docs**

```bash
git add .github/workflows/pr-review.yml docs/integration/pr-review-check.md tests/test_pr_review.py
git commit -m "ci: add blocking automated PR review checks"
```

### Task 4: Full verification and publish

**Files:**
- Verify: `.github/workflows/pr-review.yml`
- Verify: `tools/pr_review.py`
- Verify: `tests/test_pr_review.py`
- Verify: `docs/integration/pr-review-check.md`

- [ ] **Step 1: Run the complete local test suite**

Run:

```bash
python -m pytest -q
```

Expected: all available tests pass; if optional dependencies are unavailable, report the exact skipped/blocked tests rather than treating them as passed.

- [ ] **Step 2: Inspect the final diff and status**

Run:

```bash
git diff --check origin/worktree/framework..HEAD
git status --short --branch
git log --oneline -5
```

Confirm no API key, endpoint value, or temporary response is tracked.

- [ ] **Step 3: Push the framework base branch**

```bash
git push origin worktree/framework
```

Expected: GitHub receives the workflow on the PR base branch and schedules the new checks for PRs targeting `worktree/framework`.
