# Framework PR 自动审查 Check

该仓库的 PR 检查包含两个 check：

- `Framework PR checks / framework-tests`：在 PR 合并引用上运行完整 pytest。
- `Framework PR checks / ai-pr-review`：读取 PR diff，调用外部审查 endpoint。

两个 check 任意失败时都会返回失败；只有在仓库保护规则中将它们设为 required 后，失败才会阻止合并。workflow 监听所有 PR 目标分支，但目标分支必须已经包含该 workflow 文件；长期接收 PR 的分支需要同步同一份 workflow。

## GitHub 配置

在仓库 Settings → Secrets and variables → Actions 中配置：

### Repository Secrets

- `PR_REVIEW_API_KEY`：API Bearer token。
- `PR_REVIEW_ENDPOINT`：完整 HTTP POST endpoint，例如 OpenAI Responses 或 Chat Completions endpoint。

### Repository Variables

- `PR_REVIEW_MODEL`：审查模型名，必填。
- `PR_REVIEW_API_MODE`：`responses` 或 `chat_completions`；不配置时默认为 `responses`。

如果使用 OpenAI GPT-5.6 Sol，填写 `PR_REVIEW_MODEL=gpt-5.6-sol`、
`PR_REVIEW_API_MODE=responses`。审查请求已经固定使用 high reasoning effort：
Responses 发送 `reasoning.effort=high`，Chat Completions 发送
`reasoning_effort=high`，不需要额外配置变量。

API key 不应写入 workflow、代码、PR 描述或普通变量。workflow 只在模型审查步骤注入该 Secret。

## Endpoint 请求与响应

Responses 模式优先发送 `model`、`instructions`、`input`、`reasoning.effort=high` 和 strict JSON Schema text format；Chat Completions 模式优先发送 `model`、`messages`、`reasoning_effort=high` 和 strict JSON Schema response format。两种响应都会被转换为同一审查文档：

```json
{
  "decision": "pass",
  "summary": "审查摘要",
  "findings": [
    {
      "severity": "P1",
      "path": "src/example.py",
      "line": 42,
      "message": "问题说明",
      "suggestion": "修改建议"
    }
  ]
}
```

结构化输出要求 `decision`、`summary` 和 `findings` 始终存在；没有可行动问题时必须返回 `findings: []`。每个 finding 都必须包含 `severity`、`path`、`line`、`message` 和 `suggestion`，其中不适用的可选值使用 `null`，`message` 必须是具体的非空说明。框架仍会在解析后再次校验这些约束，并对非法或不完整响应 fail-closed。

如果 endpoint 以 HTTP 400 或 422 明确拒绝 reasoning 或 Structured Outputs 参数，框架会对同一请求自动回退一次历史兼容格式：`json_object`，且不发送 reasoning 参数。如果 endpoint 返回 502、503 或 504 网关错误，框架会先重试一次 strict JSON Schema 但不发送 reasoning 参数；若网关仍拒绝，再回退到不带 reasoning 的 `json_object`，以应对 high reasoning 或 Structured Outputs 超过上游网关能力的情况。所有回退响应都经过同一严格解析器；其他 HTTP 错误不会回退，仍然 fail-closed。

`decision` 只能是 `pass` 或 `fail`；严重级别只能是 `P0`、`P1`、`P2`、`P3`。`decision=fail` 或任意 P0/P1 finding 会使 check 失败。HTTP 错误、超时、缺少配置、非法 JSON、字段非法也会失败。

## 安全边界

模型审查 job 只 checkout PR 的 trusted base revision，并通过 GitHub API 读取 diff；它不会执行 PR head 中的代码。测试 job 可以执行 PR 合并引用，但不接触 API key。

PR diff 最大发送 350,000 字节；超过上限时仍发送前缀，并在请求中设置 `diff_truncated=true`。审查结果只写入 Actions Summary 和 workflow annotations，不自动修改 PR 或提交代码。

Fork PR 默认无法读取仓库 Secrets。当前策略是 fail-closed：如果没有审查凭据，`ai-pr-review` 会失败，管理员需要批准可信执行或配置适用的仓库策略后才能合并。

## 首次接入 main 的两阶段 bootstrap

`main` 首次接入时必须分两步完成。第一步只把 `tools/pr_review.py`、它的测试和本文档放入 `main`，不同时放入 workflow；这是因为审查 job 必须 checkout trusted base revision，而 bootstrap PR 的 base 还没有审查工具。第一步合并后，再提交第二个 PR，把 `.github/workflows/pr-review.yml` 和 workflow 契约测试加入 `main`。第二步成功后，后续 PR 才会正常同时运行两个 check，随后才能在 `main` 上启用 required checks。

这个 bootstrap 顺序避免了让 AI job 执行 PR head 中尚未进入 trusted base 的代码，也避免了用一次性的 PR 编号特判 workflow。

## 分支保护

当前仓库已经是 public，GitHub Free 支持 Branch protection rules。最终集成分支是 `main`，研发阶段集成分支是 `worktree/framework`；两个分支都应将以下 checks 设为 required：

```text
Framework PR checks / framework-tests
Framework PR checks / ai-pr-review
```

其他实际接收 PR 的长期分支也应同步 workflow 并设置相同的 required checks；临时 feature 分支不单独设置保护规则。

开发分支先通过 PR 合并到 `worktree/framework`，稳定后再通过 promotion PR 合并到 `main`。不自动执行 merge，required checks 只负责阻止不合格 PR 合并。

由于 GitHub 不会为 workflow 加入前的历史事件自动补跑，workflow 首次进入 `main` 后，已有目标为 `main` 的 PR 需要通过新的 commit、synchronize、reopen 或其他新的有效事件重新触发检查。

公开仓库后，GitHub Actions 检查本身会正常运行；required checks 仍需在各目标集成分支的设置中启用，启用后失败的 PR 才会被 GitHub 阻止合并。
