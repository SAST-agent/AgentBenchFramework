# Framework PR 自动审查 Check

该仓库的 PR 检查包含两个 required checks：

- `Framework PR checks / framework-tests`：在 PR 合并引用上运行完整 pytest。
- `Framework PR checks / ai-pr-review`：读取 PR diff，调用外部审查 endpoint。

两个 check 任意失败都应阻止合并。workflow 位于 `worktree/framework` 基础分支，只监听目标为该分支的 PR。

## GitHub 配置

在仓库 Settings → Secrets and variables → Actions 中配置：

### Repository Secrets

- `PR_REVIEW_API_KEY`：API Bearer token。
- `PR_REVIEW_ENDPOINT`：完整 HTTP POST endpoint，例如 OpenAI Responses 或 Chat Completions endpoint。

### Repository Variables

- `PR_REVIEW_MODEL`：审查模型名，必填。
- `PR_REVIEW_API_MODE`：`responses` 或 `chat_completions`；不配置时默认为 `responses`。

API key 不应写入 workflow、代码、PR 描述或普通变量。workflow 只在模型审查步骤注入该 Secret。

## Endpoint 请求与响应

Responses 模式发送 `model`、`instructions`、`input` 和 JSON object text format；Chat Completions 模式发送 `model`、`messages` 和 JSON object response format。两种响应都会被转换为同一审查文档：

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

`decision` 只能是 `pass` 或 `fail`；严重级别只能是 `P0`、`P1`、`P2`、`P3`。`decision=fail` 或任意 P0/P1 finding 会使 check 失败。HTTP 错误、超时、缺少配置、非法 JSON、字段非法也会失败。

## 安全边界

模型审查 job 只 checkout PR 的 trusted base revision，并通过 GitHub API 读取 diff；它不会执行 PR head 中的代码。测试 job 可以执行 PR 合并引用，但不接触 API key。

PR diff 最大发送 350,000 字节；超过上限时仍发送前缀，并在请求中设置 `diff_truncated=true`。审查结果只写入 Actions Summary 和 workflow annotations，不自动修改 PR 或提交代码。

Fork PR 默认无法读取仓库 Secrets。当前策略是 fail-closed：如果没有审查凭据，`ai-pr-review` 会失败，管理员需要批准可信执行或配置适用的仓库策略后才能合并。

## 分支保护

在 Branch protection rules 中将以下 checks 设为 required：

```text
Framework PR checks / framework-tests
Framework PR checks / ai-pr-review
```
