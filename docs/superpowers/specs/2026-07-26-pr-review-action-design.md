# Framework PR 自动审查 Check 设计

## 目标

为 `SAST-agent/AgentBenchFramework` 增加自动 PR 检查，覆盖两类风险：

1. 确定性检查：测试、静态语法和数据契约检查。
2. 外部模型审查：读取 PR diff，调用支持 OpenAI Responses API 或 Chat Completions API 的 endpoint，返回结构化审查结果。

两类检查均失败即阻止 PR 合并。endpoint 不可用、认证配置缺失或返回格式非法时采用 fail-closed。

## 触发与执行边界

- workflow 放在 `worktree/framework` 基础分支，监听指向该分支的 `pull_request` 事件：`opened`、`synchronize`、`reopened`、`ready_for_review`。
- 确定性检查 checkout PR 的合并引用并运行测试；该 job 不读取外部审查密钥。
- 模型审查 job 不 checkout、不执行 PR 代码，只通过 GitHub API 读取 PR 元数据和 diff。
- workflow 权限仅为 `contents: read`、`pull-requests: read`。
- 不使用 `pull_request_target` 执行 PR 代码，避免不可信代码接触 Secrets。
- 设置 concurrency，新的提交到达时取消同一 PR 的旧检查。

## Secrets 与配置

仓库管理员在 GitHub Settings 中配置：

- `PR_REVIEW_API_KEY`：endpoint 认证密钥。
- `PR_REVIEW_ENDPOINT`：完整 HTTP endpoint。
- `PR_REVIEW_MODEL`：模型名；不写入代码，便于替换模型。
- `PR_REVIEW_API_MODE`：`responses` 或 `chat_completions`，默认 `responses`。

API key 不出现在 workflow 文件、PR diff、artifact 或日志中。Fork PR 无法读取仓库 Secrets；在严格阻止合并策略下，这类 PR 的模型审查 job 明确失败并提示管理员配置可信执行方式。

## Diff 采集与请求

模型 job 使用只读 `GITHUB_TOKEN` 访问 PR diff，不执行 PR 提供的脚本。为控制请求大小，发送完整 diff 直到固定字节上限；超出上限时附带 `diff_truncated=true` 和截断原因，不能静默丢失上下文。

请求包含：

- repository、PR number、标题、base/head SHA、PR URL；
- diff 文本和是否截断；
- 审查规则：关注正确性、科学数据完整性、可复现性、进程/Secrets 安全、向前兼容性；
- 强制返回 JSON 的 response schema。

Responses API 使用 `input`；Chat Completions 使用 `messages`。两种协议共用同一份审查指令和结果解析器。

## Endpoint 响应契约

endpoint 必须返回 JSON：

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

`decision` 只能是 `pass` 或 `fail`；`severity` 只能是 `P0`、`P1`、`P2`、`P3`。P0/P1 自动失败；endpoint 明确返回 `fail` 也失败。非法 JSON、字段缺失、未知严重级别或网络/HTTP 错误均失败。

检查结果写入 GitHub Actions job summary，并为带合法 path/line 的 finding 输出 workflow annotation；不自动修改代码、不自动提交评论。

## 确定性检查

CI 使用 Python 3.11，安装项目及 `miracle`、`report` 可选依赖和 pytest，运行完整测试集。当前项目没有 dev dependency 定义，因此安装命令显式包含 pytest；不安装 RL 的 torch，避免无关的大型依赖和平台差异。

## 验收标准

- 普通 PR 能同时产生测试 check 和模型审查 check。
- 测试失败、P0/P1 finding、`decision=fail`、endpoint 错误、响应非法、Secrets 缺失都会使对应 check 失败。
- Responses 与 Chat Completions 两种模式都能解析为同一结果契约。
- PR diff 中出现 shell 特殊字符、引号、换行或恶意标题时，不会被拼接进 shell 命令执行。
- 日志中不出现 API key；workflow 不 checkout 或执行模型审查 job 中的 PR 代码。
- `git diff --check`、workflow YAML 解析和本地响应解析测试通过。
