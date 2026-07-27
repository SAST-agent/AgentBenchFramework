# 全分支 PR 自动审查与 main 集成设计

## 状态

本设计已经确认，作为后续实现 workflow、分支保护和迁移工作的边界。

## 目标

1. 所有可能作为 PR 目标的远端长期分支都自动运行统一的确定性测试和 AI 审查。
2. `main` 成为最终集成分支，只有通过 required checks 的 PR 才允许合并。
3. `worktree/framework` 保留为研发阶段的中间集成分支。
4. AI 审查继续保持可信 base checkout、只读 diff、Secrets 隔离和 fail-closed 约束。

## 非目标

- 不自动替用户执行 merge。
- 不执行 PR head 中的代码来完成 AI 审查。
- 不为每个临时 feature 分支配置独立的 branch protection rule。
- 不因为 endpoint 暂时不可用而伪造审查通过。

## 当前上下文

现有 `.github/workflows/pr-review.yml` 只监听目标为 `worktree/framework` 的 PR，因此目标为 `main` 的 PR 不会触发该 workflow。PR #2 目标为 `main`，PR #3 目标为 `worktree/framework` 但在 workflow 加入前已经停止更新；PR #4 是当前自动审查链路的验证对象。

GitHub 的 PR workflow 必须存在于目标分支的基础版本中。因此删除 `branches` 过滤器是必要条件，但还必须把统一 workflow 纳入每一个长期接收 PR 的目标分支。

## 设计

### 1. 触发范围

将 workflow 的 `pull_request.branches` 过滤器删除，保留以下事件：

- `opened`
- `synchronize`
- `reopened`
- `ready_for_review`

这样 workflow 对所有目标分支生效。实际能否触发仍取决于目标分支已经包含该 workflow 文件；因此至少要维护以下长期分支中的同一份 workflow：

- `main`
- `worktree/framework`
- 任何实际接收 PR 的长期分支，例如 `worktree/ci-report`

“所有远端分支都监听”在这里指所有作为 PR base 的分支，而不是要求每个临时 head 分支都配置保护规则。

### 2. 检查 job 与安全边界

所有目标分支使用同一组稳定的 job 名称：

- `framework-tests`
- `ai-pr-review`

`framework-tests` checkout PR merge ref 并运行完整测试。`ai-pr-review` checkout PR base SHA，通过 GitHub API 读取 PR metadata 和 diff，不 checkout 或执行 PR head 代码。

AI job 继续只拥有 `contents: read` 和 `pull-requests: read` 权限。API key 只从 GitHub Secret 注入；缺少 Secret、HTTP 错误、超时、非法 JSON 或不合规 finding 都使 job 失败。

### 3. 分支保护

#### `main`

- 设置 `framework-tests` 和 `ai-pr-review` 为 required checks。
- 启用 strict required checks。
- 禁止 force push 和分支删除。
- 启用管理员也受保护规则约束。

#### `worktree/framework`

保留现有的同名 required checks、strict 检查、管理员约束、禁止 force push 和禁止删除。

#### 其他长期分支

如果某个长期分支实际接收 PR，则将同一 workflow 纳入该分支，并设置同名 required checks。临时 feature 分支默认不单独设置保护规则。

### 4. 集成与 merge 流程

开发流程分为两层：

```text
feature branch
      │ PR + tests + AI review
      ▼
worktree/framework
      │ 稳定后建立 promotion PR + tests + AI review
      ▼
main
```

合并到 `worktree/framework` 是研发阶段集成；合并到 `main` 是最终集成。promotion PR 会相对于 `main` 重新计算 diff 并重新执行审查，不复用之前的通过结果。

所有 merge 仍由人工决定，required checks 只负责阻止不合格 PR 合并。

### 5. 迁移顺序

1. 在 framework 研发分支中删除 workflow 的目标分支过滤器。
2. 将同一 workflow 纳入 `main`。
3. 通过 bootstrap promotion PR 将 workflow 合并到 `main`。在 workflow 尚未进入 `main` 前，该 PR 本身可能没有自动 check，需要人工审阅其 workflow 和变更范围。
4. workflow 进入 `main` 后，在 `main` 上设置 required checks 和分支保护。
5. 将同一 workflow 同步到其他接收 PR 的长期分支，并设置对应保护规则。
6. 对已有 PR 通过新的 commit、synchronize 或重新打开事件重新触发检查；GitHub 不会为 workflow 加入前的历史事件自动补跑。

### 6. 失败与回退

严格 JSON Schema 和 high reasoning 是 AI 审查的首选请求。对于 endpoint 能力拒绝或网关超时，工具允许有限、明确的兼容回退，但所有回退结果仍由同一解析器校验。

任何最终请求失败、响应非法或审查结论为阻塞都返回失败。不得用默认 pass、空 finding 或忽略网络错误来绕过 required check。

## 验证标准

实现完成后必须验证：

1. workflow YAML 不再只包含 `worktree/framework` 分支过滤器。
2. `framework-tests` 和 `ai-pr-review` 的 job 名称与分支保护 required contexts 完全一致。
3. 本地完整测试、YAML 解析和 `git diff --check` 通过。
4. 目标为 `main` 的测试 PR 能触发两个 job。
5. 目标为 `worktree/framework` 的测试 PR 能触发两个 job。
6. 任一 required check 失败时，GitHub 不允许合并。
7. AI job 的日志不包含 API key，也不执行 PR head 代码。

## 风险与边界

- 某个新建的远端分支如果没有 workflow 文件，不能仅靠其他分支的 workflow 保证它的 PR 触发；它必须先同步 workflow，或者成为一个已维护长期分支。
- bootstrap promotion PR 在保护规则完全生效前需要人工复核，这是将 CI 治理引入既有 `main` 的一次性迁移成本。
- endpoint 服务不可用时 required check 会失败，这是 fail-closed 设计的预期行为。
