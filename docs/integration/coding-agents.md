# Codex / Claude Code 接入

framework 把一次完整 provider invocation 记作一个 `coding_agent_act`。调用方只需要提供 prompt 和 workspace；controller 会负责：

- 在调用前后生成 workspace manifest/hash/diff，并记录 `version_before` / `version_after`；
- 将一次 invocation 计为一次 coding-agent act；
- 读取 provider 暴露的 prompt/output token、工具调用数、耗时和退出状态；
- 将 stdout JSONL 原样写到 `runs/.../provider_output/<act_id>.jsonl`；
- 将可解析的 usage 和 provider 元数据写入 `coding_agent_act` 事件。

```python
from agentbench_frame.tracking import CodexProvider, ClaudeCodeProvider, Run

run = Run.start("game", "agent", run_type="rule_iter", data_dir="./data")
controller = run.create_coding_agent_controller(CodexProvider())
record = controller.run_act(
    {"prompt": "inspect the workspace and implement the next rule", "workspace_root": "."},
    workspace_root=".",
    version_before="v0",
)
run.finish()
```

Claude Code 使用同一个 controller：

```python
controller = run.create_coding_agent_controller(ClaudeCodeProvider())
record = controller.run_act(
    {"prompt": "inspect the workspace and implement the next rule", "workspace_root": "."},
    workspace_root=".",
)
```

默认权限是 Codex `workspace-write` 和 Claude Code `acceptEdits`，不会自动开启危险的全权限模式。认证、CLI 安装和真实运行环境由调用方负责；provider 没有暴露的 usage 保留为 `None` / `unknown`，不填零。

适配依据是官方非交互接口：[Codex exec JSONL](https://developers.openai.com/codex/non-interactive-mode)、[Codex SDK](https://developers.openai.com/codex/sdk)、[Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code/cli-usage)。

本仓库不内置具体 benchmark 测试集；benchmark version、case、opponent、seed 和结果由评测方另行提供并保存在 run 数据中。
