# Core Integration Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复框架当前已复现的环境注册、MCP 导出、交替先手统计和 RL runner 空训练问题，并为每个修复建立可重复的回归验证。

**Architecture:** 保持现有模块边界。环境包负责内置环境注册，MCP 包负责公共 API 导出，Match 按 Agent 身份而不是底层玩家编号统计，BaseRLRunner 仅在内置 Generals 环境和可选依赖满足时委托给 PPOTrainer，在其他环境保留受 timestep 预算约束的通用 fallback，并将已完成 PPO episode 回写到 Run。

**Tech Stack:** Python 3.11+, standard-library `unittest`, optional PyTorch, existing `src` layout.

## Global Constraints

- 不改变公开的 `BaseEnv`、`BaseAgent`、`MatchResult` 基本接口。
- 不把 PyTorch 变成硬依赖；无 PyTorch 时必须保留可解释的 fallback 行为。
- 每个行为修复先添加会失败的回归测试，再写最小实现。
- 修改范围限定在 `framework` 工作树的框架代码和测试，不改动 `ci-report` 工作树。

### Task 1: 自动注册内置环境并导出 MCP 工厂

**Files:**
- Modify: `src/agentbench_frame/env/__init__.py`
- Modify: `src/agentbench_frame/mcp/__init__.py`
- Test: `tests/test_core_regressions.py`

**Interfaces:**
- `from agentbench_frame.env import make_env` must create `generals` without caller-side registration.
- `from agentbench_frame.mcp import create_default_server` must import and return an `MCPServer`.

- [x] **Step 1: Write failing tests**

```python
def test_generals_is_registered_by_default():
    from agentbench_frame.env import ENV_REGISTRY, make_env
    assert "generals" in ENV_REGISTRY
    assert make_env("generals").game_name == "Generals"

def test_default_mcp_server_is_public_api():
    from agentbench_frame.mcp import MCPServer, create_default_server
    assert isinstance(create_default_server(), MCPServer)
```

- [x] **Step 2: Run tests and verify the failures are the missing registration/export**

```bash
PYTHONPATH=src python -m unittest tests.test_core_regressions -v
```

- [x] **Step 3: Implement the two package-level exports**

Register `GeneralsEnv` under `"generals"` after importing the registry helper, and re-export `create_default_server` from `mcp/__init__.py`.

- [x] **Step 4: Run the focused tests and verify they pass**

```bash
PYTHONPATH=src python -m unittest tests.test_core_regressions.CoreIntegrationTests.test_generals_is_registered_by_default tests.test_core_regressions.CoreIntegrationTests.test_default_mcp_server_is_public_api -v
```

### Task 2: 修复交替先手下的 Match 统计

**Files:**
- Modify: `src/agentbench_frame/arena/match.py`
- Test: `tests/test_core_regressions.py`

**Interfaces:**
- `MatchResult.agent1_wins`, `agent2_wins`, `win_rate`, and average rewards count the configured Agent identities even when starting sides alternate.
- Per-game raw `winner` remains the environment player id for compatibility; `winner_agent` remains the identity mapping.

- [x] **Step 1: Add a deterministic fake environment test**

Use a one-step fake environment whose raw winner is always player 1. With alternating starts across two games, the first game belongs to Agent 2 and the second to Agent 1, so the expected result is one win each.

- [x] **Step 2: Run the focused test and verify current code reports two Agent 2 wins**

```bash
PYTHONPATH=src python -m unittest tests.test_core_regressions.CoreIntegrationTests.test_match_counts_agent_identity_when_starts_alternate -v
```

- [x] **Step 3: Count wins and rewards from Agent identity**

Update only the aggregation branch in `Match.run`; retain raw player ids in `game_results`.

- [x] **Step 4: Run the focused test and verify it passes**

### Task 3: 让 BaseRLRunner 进入 PPO 训练路径并保存结果

**Files:**
- Modify: `src/agentbench_frame/runner/rl_runner.py`
- Modify: `src/agentbench_frame/runner/base.py`
- Modify: `src/agentbench_frame/tracking/run.py` only if an extra-summary API is needed
- Test: `tests/test_core_regressions.py`

**Interfaces:**
- If PyTorch/NumPy and the built-in `GeneralsEnv` are available, `BaseRLRunner._execute` delegates to `PPOTrainer` with the configured timestep budget.
- PPO completed episodes are written through `Run.log_episode`; otherwise the runner uses a generic episode fallback and calls `agent.learn` with the collected episode batch.
- The generic fallback never starts a step beyond `total_timesteps`.
- Runner-specific result fields are present in the persisted `summary.json`, not only in the returned in-memory dict.

- [x] **Step 1: Add tests for PPO delegation, fallback learning, capability gating, episode metrics, timestep budget, and persisted result fields**

Patch the runner module’s PPO trainer with a recording fake, assert the configured timestep budget is passed, and exercise `BaseRunner.run` with a fake result to assert that result fields are written to `summary.json`.

- [x] **Step 2: Run focused tests and verify they fail because PPO is not called and runner results are written too late**

```bash
PYTHONPATH=src python -m unittest tests.test_core_regressions.CoreIntegrationTests.test_rl_runner_delegates_to_ppo tests.test_core_regressions.CoreIntegrationTests.test_runner_persists_execute_result -v
```

- [x] **Step 3: Implement the smallest delegation and persistence changes**

Use `PPOConfig`/`PPOTrainer` only when available and supported; make `Run.finish` accept optional summary extras or otherwise persist the merged result before closing.

- [x] **Step 4: Run focused tests and then the complete test suite**

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

### Task 4: Final verification

**Files:**
- No additional production changes unless a failing regression requires one.

- [x] Run `PYTHONPATH=src python -m compileall -q src tests`.
- [x] Run CLI help and a one-step environment/Match smoke test.
- [x] Inspect `git diff` and `git status`; leave unrelated worktree content untouched.
