# AgentBenchFrame Data Contract

每个人本地跑实验，框架自动产出标准化数据。push 到 `AgentBenchResults` 仓库后，CI 自动可视化。

## 目录结构

```
agentbench_data/                    # $AGENTBENCH_DATA 指向这里
└── runs/
    └── {game}/                     # 例: 28_generals, 25_lostspace
        └── {agent}/                # 例: ppo_v3, rule_expansionist
            └── {run_id}/           # 例: 20260720_1423_a1b2c3d
                ├── run.toml        # CI 必需
                ├── summary.json    # CI 必需
                └── events.jsonl    # 可选，调试用
```

## run.toml 格式 (CI 必需)

```toml
[run]
run_id = "20260720_1423_a1b2c3d"
game = "28_generals"
agent = "ppo_v3"
type = "rl"                         # "rl" | "rule_iter" | "eval"
created = "2026-07-20T14:23:00Z"   # ISO 8601 UTC
git_commit = "a1b2c3d"             # git rev-parse --short HEAD
started_at = 1753021380.0
finished_at = 1753024980.0
total_steps = 50000
total_episodes = 250

[config]                            # 可选，实验超参
learning_rate = 0.0003
batch_size = 64
```

## summary.json 格式 (CI 必需)

```json
{
  "run_id": "20260720_1423_a1b2c3d",
  "game": "28_generals",
  "agent": "ppo_v3",
  "run_type": "rl",
  "created": "2026-07-20T14:23:00Z",
  "git_commit": "a1b2c3d",
  "wall_hours": 2.5,
  "total_episodes": 250,
  "total_steps": 50000,
  "win_rate": 0.65,
  "best_elo": 1520,
  "final_elo": 1485,
  "elo_history": [
    {"step": 1000, "elo": 1350},
    {"step": 5000, "elo": 1480}
  ],
  "h2h": {
    "ppo_v3": {"expansionist": 0.65, "random": 0.92}
  },
  "resource_summary": {
    "avg_rss_mb": 245.3,
    "max_rss_mb": 512.0,
    "avg_cpu_pct": 78.5
  }
}
```

## 字段说明

| 字段 | run.toml | summary.json | 说明 |
|---|---|---|---|
| `run_id` | ✓ | ✓ | 唯一标识，格式 `{YYYYmmdd_HHMM}_{8位hex}` |
| `game` | ✓ | ✓ | 游戏标识，格式 `{届次}_{游戏名}`, 如 `28_generals` |
| `agent` | ✓ | ✓ | Agent 名称，自由命名 |
| `type` | ✓ | — | `rl` / `rule_iter` / `eval` |
| `created` | ✓ | ✓ | ISO 8601 UTC 时间 |
| `git_commit` | ✓ | ✓ | 框架代码的 git commit |
| `wall_hours` | — | ✓ | 总耗时（小时） |
| `total_steps` | ✓ | ✓ | 总交互步数 |
| `win_rate` | — | ✓ | 胜率（对 eval opponent） |
| `best_elo` | — | ✓ | 最高 Elo |
| `elo_history` | — | ✓ | Elo 变化序列 |
| `h2h` | — | ✓ | 对手胜率矩阵 |
| `resource_summary` | — | ✓ | CPU/内存聚合 |
| `config` | ✓ | — | 实验超参 |

## 使用方式

### 方式 1: 框架自动产出（推荐）

```python
from agentbench_frame.tracking import Run

run = Run.start(game="28_generals", agent="ppo_v3", run_type="rl")
env = run.wrap_env(GeneralsEnv())
agent = run.wrap_agent(my_agent)
run.start_sampler()

# 训练循环...
for episode in range(n_episodes):
    ...
    run.log_episode(reward, steps, winner)
    if episode % eval_interval == 0:
        run.log_elo(current_elo)

run.finish()
# → 自动写入 runs/28_generals/ppo_v3/{run_id}/
```

### 方式 2: 手动产出（不使用框架时）

按照上述 schema 手工创建 `run.toml` 和 `summary.json`，放到正确的目录结构下即可。

## 验证数据

```bash
# 检查数据格式是否符合 CI 预期
agentbench data check --data-dir ./agentbench_data

# 列出当前所有 run
agentbench data list --data-dir ./agentbench_data
```

## 推送数据

```bash
cd $AGENTBENCH_DATA                    # 进入数据仓库
git add runs/                          # 添加你的新 run
git commit -m "ppo_v3: iter 5, Elo 1520 (+45)"
git push

# CI 自动触发 → https://sast-agent.github.io/AgentBenchResults/
```

## 环境变量

```bash
export AGENTBENCH_DATA=/path/to/cloned/AgentBenchResults
```

框架在 `Run.start()` 时直接写入这个目录。多人协作用同一个 AgentBenchResults clone，各自 push 自己的 agent 子目录。
