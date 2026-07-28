# AgentBenchFrame

统一游戏 AI 实验框架，支持规则迭代、RL 训练、对抗竞技场、透明追踪与可视化。

## 安装

```bash
git clone git@github.com:SAST-agent/AgentBenchFramework.git
cd AgentBenchFramework
uv sync                          # 零硬依赖，uv 管理环境
```

可选依赖按需安装：
```bash
uv sync --extra rl               # + torch, numpy (RL 训练)
uv sync --extra report           # + jinja2 (报告生成)
uv sync --extra all              # 全部
```

## 快速开始

### Generals HL 官方引擎闭环

以下 pilot 不使用 `GeneralsEnv` 计分，而是直接调用 AgentBench 中保存的官方
Python 逻辑。先设置资产仓库和实验清单：

```bash
ASSET_ROOT=/path/to/AgentBench
MANIFEST="$ASSET_ROOT/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml"

agentbench generals prepare \
  --agentbench-root "$ASSET_ROOT" --manifest "$MANIFEST"

agentbench generals eval \
  --agentbench-root "$ASSET_ROOT" --manifest "$MANIFEST" \
  --version v0 --data-dir ./agentbench_data

agentbench generals iterate \
  --agentbench-root "$ASSET_ROOT" --manifest "$MANIFEST" \
  --data-dir ./agentbench_data --codex-executable "$(command -v codex)"
```

`iterate` 固定执行 18 局 v0 评测、12 局独立 learning replay、一次非交互
Codex act 和相同 18 局 v1 评测。对手为
`advanced-rank02-robinliu-v18`、`advanced-rank08-nashjunheng-v20` 和
`popular-rank16-xiaoaojianghu-v1`；评测 seed 为 280101/280202/280303，
learning seed 为 281101/281202/281303。

每个 run 位于
`agentbench_data/runs/28_generals/generals-hl/{run_id}/`，包含
`events.jsonl`、`summary.json`、`quality.json`、逐局
`matches/`、provider 原始 JSONL/stderr、`versions/v0`、`versions/v1`
和统一 patch。可用下面的命令校验和生成报告：

```bash
agentbench data check --data-dir ./agentbench_data
agentbench report --data-dir ./agentbench_data --output-dir ./_site
```

本地子进程限制用于实验可靠性，不是安全沙箱，也不使用 Docker。官方非法动作和
单步 TLE 是有效负局；进程崩溃、协议错误和整局 harness 超时使 case 无效，聚合
分数保持缺失。规则策略没有完整动作分布，因此严格 policy KL 明确记为不可用；
action disagreement 与 occupancy shift 分开记录。真实 act 需要本机已安装并认证
Codex CLI。

在已有、已冻结的 v3 pilot 上执行 replay-guided v4：

```bash
agentbench generals iterate-v4 \
  --agentbench-root "$ASSET_ROOT" \
  --manifest "$MANIFEST" \
  --learning-manifest \
    "$ASSET_ROOT/backend_sources/corpus/28_generals/benchmark/v4-strongest-learning-v1.toml" \
  --replay-skill \
    backend_sources/corpus/28_generals/skills/replay-analysis-v1/SKILL.md \
  --parent-run ./agentbench_data/runs/28_generals/generals-hl/PARENT_RUN_ID \
  --expected-parent-hash PARENT_VERSION_CONTENT_HASH \
  --data-dir ./agentbench_data \
  --codex-executable "$(command -v codex)"
```

v4 用三个新 seed、双座位对最强人类算法生成 6 局学习回放，从每局确定性抽取
关键决策窗口，并把人工编写的 replay skill 和实际读取计数送入一次 Codex act。
只要新策略可运行且测试通过，就一定执行 6 局配对诊断和原 18 局正式评测；
行为变化或稠密指标不作为隐藏版本的 gate。失败和退步版本同样保存，严格 policy
KL/信息增益以及跨历史缺失点的 AUC 保持缺失。

### 5 行跑一场对战

```python
from agentbench_frame.env import GeneralsEnv, EnvMode, register_env, make_env
from agentbench_frame.agent import RuleBasedAgent, RandomAgent
from agentbench_frame.arena import Match

register_env("generals", GeneralsEnv)
env = make_env("generals", mode=EnvMode.DIRECT)
a1 = RuleBasedAgent("expansionist", rules=[...])
a2 = RandomAgent("random")

match = Match(env, a1, a2)
result = match.run(n_games=100)
print(f"{result.agent1_name} win rate: {result.win_rate:.0%}")
```

### 跑一场锦标赛

```python
from agentbench_frame.arena import Arena

agents = [agent_a, agent_b, agent_c, agent_d]
arena = Arena(env, agents)
result = arena.round_robin(n_games=20)
for rank, name, elo in result.rankings:
    print(f"#{rank} {name} Elo={elo:.0f}")
```

### 带透明追踪的训练

```python
from agentbench_frame.tracking import Run

run = Run.start(game="28_generals", agent="ppo_v3", run_type="rl")
env = run.wrap_env(GeneralsEnv())
agent = run.wrap_agent(my_agent)
run.start_sampler()          # 后台采集 CPU/内存

for ep in range(1000):
    obs = env.reset()
    done = False
    while not done:
        action = agent.act(obs.to_dict())
        obs, reward, done, _ = env.step(action)
    run.log_episode(reward, obs.round_num, obs.state["winner"])
    if ep % 50 == 0:
        run.log_elo(evaluate(my_agent))   # 记录 Elo 变化

run.finish()                  # 自动写入 run.toml + summary.json
# → agentbench_data/runs/28_generals/ppo_v3/{run_id}/
```

### 推送数据到可视化

```bash
export AGENTBENCH_DATA=/path/to/AgentBenchResults
# 框架自动写入上述目录
cd $AGENTBENCH_DATA
git add runs/ && git commit -m "ppo_v3: Elo 1520" && git push
# → CI 自动聚合 → https://sast-agent.github.io/AgentBenchResults/
```

---

## 架构

```
┌──────────────────────────────────────────────┐
│                  runner/                      │  ← 策略基类（可扩展）
│  BaseRunner → BaseRLRunner / BaseRuleRunner  │
├──────────────────────────────────────────────┤
│                  training/                    │  ← 训练算法
│  PPOTrainer, RLTrainer, RuleIterator         │
├──────────────────────────────────────────────┤
│  agent/       │  skills/    │  mcp/           │  ← 决策层
│  RuleBasedAgent│ ReplayReader│ MCPTool        │
│  RLAgent      │ MapAnalyzer │ MCPServer      │
├───────────────┼─────────────┼────────────────┤
│              env/                             │  ← 环境层
│  BaseEnv, GeneralsEnv, StdioProtocol         │
├──────────────────────────────────────────────┤
│  arena/       │  eval/      │  tracking/      │  ← 评估与追踪
│  Match,Arena  │ Trajectory  │  Run,Sampler    │
└──────────────────────────────────────────────┘
```

## 核心基类与扩展

框架设计原则：所有关键行为都通过基类约束，子类只需重写 1-2 个方法。

### 1. 扩展游戏环境 — `BaseEnv`

```python
from agentbench_frame.env import BaseEnv, Observation, ActionSpace, EnvMode

class MyGameEnv(BaseEnv):
    game_name = "MyGame"
    num_players = 2

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(type="discrete", n=4)

    @property
    def observation_space(self) -> dict:
        return {"type": "dict", "keys": ["board", "score"]}

    def _reset_direct(self, seed=None) -> Observation:
        self._state = self._init_game(seed)
        return self._build_obs()

    def _step_direct(self, action) -> tuple[Observation, float, bool, dict]:
        self._apply(action)
        reward = self._calc_reward()
        done = self._is_terminal()
        return self._build_obs(), reward, done, {}
```

注册后即可使用：
```python
from agentbench_frame.env import register_env, make_env
register_env("mygame", MyGameEnv)
env = make_env("mygame", mode=EnvMode.DIRECT)
```

### 2. 扩展 Agent — `BaseAgent`

**规则 Agent**（组合已有规则）：
```python
from agentbench_frame.agent import RuleBasedAgent

def my_custom_rule(obs, state):
    if obs["state"]["round"] < 5:
        return [[1, 0, 0, 4, 2]]   # 开局 rush
    return None                      # 交给下一条规则

agent = RuleBasedAgent(
    name="rush_agent",
    skills=[MapAnalyzerSkill()],     # 可选：附加技能
    rules=[my_custom_rule, expand_rule, end_turn_rule],
)
```

**RL Agent**（自定义策略网络）：
```python
from agentbench_frame.agent import RLAgent, PolicyNetwork

class MyPolicy(PolicyNetwork):
    def predict(self, obs):
        # 你的推理逻辑
        return action_id, action_probs

agent = RLAgent(name="my_rl", policy=MyPolicy())
```

### 3. 扩展训练策略 — `BaseRunner`

```python
from agentbench_frame.runner import BaseRunner, BaseRLRunner

class MyTrainer(BaseRLRunner):
    """自定义 RL 训练器"""
    def _execute(self, env, agent):
        # env 和 agent 已被框架自动包装（追踪 + 计时 + 采样）
        for episode in range(self.config.total_episodes):
            obs = env.reset()
            done = False
            while not done:
                action = agent.act(obs.to_dict())
                obs, reward, done, _ = env.step(action)
            self._run.log_episode(reward, obs.round_num, obs.state["winner"])
            if episode % 100 == 0:
                self._run.log_elo(self._evaluate(agent))
        # run.finish() 由 BaseRunner 自动调用

# 使用
trainer = MyTrainer(env, agent, config={"total_episodes": 1000})
run = trainer.run()   # 返回 Run 对象，可直接查询
print(run.run_dir)    # agentbench_data/runs/mygame/myagent/{run_id}/
```

### 4. 扩展迭代策略 — `BaseRuleRunner`

```python
from agentbench_frame.runner import BaseRuleRunner

class EvolutionIterator(BaseRuleRunner):
    def _execute(self, env, agent):
        best = agent
        for gen in range(self.config["generations"]):
            variants = self._mutate(best)
            winner = self._evaluate_population(env, variants)
            self._run.write("generation", gen=gen, winner=winner.name,
                            win_rate=self._last_win_rate)
            if winner.win_rate > self._baseline + 0.05:
                best = winner
                self._mark_accepted(gen)     # 写 accepted 标记文件
```

### 5. 扩展技能 — `Skill`

```python
from agentbench_frame.skills import Skill, SkillMeta

class MyAnalyzer(Skill):
    def __init__(self):
        super().__init__(SkillMeta(
            name="my_analyzer", version="1.0",
            description="Custom game analysis",
            game="28_generals",
        ))

    def can_activate(self, obs, ctx):
        return obs.get("round_num", 0) % 5 == 0

    def execute(self, obs, ctx):
        ctx["analysis"] = self._analyze(obs)
        return None    # 纯信息技能，不产生动作
```

### 6. 扩展 MCP 工具 — `MCPTool`

```python
from agentbench_frame.mcp import MCPTool

class QueryDatabaseTool(MCPTool):
    def __init__(self):
        super().__init__(name="query_db", description="Query game database")

    def get_input_schema(self):
        return {"type": "object", "properties": {"query": {"type": "string"}}}

    def call(self, **kwargs):
        return {"results": db.query(kwargs["query"])}
```

---

## 数据契约

框架自动产出 CI 兼容的数据。详见 [CONVENTIONS.md](CONVENTIONS.md)。

```bash
# 验证本地数据格式
agentbench data check

# 列出所有 run
agentbench data list
```

数据目录结构：
```
$AGENTBENCH_DATA/
└── runs/{game}/{agent}/{run_id}/
    ├── run.toml         # 元信息（type, created, git_commit）
    └── summary.json     # 聚合指标（best_elo, elo_history, h2h, wall_hours）
```

## CLI

```bash
agentbench train   --game 28_generals --agent ppo_v3    # RL 训练
agentbench eval    --game 28_generals --agent ppo_v3    # 评估
agentbench iterate --game 28_generals --agent rules_v1  # 规则迭代
agentbench arena   --game 28_generals --agents a,b,c    # 锦标赛
agentbench report  --data-dir ./data --output ./_site   # 生成静态站点
agentbench mcp                                           # 启动 MCP 服务器
agentbench data check --data-dir ./data                  # 验证数据格式
agentbench data list --data-dir ./data                   # 列出所有 run
```

## 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `AGENTBENCH_DATA` | 数据根目录 | `./agentbench_data` |

## 依赖策略

- **核心零依赖** — `env`, `agent`, `arena`, `skills`, `mcp` 可直接使用
- `torch`, `numpy` — RL 训练时需要 (`[rl]` extra)
- `jinja2` — 报告生成时需要 (`[report]` extra)
- `psutil` — 资源追踪时需要 (`[tracking]` extra)
- 无对应库时自动降级，不报错
