---
name: miracle-harness
description: 运行 Miracle（第 24 届）官方逻辑对战、保存回放，并比较新旧确定性策略的严格 KL 状态
---

# Miracle 对战 Harness

所有命令在 `AgentBenchFramework` 根目录执行。唯一入口是：

```bash
uv run python -m agentbench_frame.miracle <match|replay|ig|loop>
```

## 1. Agent 接口

在 `src/agentbench_frame/miracle/agent_bridge.py` 中实现 `MiracleAgent`：

- `choose_cards(camp)` 返回 `{"artifacts": [名称], "creatures": [名称, 名称, 名称]}`；
- `act(obs)` 返回 `{"operation_type": 类型, "operation_parameters": 参数}`；
- 在同一文件末尾的唯一注册表 `AGENTS` 中加入 `"名称": AgentClass`。

CLI 每次启动时读取 `AGENTS`，不需要修改其他 choices 或入口。内置 `endround` 只结束回合，
`sample` 是可运行示例；它们不是待迭代策略的历史版本。

## 2. 运行对战

```bash
uv run python -m agentbench_frame.miracle match \
  --agent0 sample \
  --agent1 endround \
  --seed 11 \
  --output-dir agentbench_data/replays/24_miracle \
  --tag sample-vs-endround
```

- `agent0` 是先手，`agent1` 是后手；严谨比较应交换双方再跑一次；
- `seed` 固定官方逻辑的地图类型和昼夜随机值；
- `output-dir` 同时保存官方 `.mrc` 和可读 `.mrc.trace.jsonl`；
- stdout JSON 给出双方、winner、scores、rounds、terminated_by、errors 和两个回放路径；
- `terminated_by=normal` 且 `errors=[]` 才是完整正常对局。

代码调用不要求注册：

```python
from agentbench_frame.miracle import run_match
from my_agent import CandidateAgent, OpponentAgent

result = run_match(
    CandidateAgent(), OpponentAgent(),
    seed=11,
    replay_dir="agentbench_data/replays/24_miracle",
    tag="candidate-vs-opponent",
)
print(result.winner, result.scores, result.replay_path, result.trace_path)
```

## 3. 解析官方回放

```bash
uv run python -m agentbench_frame.miracle replay \
  --path agentbench_data/replays/24_miracle/<match>.mrc \
  --jsonl agentbench_data/replays/24_miracle/<match>.events.jsonl
```

命令打印 winner、终局回合和事件计数；`--jsonl` 可选，写出逐事件时间线。
分析局面和 Agent 实际收发内容时读取配套 `.mrc.trace.jsonl`。字段与事件数字含义见
`miracle-replay-reader` Skill。

## 4. 比较一次策略更新

先用更新前或更新后的策略完成对战，取得 `match` 输出中的 `trace` 路径。然后让旧、新策略
在同一份真实 observation 序列上重新决策：

```bash
uv run python -m agentbench_frame.miracle ig \
  --trace agentbench_data/replays/24_miracle/<match>.mrc.trace.jsonl \
  --old llm_v0 \
  --new llm_v1 \
  --camp 0 \
  --iteration 1 \
  --output-dir agentbench_data/ig/24_miracle
```

`old` 和 `new` 都是 `AGENTS` 中的注册名。命令不会修改游戏，也不会要求 Agent 输出概率；
它只调用现有 `act(obs)`。同一 trace 的另一阵营需要另跑一次并改为 `--camp 1`。

输出包括：

- `iteration-0001/<episode>-camp0.json`：逐决策记录，含 observation 指纹、trace 序号、
  新旧动作、状态及缺失原因；
- `ig_curve.json`：汇总输出目录内所有 iteration/episode，版本名与数据对齐；
- stdout：本 episode 的路径和四个核心数值。

严格口径如下：

- `unchanged_ratio`：新旧确定性动作相同；严格 KL 为 0；
- `infinite_ratio`：动作不同；旧策略对新动作的概率为 0，严格 KL 发散；
- `missing_ratio`：动作不在该 observation 的有限合法支持集内，或 Agent 返回无法解析；
- `finite_kl_mean`：只平均真实有限 KL。当前确定性接口通常只有 0，若全是变化或缺失则为
  `null`，不得用动作距离、胜率变化等指标代替。

有限动作支持集由当前 observation 生成，覆盖 `endround`、`surrender`、合法召唤、移动、
攻击和神器使用。官方逻辑仍是实际对战的最终仲裁者；官方 `WindBlessing` 接受无界坐标，
Benchmark 为保持支持集有限只纳入地图内坐标，支持集外动作如实记为 `missing`。

## 5. 运行完整 LLM 迭代闭环

复制并编辑 `examples/miracle-loop.toml`，然后运行：

```bash
export OPENAI_API_KEY=<API key>
uv run python -m agentbench_frame.miracle loop \
  --config examples/miracle-loop.toml \
  --data-dir ../AgentBenchResults
```

`loop` 使用 OpenAI-compatible `POST /v1/chat/completions`。当前是最小的完整源码模式，
尚未启用 Tool Calling。Harness 把当前策略、这两个 Miracle Skill、选中的结构化回放、上轮指标
和累计 budget 放入 messages。LLM 必须返回：

```json
{
  "analysis": "本轮发现的问题和修改理由",
  "strategy_code": "定义 CandidateAgent 的完整 Python 源码"
}
```

`CandidateAgent` 必须继承 `MiracleAgent` 并实现 `choose_cards(camp)`、`act(obs)`。不要返回
diff、命令或只包含方法片段的代码。Harness 自己保存、加载和评测源码，不把 `v0/v1/v2`
加入 `AGENTS`。

### 配置与预算

- `agent`：本次被测 LLM/Agent 的 Results 目录名；
- `initial_strategy`：定义初始 `CandidateAgent` 的源码，相对配置文件解析；
- `opponent`：`AGENTS` 中的固定评测对手；
- `evaluation.seeds/seats`：每个版本的对齐评测集合；
- `max_iterations`：LLM 策略更新次数，不含 iteration 0；
- `max_rollouts`：实际启动的对战总数；
- `max_episode_reads`：提供给 LLM 的回放 episode 总数；
- `max_decision_reads`：提供给 LLM 的 observation/decision 总数；
- `max_total_tokens`：API 返回 usage 的累计 token 上限；
- `max_wall_seconds`：整个 Run 的累计墙钟时间上限。

Run 是一次完整测评；iteration 是一次策略更新尝试；episode 是一局完整对战；round 是游戏内
回合。无效代码、API 错误、超时、性能倒退都保留，不从曲线中删除。

### 保存位置

设置 `AGENTBENCH_DATA=/path/to/AgentBenchResults` 或使用 `--data-dir` 后，输出为：

```text
runs/24_miracle/<agent>/<run_id>/
├── run.toml
├── summary.json
├── events.jsonl
├── score_curve.json
├── ig_curve.json
├── skills/
└── iterations/
    ├── iteration-0000/strategy.py
    └── iteration-0001/
        ├── llm_request.json
        ├── llm_response.json
        ├── candidate.py
        ├── strategy.py
        ├── episodes/
        └── ig/
```

`summary.json` 保存 `raw`、`final_evo`、`final_gain`、最佳 iteration、AUC、失败计数和
累计 budget。`events.jsonl` 是顺序日志，可用于事后重算 episode-read、decision-read、
rollout、token 和耗时曲线。API key 永不写入这些文件。

## 6. 最小验收

```bash
uv run --with pytest python -m pytest \
  tests/miracle/test_cli.py \
  tests/miracle/test_smoke.py \
  tests/miracle/test_replay.py \
  tests/miracle/test_decision_space.py \
  tests/miracle/test_ig.py \
  tests/miracle/test_loop_config.py \
  tests/miracle/test_llm_client.py \
  tests/miracle/test_strategy_loader.py \
  tests/miracle/test_run_store.py \
  tests/miracle/test_score.py \
  tests/miracle/test_loop.py
```

不要把 `agentbench_data/` 当源码提交；它是每次对战可重新生成的运行产物。
