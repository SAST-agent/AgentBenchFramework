# Rollman HL 可复现实验

## 实验协议

`29_rollman-k4-repair.yaml` 定义四个并列初始候选、两个线性修复候选和两个 finalist。每个 planner branch 必须给出可观测触发条件与保持契约。修复候选只允许修正所属 branch，并以该 branch 的初始候选为父版本。严格优胜规则保证失败、超时、退化或打平的修复不会覆盖初始候选。

实验起点为 `run-20260802-gpt55-k4-interface-v4` 的不可变版本 `v000037`。来源运行、来源版本和内容哈希写入事件日志。API 凭据只从 `.env` 对应的 `AGENTBENCH_API_KEY` 读取，不进入配置、checkpoint、事件或报告。

## 环境

```bash
export AGENTBENCH_SAST_ROOT=/Users/qingle/Code/SAST
export MPLCONFIGDIR=/private/tmp/agentbench-mpl
```

`.env`：

```dotenv
AGENTBENCH_API_KEY=sk-...
```

## 校验

```bash
.venv/bin/python -m agentbench_frame.hl.cli validate \
  --config configs/hl/29_rollman-k4-repair.yaml

.venv/bin/python -m agentbench_frame.hl.cli run \
  --config configs/hl/29_rollman-k4-repair.yaml \
  --dry-run
```

## 启动一个 proposal cycle

```bash
.venv/bin/python -m agentbench_frame.hl.cli run \
  --config configs/hl/29_rollman-k4-repair.yaml \
  --run-dir .agentbench/29_rollman/runs/run-20260802-gpt55-k4-repair-v1 \
  --workspace .agentbench/29_rollman/candidate-k4-repair-v1 \
  --acts 1
```

一次完整 cycle 的模型调用顺序为 planner、四个初始候选、两个修复候选、reducer。比赛评测由本地冻结后端执行，不消耗模型 API token。

## 断点续跑

```bash
.venv/bin/python -m agentbench_frame.hl.cli resume \
  --config configs/hl/29_rollman-k4-repair.yaml \
  --run-dir .agentbench/29_rollman/runs/run-20260802-gpt55-k4-repair-v1 \
  --workspace .agentbench/29_rollman/candidate-k4-repair-v1 \
  --acts 1
```

恢复逻辑校验 checkpoint、版本内容哈希、branch 编号和父版本关系。通过校验的 planner、初始候选和修复候选不会重复调用模型。

## 单阶段报告

```bash
.venv/bin/python -m agentbench_frame.hl.cli report \
  --run-dir .agentbench/29_rollman/runs/run-20260802-gpt55-k4-repair-v1
```

报告包含四个主面板：信息增益（决策级局部策略 KL）、Rollman Elo、全人类池胜率和平均分差。横轴为整数 HL iteration。`branches.csv` 记录 initial 与 `repair-1` 阶段、父版本和 branch representative。

## 跨阶段聚合报告

```bash
.venv/bin/python -m agentbench_frame.hl.cli aggregate-report \
  --source-run .agentbench/29_rollman/runs/run-20260802-gpt55-k4-interface-v4 \
  --phase-run .agentbench/29_rollman/runs/run-20260802-gpt55-k4-repair-v1 \
  --origin-iteration 11
```

输出文件为 `aggregate-curves.csv`、`aggregate-curves.png` 和 `aggregate-curves.svg`。来源运行的 iteration 11 与修复运行的 imported origin 只保留一个曲线点；修复运行的第一轮记为全局 iteration 12。

## 消融参数

所有消融均通过独立配置文件表达并随运行冻结：

- `iteration.repair_enabled`: 启用或关闭反馈修复。
- `iteration.repair_top_k`: 每轮获得修复机会的初始候选数量。
- `iteration.repair_rounds`: `0` 或 `1`，控制线性修复轮数。
- `iteration.candidates_per_cycle`: rollout 数量；Rollman proposal 协议固定为 `4`。
- `iteration.scope_contract_required`: 是否强制触发条件与保持契约。
- `iteration.finalist_count`: 进入多种子 finalist 评测的代表数量。
- `iteration.quick_screen_seeds` 与 `iteration.finalist_seeds`: 快速筛选和 finalist 的比赛预算。

每组消融使用独立 run directory，保留对应的 `run-config.json`、事件日志、模型原始流、checkpoint、版本对象、比赛回放和报告。
