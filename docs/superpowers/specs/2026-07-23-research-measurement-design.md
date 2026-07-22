# 科研测评与信息增益基础层设计

## 目标

为 RL/HL 和 coding-agent 迭代提供稳定、前向兼容、保留一手数据的 framework 基础层。provider 进程边界由统一 adapter 接入；framework 不猜测 provider 无法提供的 usage，缺失值仍记录为 unknown。

## 已确定边界

- 一个 episode 从 `reset()` 到自然终止或明确截断；`env_step` 是一次环境状态转移；目标 agent 的 act 是 `game_agent_decision_step`；外层 coding agent 的完整 invocation 是 `coding_agent_act`。
- 评测集固定并版本化。主分是全评测集胜局率，平局按半胜：`(W + 0.5D) / (W + L + D)`。不设门槛或等级通过判定。
- 每个版本都保留实际 `raw_score`、`evo_score` 和 `gain=evo_score-raw_score`，不自动 accept/reject 或 rollback。
- 主原始信息增益数据是每个真实目标 agent 决策点的 `local_policy_kl_trace`；独立辅助原始数据是 `occupancy_shift`。`trajectory_kl` 只作为派生汇总，不是第三个独立 KL。
- 事件日志为 append-only JSONL。每条事件带公共 schema 字段；新字段只增不删，未知字段可忽略，unknown usage 不转成 0。

## 方案

新增独立的 tracking/evaluation/measurement 数据类和纯函数，尽量不改变已有 runner 的调用方式：

1. `SchemaEvent`/`JSONLWriter` 负责公共事件字段、唯一 event id 和追加写入。
2. `BudgetLedger` 负责 learning/evaluation/total 的 episode、env step、token、time 计数；usage 值允许 `None` 表示 unknown。
3. `VersionedActRecorder` 负责 act 生命周期、版本 hash/diff 元数据和评测关联；snapshot 内容由调用方提供，避免假设 workspace 管理方式。
4. `BenchmarkSpec`/`BenchmarkEvaluator` 负责固定 case 清单、逐局原始结果、完整性判断和 score/raw/evo/gain。
5. `local_policy_kl`、`episode_kl_trace`、`occupancy_shift` 提供纯计算函数；策略分布和 canonical state id 由 agent/env adapter 提供。
6. wrappers 只增加可选字段与兼容的 4-tuple/5-tuple termination 解析，不强行改变既有 env API。

## 错误与兼容性

- 外部评测异常生成 incomplete 结果，不进入 W/L/D；规则定义的游戏超时仍是有效结果。
- 对 action 支持集缺失、策略分布和动作空间不一致、occupancy 样本为空等情况显式抛出 `ValueError`，不静默产生指标。
- 旧事件可被读取；新公共字段只在新 writer 生成。原有 `event` 字段保留为兼容别名，同时写入 `event_type`。
- Codex/Claude Code 的 CLI JSONL 接入通过可替换 adapter 实现；进程输出按需原样写入 provider artifact，workspace snapshot 仍由 framework controller 负责。

## 验证标准

- 纯计算函数覆盖 tie score、incomplete benchmark、AUC、epsilon smoothing 后 KL、episode trace 和 occupancy histogram。
- writer 覆盖追加写入和公共字段。
- ledger 覆盖 learning/evaluation/total 聚合与 unknown。
- version/act recorder 覆盖 unchanged version、failed act 和 version linkage。
- 既有核心集成测试继续通过。
