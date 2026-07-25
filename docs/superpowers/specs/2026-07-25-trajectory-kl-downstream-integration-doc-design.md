# Trajectory KL 下游接入文档设计

## 目标

为 Saiblo 游戏运行时和游戏 agent 的接入方提供一份可独立阅读、可照着实施、
可验收的 trajectory KL 接入手册。读者不需要先阅读科研讨论文档，也不需要
自行推导 KL、episode 汇总或持久化格式。

## 交付物

1. 新建 `docs/integration/trajectory-kl.md` 作为正式接入手册。
2. 在 `docs/research/information-gain-design.md` 的在线测量接口章节增加链接，
   将数学定义与工程接入说明分开。
3. 不修改 trajectory KL 运行时代码、事件 Schema 或 CI 计算逻辑。

## 读者与前置条件

主要读者是：

- 为某个 Saiblo 游戏实现 framework adapter 的开发者；
- 为 RL、HL 或混合 game-playing agent 暴露策略分布的开发者；
- 使用自定义对局 runner、不能直接使用内置 `Match` 的开发者；
- 负责验证实验协议和数据完整性的研究人员。

读者应当已经能运行一个 episode，并能取得目标 agent 的 observation、合法
动作、实际 action、环境 transition 和终局信号。

## 文档结构

正式手册按执行顺序组织：

1. **接入结果和责任边界**：明确 framework 自动完成什么、下游必须提供什么。
2. **术语和数据流**：说明新策略实际 rollout、旧策略只读查询、对手如何影响
   trajectory 的生成但不直接进入 local KL 公式。
3. **完整动作支持集**：定义 `ActionCandidate`、`ActionSupport`、稳定 action
   ID、schema version、动态合法动作和不允许静默近似的失败语义。
4. **RL/HL adapter**：给出完整接口和代码；RL 返回合法动作 mask 后的归一化
   分布，HL 按已确定协议返回 one-hot。
5. **在线旧版本 session**：解释 `distribution_for_measurement()` 必须只读，
   以及 stateful agent 如何通过真实 transition 与新策略轨迹同步。
6. **两种 runner 接入路径**：
   - 使用内置 `Match` 的最小接线；
   - 使用 Saiblo 自定义 runner 时完整的 episode 生命周期和异常处理。
7. **持久化与报告**：使用
   `on_episode_complete=run.log_trajectory_kl_result`，说明一手事件和派生字段。
8. **实验协议**：固定或记录对手集合、seed、先后手、版本身份、epsilon 和
   episode 调度；避免把对手改变造成的状态访问变化误判成学习变化。
9. **错误语义和排障表**：列出 invalid support、分布不对齐、旧策略查询失败、
   空 episode、runner 异常和写入失败的结果。
10. **验收清单**：分别验证单决策、完整 episode、incomplete episode、
    RL/HL、一手记录、报告缺口和可复现性。

## 示例要求

所有示例使用当前公开接口：

```text
ActionCandidate(action_id, action)
ActionSupport(actions, schema_version)
PolicyDecision(action_id, probabilities)
TrajectoryKLConfig(version_before, version_after, epsilon, metadata={})
TrajectoryKLAgent(
    active_policy,
    reference_policy,
    support_provider,
    config,
    on_episode_complete,
)
Run.log_trajectory_kl_result(result)
```

示例必须满足：

- action ID 与环境 action payload 明确分离；
- 新策略在一次调用中返回动作和同一决策的分布；
- 旧策略不能向环境提交动作；
- RL 和 HL 使用相同的 `action_id -> probability` 数据格式；
- 自定义 runner 在每一个实际环境 step 后同步 transition，包括对手动作；
- `terminated` 或 `truncated` 正确结束 episode；
- 异常 episode 调用 `abort_episode()`，不丢弃已有一手数据；
- 不展示 replay-based KL、历史动作频率估计或候选动作子集近似。

## 科学表述

手册将主量写为：

```text
epsilon_regularized_local_kl_sum_under_new_policy_occupancy
```

单位为 `nats / episode`；`mean_local_policy_kl` 仅作为
`nats / decision` 辅助量。固定 trajectory 后，对手不直接出现在 local KL
公式中；对手通过改变状态访问、决策次数和 episode 长度影响实际 trajectory。

任何目标决策点缺失严格分布、完整 support 或旧策略查询时，episode 必须为
`incomplete`，主标量和均值均缺失，报告保留缺口。

## 验证

交付前执行：

1. 对照 `src/agentbench_frame/eval/measurement.py`、
   `src/agentbench_frame/eval/trajectory_kl.py`、
   `src/agentbench_frame/arena/match.py` 和
   `src/agentbench_frame/tracking/run.py` 核验所有签名与字段。
2. 搜索文档中的占位符、旧接口和 replay-based KL 误导性表述。
3. 检查相对链接和 Markdown 格式。
4. 运行现有 framework 全量测试，证明纯文档改动未引入仓库问题。

## 非目标

- 不为具体 Saiblo 游戏实现动作枚举器。
- 不新增 RL/HL 基类或自动推断策略分布。
- 不实现 replay-based KL。
- 不设计 benchmark 测试集内容。
- 不修改本地 `main`，只在 `worktree/framework` 提交并推送。
