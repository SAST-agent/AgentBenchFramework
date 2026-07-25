# Trajectory KL 与 Replay-based KL 方案决策

> 日期：2026-07-25
>
> 状态：Trajectory KL 继续作为当前方案；replay-based KL 记录为暂缓方案，不进入本轮实现。

## 1. 决策摘要

当前 framework 继续使用实际 rollout 上的局部策略 KL trace，并按
episode 求和得到 trajectory KL：

```text
local_policy_kl_t
= KL(π_k(·|z_t) || π_{k-1}(·|z_t))

trajectory_kl_episode
= Σ_t local_policy_kl_t
```

每个 episode 的 `trajectory_kl_episode` 单独保存并按 episode/迭代顺序
展示。`mean_local_policy_kl = trajectory_kl_episode / decision_steps` 只作为
长度归一化的辅助统计，不能替代 trajectory KL，也不能标成
`nats / episode`。

Replay-based KL 的方案已完成概念讨论，但由于它要求统一不同游戏的
回放语义、恢复 stateful agent、冻结 replay/parser/policy artifact 身份，
并建立只读概率查询协议，本轮暂不实现。

## 2. 当前生效的 Trajectory KL

对于版本对 `v_{k-1} -> v_k`，在一个实际 rollout 中，只记录目标 agent
真实到达的决策上下文：

```text
z_0, z_1, ..., z_{T-1}
```

游戏/agent 接入方必须在每个 `z_t` 上提供：

- 完整、有限、顺序稳定的合法动作支持集；
- `π_k(·|z_t)` 与 `π_{k-1}(·|z_t)` 的完整概率分布；
- RL 与 HL 共用的动作编码；
- benchmark 固定的 epsilon 测量参数。

Framework 不区分 RL、HL 或其他策略内部实现。所有科研模式 agent 都返回
概率分布；HL 的 one-hot 分布由 HL 自己提供，framework 不根据动作或规则
类型推断分布。

实际 rollout 使用一个在线对照 session，而不是历史 replay：

```text
new policy decide(observation, legal support)
-> 同一次调用返回 selected action id + π_k
-> old policy 在同一 observation/legal support 上只读返回 π_{k-1}
-> framework 执行 new policy 的 selected action
-> new/old session 接收同一条实际 environment transition
```

旧策略不产生环境动作，也不生成另一条轨迹。它只沿着新策略实际产生的
轨迹维护自己的测量状态。若 agent 有内部记忆，其 adapter 必须保证概率
查询不提交一个未执行的旧策略动作；内部状态只能按照实际发生的外部事件
推进。无法满足该约束的版本对必须标记为 `incomplete`，不能退回到从动作
猜测概率。

每个决策支持集由下游运行时显式提供，至少包含：

- 非空且在动作 schema 内稳定的字符串 `action_id`；
- 与 `action_id` 一一对应、实际传给环境的 action payload；
- 非空的 `action_schema_version`；
- 对当前决策点完整、有限、无重复的支持集。

策略分布使用 `action_id -> probability` 映射，键集合必须与支持集严格相等。
framework 不接受只按长度对齐但无法核对动作身份的科研测量。

原始事实源是：

```text
local_policy_kl_trace
= [local_policy_kl_0, ..., local_policy_kl_{T-1}]
```

派生量是：

```text
trajectory_kl_episode = sum(local_policy_kl_trace)
mean_local_policy_kl  = trajectory_kl_episode / T
```

前者单位为 `nats / episode`，后者单位为 `nats / decision`。主曲线保留
每个 episode 的 trajectory KL，不先对全部 episode 求一个全局平均。

在固定初始分布、环境转移核和对手策略，并由新策略 rollout 提供测量轨迹
时，episode trace 的期望对应 forward trajectory KL 的链式分解。若接入方
使用其他 rollout 分布，必须在 measurement metadata 中明确记录，不能仍
声称是 `KL(P_k || P_{k-1})`。

`occupancy_shift` 继续作为独立辅助量，不能与 trajectory KL 相加。

## 3. Replay-based KL 讨论方案

讨论过的 replay-based KL 方案将游戏运行与 KL 计算分开：

```text
rollout collection
-> 保存完整、不可变的 replay
-> 之后选择任意两个策略版本
-> 将同一 replay 只读重放给两个版本
-> 在 replay 中每个历史决策点查询两个概率分布
-> framework 计算 local KL 与 episode 汇总
```

其优点是：

- 能覆盖历史 rollout 中出现过的全部决策点；
- rollout 采集与 KL 计算解耦；
- 同一 replay corpus 可以复算不同版本对；
- 可以对历史状态上的遗忘和回归进行离线分析。

讨论中的职责边界是：

- 下游游戏运行时负责解析本游戏 replay、标记决策点、恢复外部事件顺序、
  枚举合法动作；
- agent adapter 负责只读重放历史并返回概率分布；
- framework 负责身份校验、概率校验、epsilon、KL、存储和报告；
- 下游不能直接返回最终 KL，以免绕过统一测量协议。

## 4. 本轮暂缓原因

Replay-based KL 不是一个轻量 adapter 即可可靠完成的功能。至少还需解决：

1. 不同 Saiblo 游戏使用 JSON、JSONL、二进制录像、Judge 重放或自定义
   executable，回放语义无法由一个通用 parser 推断。
2. Stateful agent 不能仅凭单个 observation 恢复；必须保存并重放完整外部
   事件历史和实际执行动作。
3. Replay、replay adapter、策略版本、动作编码和合法支持集都需要不可变
   身份与版本校验。
4. Python、C++ 和独立进程 agent 需要统一的只读查询传输协议。
5. 完整动作支持和双策略概率向量可能很大，需要 artifact、压缩和审计策略。
6. 任一决策点无法恢复或概率支持不一致时，整条 episode measurement 的
   完整性语义需要单独设计。

在这些边界未完整设计和验证前，加入 replay-based KL 会让“可重放”看似
通用，实际由不同下游各自解释，反而降低科研可比性。

## 5. 当前不做的事情

本轮不新增：

- `GameReplayAdapter`；
- 历史 replay `PolicyQuerySession`（本轮新增的在线对照 session 不属于它）；
- replay corpus manifest；
- 历史 reference KL；
- 对任意旧策略版本的离线重放；
- 从历史动作频率估计策略概率。

现有历史 trajectory 仍可作为训练、调试和原始证据保存，但不自动进入
trajectory KL 计算。

## 6. 重新启动 Replay-based KL 的条件

只有同时满足下列条件后才重新进入设计：

- 至少两个结构差异明显的 Saiblo 游戏提供真实 replay 样本；
- 至少一个 stateful HL 和一个 RL agent 能在只读重放中返回完整分布；
- replay 身份、adapter 身份、策略身份和动作支持身份均可冻结；
- 跨语言查询协议和失败完整性语义已确定；
- 存储规模与隐私边界可接受。

在此之前，framework 的科研主口径保持为实际 rollout 上的 trajectory KL。
