# RL/HL 策略信息增益：讨论结论

> 讨论日期：2026-07-22
>
> 状态：核心 framework 计算、记录、Provider 边界和本地 snapshot 接口已实现；具体 Provider 进程接入和完整 CI 展示仍待外部适配。

## 1. 目标与术语

RL 和 HL（heuristic learning，规则/启发式 agent 迭代）需要使用同一量纲、同一概率空间下的指标进行比较。

当前 bench 能够支持的首先是**策略/行为信息增益**（Policy/Behavioral Information Gain），而不是严格意义上的 epistemic information gain。后者要求 agent 维护关于隐藏环境参数或对手类型的 posterior，并计算 posterior entropy reduction；当前 bench 尚未提供这一模型。

## 2. 统一策略空间

对任意决策状态 (z)，定义相同的合法动作集合：

$$
A(s)=\{a: a\text{ 在状态 }s\text{ 下合法}\}.
$$

RL 和 HL 都必须被表示为 (A(s)) 上的概率分布：

$$
\pi_X(\cdot\mid z)\in\Delta(A(s)),
\qquad X\in\{RL,HL\}.
$$

Generals 中的一个 action 是当前环境 `step(action)` 接收的 macro-action（命令列表），因此后续需要明确一个规范化、完整的合法 macro-action 支持集。当前 `get_legal_actions()` 返回的是简化动作模板，不能直接视为严格 KL 所需的完整支持集。

HL 当前直接返回确定性动作 (a=f(s))。若规则 agent 有内部记忆，则严格的决策变量不是只有可见状态 (s)，而应为：

$$
z=(s,m),
$$

其中 (m) 是规则/技能模块内部状态。

## 3. 主指标

对于策略版本 $k-1\rightarrow k$，在共同的参考状态分布 $\nu$ 上定义：

$$
IG_k
=
\mathbb E_{z\sim\nu}
\left[
D_{KL}\left(\pi^{k}(\cdot\mid z)\,\middle\|\,\pi^{k-1}(\cdot\mid z)\right)
\right].
$$

自然对数对应 nats。这里的 KL 是策略行为变化量；它不应直接称为 agent 对环境的认知增益。

RL 使用策略本身的 action distribution。HL 若保持确定性，直接 KL 会得到：动作不变时为 0，动作改变时为 (+\infty)，因此不能用于有区分度的比较。

## 4. HL 的确定性策略处理

如果要继续使用 KL，需要一个共同的测量/执行随机化层。令：

$$
U_s(a)=\frac{1}{|A(s)|},
$$

则使用固定的 (0<\epsilon<1)：

$$
\tilde\pi_{HL}(a\mid s)
=
(1-\epsilon)\mathbf 1[a=f_{HL}(s)]
+\epsilon U_s(a),
$$

RL 也经过同一个测量通道：

$$
\tilde\pi_{RL}(a\mid s)
=
(1-\epsilon)\pi_{RL}(a\mid s)
+\epsilon U_s(a).
$$

$\epsilon$ 是均匀概率底噪的比例，不是必须加入训练过程的 exploration。若只用于计算指标，应在实验协议中称为 epsilon-regularized policy KL，并预先固定且对 RL/HL 一致；还应做 epsilon 敏感性分析。

如果不希望引入这一测量约定，可以使用 Jensen–Shannon divergence 或 action disagreement rate。它们能处理确定性 HL，但 JS 不具有 KL 的轨迹链式分解性质。

## 5. Episode 定义

一个 episode 定义为：

> 从一次 `env.reset()` 开始，到游戏自然终止或被明确截断为止的一条完整交互轨迹。

$$
\tau_i=(s_0,a_0,r_0,\ldots,s_{T_i-1},a_{T_i-1},r_{T_i-1},s_{T_i}).
$$

- (T_i)：第 (i) 个 episode 的决策步数。
- 只在 (s_0,\ldots,s_{T_i-1}) 上计算策略 KL；终止状态 (s_{T_i}) 不再产生动作。
- 当前 Generals 中，一个完整游戏对应一个 episode。
- 当前 `env.step(action)` 对应一个玩家的一次回合决策；action 本身可以包含多个命令。
- `Match.run(n_games=M)` 运行的是 (M) 个 episode。

必须区分：

- `terminated`：主将被消灭、投降、达到游戏规则最大回合等自然游戏终止；
- `truncated`：外部 timestep 预算、超时或实验主动停止。

当前环境 API 只有一个 `done`，后续若要严格统计，应记录终止原因并区分这两类结束。

## 6. 不把所有 episode 压成一个总平均

学习曲线应保留每个 episode/策略迭代的值，而不是只报告整个实验的一个 (M) 平均值。

若第 (k) 次策略迭代对应第 (e) 个评测 episode：

$$
TrajectoryKL_{k,e}
=
\sum_{t=0}^{T_{k,e}-1}
D_{KL}\left(
\pi^k(\cdot\mid z_{k,e,t})\,\middle\|\,\pi^{k-1}(\cdot\mid z_{k,e,t})
\right).
$$

长度归一化辅助量为：

$$
MeanLocalKL_{k,e}
=
\frac{TrajectoryKL_{k,e}}{T_{k,e}}.
$$

主要图像应为：

$$
x=\text{policy iteration }k,
\qquad y=TrajectoryKL_{k,e}.
$$

如果每轮有多个 episode，则保留散点，同时可绘制该轮均值/中位数和置信区间：

$$
\overline{TrajectoryKL}_k
=
\frac{1}{M_k}\sum_{e=1}^{M_k}TrajectoryKL_{k,e}.
$$

这里的 (M_k) 只是同一轮的重复实验数量，不是学习曲线的主指标。

## 7. Episode 汇总与决策归一化

随机抽取一个 episode 时的平均 trajectory KL：

$$
\frac{1}{M}\sum_i
\left(\sum_t k_{i,t}\right)
$$

描述每个 episode 的总策略轨迹变化，是主 trajectory KL 的跨 episode
统计汇总。

Episode-balanced 的平均局部 KL：

$$
\frac{1}{M}\sum_i
\left(\frac{1}{T_i}\sum_t k_{i,t}\right)
$$

描述先随机抽一个 episode、再抽取其中一个 decision 时的平均局部变化。

Decision-balanced 的平均局部 KL：

$$
\frac{\sum_i\sum_t k_{i,t}}{\sum_i T_i}
$$

描述从所有已采集 decision 中随机抽一个 decision 时的平均策略变化，长 episode 权重更大。

主曲线保留逐 episode 的 `TrajectoryKL`；后两种归一化只作为辅助分析，
不能与主值共用名称或单位。

## 8. 时间权重不是必需的

$\lambda^t$ 只是 discounted occupancy 的一种选择，不是 KL 的要求。指数权重对应“每一步以固定比例衰减重要性”，也可解释为几何随机终止时间。

对于当前有限 horizon 的 Generals，默认主指标不使用时间折扣更直观：每个决策同等重要。若要与 reward 的 discounted objective 对齐，可以额外报告 discounted trajectory KL：

$$
TrajectoryKL_{\lambda}
=
\sum_t\lambda^t k_t.
$$

若再除以 $\sum_t\lambda^t$，得到的是归一化的 discounted
`MeanLocalKL`，不是 trajectory KL。信息增益的 $\lambda$ 不应未经说明地
复用 reward 的 $\gamma$。

## 9. 评测协议要求

为了让 RL/HL 的曲线可比较，主曲线应使用：

1. 相同的初始状态/seed 分布；
2. 相同的 opponent 集合；
3. 相同的先手/后手安排；
4. 相同的 episode 终止和截断规则；
5. 相同的合法动作编码和 epsilon；
6. 相同的 rollout 生成协议，并明确记录提供测量轨迹的策略分布；只有新策略
   rollout 才能把局部 trace 的期望解释为 forward trajectory KL。

横轴可同时提供两种版本：

- policy iteration：比较学习进程；
- 累计 environment decision steps：比较数据效率。

## 10. Coding Agent 测评与迭代接口共识

本节记录后续 coding agent 测评/迭代接口的已确认边界。

### 10.1 分阶段范围

分为两个阶段：

- 第一阶段：统一迭代生命周期、版本保存、episode/step/token/time budget、冻结评测集、benchmark score、raw/evo/gain 和基础曲线数据。
- 第二阶段：统一策略分布、Policy Information Gain、AUC 和完整可视化报告。

### 10.2 冻结并版本化的评测集

评测集按 opponent 实力递增组织，但不设置通过门槛，也不计算最高通过等级。评测集版本固定并可版本化；同一版本内固定：

- opponent 集合；
- 地图/任务条件；
- seed 集合；
- 先手/后手安排；
- 每个评测等级的重复对局协议。

RL 和 HL 的每个版本都在相同的冻结评测集上重新评测。

评测集难度等级只用于分层分析和绘制胜率—难度曲线，不参与主 score 的门槛判断。

### 10.3 主性能分

主性能分命名为 `benchmark_score` 或 `evaluation_win_rate`，直接使用评测集所有对局的胜局率；平局按半胜计入：

$$
\text{benchmark\_score}
=
\frac{W+0.5D}{W+L+D}.
$$

其中 $(W,L,D)$ 分别为总胜局、总负局和总和局。每个难度等级同时保存分层胜率，但不进行等级通过/不通过判定。

### 10.4 版本与 raw/evo/gain

初始 workspace/version 为 $v_0$，其评测分为：

$$
\text{raw\_score}=\text{score}(v_0).
$$

每次 coding agent act 都产生下一版本 $v_k$，不根据性能提升与否筛选或回退。该版本在冻结评测集上的实际分数为：

$$
\text{evo\_score}_k=\text{score}(v_k).
$$

相对初始版本的变化为：

$$
\text{gain}_k
=
\text{evo\_score}_k-\text{raw\_score}.
$$

`evo_score` 表示当前 act 后版本的实际分数，不表示历史最高分。历史最高分可以作为单独的辅助字段保存。

### 10.5 Coding agent act

一次 coding agent act 定义为一个完整的 coding agent 决策闭环：

```text
接收上下文
→ 分析
→ 调用工具/修改 workspace
→ 完成本轮最终结果
```

一个 act 中的 tool call 不额外增加 act 计数，但应分别统计 tool call 数、token 和时间。主要迭代预算为：

```text
coding_agent_acts
```

同时保留：

```text
episodes_consumed
env_steps_consumed
primitive_commands
tokens_consumed
time_consumed
```

### 10.6 Step 语义

- `env_step`：一次 `env.step(action)`，即一个环境状态转移；
- `game_agent_decision_step`：目标游戏 agent 一次 `act()`；
- `coding_agent_act`：外层 coding agent 一次完整决策闭环；
- macro-action 内的 primitive command 不作为主 step，只作为辅助复杂度指标。

对于 coding agent 迭代效率，主横轴使用 `coding_agent_acts`；episode、environment step、token 和 time 作为辅助预算横轴。

### 10.7 AUC

AUC 定义为冻结评测集 `benchmark_score` 关于累计 budget 的梯形积分。根据横轴分别命名：

```text
AUC_coding_agent_act
AUC_episode
AUC_env_step
AUC_token
AUC_time
```

不能保存没有横轴单位的裸 `auc`。coding agent 迭代效率的主 AUC 使用 `AUC_coding_agent_act`；其他预算 AUC 用于辅助分析。

### 10.8 Provider-neutral 兼容性

接口不绑定 Codex、CC 或某个 LLM SDK。不同 coding agent 通过 provider adapter 接入统一协议，例如：

```text
CodexAdapter
ClaudeCodeAdapter
OtherCodingAgentAdapter
```

每次 act 的统一结果至少应能表达：

```text
act_id
provider
status
version_before
version_after
changed_files
tool_call_count
prompt_tokens
completion_tokens
total_tokens
token_accuracy
elapsed_time_s
raw_output_ref
```

token 的准确性区分 `exact`、`estimated` 和 `unknown`；unknown 不能记为 0。

### 10.9 数据记录责任与 act 边界

采用分层记录责任：

1. `Bench Controller` 负责记录 act 生命周期、`act_id`、开始/结束时间、workspace diff、version、评测结果和预算累计；
2. `Provider Adapter` 负责记录 Codex、CC 等 provider 能提供的一手 `model_call`、`tool_call` 和 token usage；
3. coding agent 不需要主动调用 bench 记录接口；它可以提供可选元数据，但不作为事实来源。

一次 `coding_agent_act` 定义为 Bench Controller 发起的一次完整 coding-agent invocation：从发送任务上下文开始，到该次调用返回最终结果或异常结束为止。一次 invocation 内部的思考、model call、tool call 和 workspace 修改都属于同一个 act。

如果 coding agent 以长会话方式运行，则由 controller 的 checkpoint 或显式任务完成事件切分 act；不能把内部 tool call 直接当作新的 act。

token 或其他 provider 数据缺失时记录为 `unknown`，不能伪装成 0。完整 prompt、response 和工具内容默认通过 artifact 引用保存，不直接塞入事件流。

### 10.10 评测异常与缺失分数

`act_status`、`version_status` 和 `evaluation_status` 分开记录。agent 崩溃、超时、环境错误或评测无法完成时：

- 仍保存 act 事件和可获得的 workspace/version 快照；
- `evo_score` 和 `gain` 记为缺失值，不记为 0，也不自动记为负局；
- 只有实际完成且结果有效的对局才进入 `W`、`L`、`D`；
- CI 不对缺失分数进行静默插值，AUC 对应位置保留缺口或标记为不可计算。

评测集只有在全部固定 case 都完成并得到有效胜、负、和结果时，才生成 aggregate `benchmark_score`。任意 case 因 agent 崩溃、环境错误或基础设施超时而无结果时，`evaluation_status=incomplete`，`benchmark_score`、`evo_score`、`gain` 和对应 AUC 点均为缺失值；已完成的逐局结果仍然保存。游戏规则内部规定的超时按 benchmark 规则记为有效结果，只有外部执行失败才算无效。

### 10.10.1 Act 与 version 生命周期

每个 coding-agent invocation 都记录一个 act。invocation 结束后，只要 workspace 仍可安全读取，就生成 `version_after`；即使没有文件变化，也生成新的逻辑 version，但允许与前一版本拥有相同的 content hash。

如果 agent 崩溃或超时但 workspace 仍可读取，保存该快照；如果无法完成 snapshot，则 `version_after` 为缺失值，但 act 事件仍然保留。不做 accept/reject，也不自动 rollback。只有存在可运行的 `version_after` 时才启动评测。

### 10.11 Budget 的主次口径

同时保存 learning-only、evaluation 和 total 三类环境预算：

```text
learning_coding_agent_acts
learning_episodes
learning_env_steps
evaluation_coding_agent_acts
evaluation_episodes
evaluation_env_steps
total_coding_agent_acts
total_episodes
total_env_steps
```

其中 `total_*` 是实际环境消耗的总量，但主效率曲线和主 AUC 使用 learning-only budget；total budget 和 evaluation cost 作为辅助报告。这样不会把固定评测成本混入 agent 学习效率，同时仍保留完整实验成本。

token 和 time 采用相同的主次口径：

```text
learning_prompt_tokens
learning_completion_tokens
learning_total_tokens
learning_time_s
evaluation_tokens
evaluation_time_s
total_tokens
total_time_s
```

provider 提供的缓存 token、reasoning token 等原始 usage 字段保留在 `model_call` 事件中；缺失 usage 记为 `unknown`，不记为 0。

### 10.12 Finalized-only 事件日志

采用更简单的 append-only 方案：事件只有在数据完整、最终确定后才写入 `events.jsonl`。正常协议不引入 `correction event`，也不修改或删除已经写入的事件。

如果事后发现严重记录错误，保留原始 run，另生成修正版 run 或报告；修正版必须注明来源和修正原因，不能静默覆盖旧结果。

### 10.13 局部策略 KL 与 occupancy shift（最终修订）

严格来说，这里不再定义“三个并列的 KL”。对于版本对 `v_{k-1} -> v_k`，只保留两个核心对象：

```text
local_policy_kl_trace
occupancy_shift
```

第一个核心对象是每个实际发生的目标 agent 决策上下文 `z` 上的局部条件策略变化：

```text
local_policy_kl_k(z)
= KL(π_k(·|z) || π_{k-1}(·|z))
```

每个 episode 保存按决策顺序排列的 trace：

```text
local_policy_kl_trace = [local_policy_kl_k(z_0), ..., local_policy_kl_k(z_{T-1})]
```

这里的 `z_t` 只包括该 episode 中真实到达的目标 agent 决策点，不需要构造全局测量域，也不把 coding agent 的 act 次数当作策略决策点。该 trace 是策略变化的原始测量数据。当前主派生量按 episode 求和：

```text
trajectory_kl_episode = Σ_t local_policy_kl_k(z_t)
```

`trajectory_kl_episode` 的单位是 `nats / episode`。长度归一化的
`mean_local_policy_kl = trajectory_kl_episode / T` 可以作为辅助统计，
其单位是 `nats / decision`，不能替代 trajectory KL。CI 保留每个
episode 的点并按 episode/迭代顺序绘图，不先压成整个实验的单一平均值。

第二个核心对象是同一评测上下文下的状态访问变化 `occupancy_shift`。它可以按时间步保存状态分布差异，或者保存由 rollout 得到的规范化 state-ID 直方图；它描述策略变化通过环境动力学和对手交互后造成的访问分布变化，不能和局部策略 KL 直接相加当作一个“总信息增益”。

在固定环境转移核和对手策略时，完整轨迹 KL 不是第三个独立测量量，而是局部策略 KL trace 在 rollout 测量域上的派生汇总：

```text
trajectory_kl_k
= E[Σ_t local_policy_kl_k(z_t)]
```

同样，只有把联合 state-action occupancy 定义为 `q_k(s,a)=d_k(s)π_k(a|s)` 时，才有链式分解：

```text
KL(q_k || q_{k-1})
= KL(d_k || d_{k-1})
  + E[s ~ d_k] KL(π_k(·|s) || π_{k-1}(·|s))
```

因此不能把任意固定参考分布下的 `policy_kl`、状态 occupancy KL 和 trajectory KL 当成三个可独立相加的指标。RL/HL 必须使用同一合法动作集和概率分布；occupancy 必须使用可比较的规范化 state ID；HL 自己提供 one-hot 分布，framework 不推断其内部逻辑；之后 RL/HL 统一使用 benchmark 固定的 epsilon smoothing。原始 trace 和 occupancy 数据优先保存，trajectory KL 是主 episode 派生量。

### 10.13.1 Replay-based KL 讨论结论

曾讨论将完整历史 rollout 固化为 replay corpus，再把同一 replay 只读重放
给任意两个策略版本，在所有历史决策点重新查询概率并计算 KL。该方案可以
解耦 rollout collection 与 KL measurement，也能分析历史状态上的遗忘。

本轮决定暂缓该方案。原因是它还需要统一不同游戏的 replay adapter、
stateful agent 的历史恢复、跨语言只读概率查询、replay/parser/policy
artifact 身份以及失败完整性语义。当前不新增 replay-based KL 接口，也不
把历史动作频率解释为策略概率。

当前生效方案仍是实际 rollout 上的 `local_policy_kl_trace` 和按 episode
求和得到的 `trajectory_kl_episode`。完整讨论与重新启动条件见
`docs/superpowers/specs/2026-07-25-trajectory-kl-replay-decision.md`。

### 10.13.2 在线测量接口

Framework 已提供 `TrajectoryKLAgent` 在线测量 wrapper。下游游戏 runtime
在每个目标 agent 决策点提供一个 `ActionSupport`：

```text
ActionSupport(
    schema_version,
    [(action_id, environment_action_payload), ...]
)
```

支持集必须完整、有限、非空、顺序稳定且 action ID 无重复。策略分布必须
使用 `action_id -> probability` 映射，键集合与支持集严格相等，概率为有限
非负数且总和在 `1e-9` 绝对误差内等于 1。

新版本 adapter 实现：

```text
decide_with_distribution(observation, support)
-> PolicyDecision(selected_action_id, new_distribution)
```

旧版本在线对照 adapter 实现：

```text
distribution_for_measurement(observation, support)
-> old_distribution
```

新版本在一次调用中同时给出动作 ID 和分布，避免独立调用 `act()` 与概率
接口造成随机性或内部状态不一致。环境只执行新版本选择的动作。两个 session
都通过可选的 `observe_transition(actual_transition)` 接收相同的真实环境
事件；旧版本不能提交一个没有实际执行的旧策略动作。

`TrajectoryKLAgent` 可把 episode 结果直接回调给
`Run.log_trajectory_kl_result()`。`policy_kl_trace` 事件保留旧字段，并增量
保存：

```text
measurement_status
direction
log_base
rollout_source
trajectory_kl_episode
mean_local_policy_kl
decisions[*].legal_action_ids
decisions[*].selected_action_id
decisions[*].new_probabilities
decisions[*].old_probabilities
decisions[*].support_id
metadata
errors
```

若任一目标决策的支持集、分布或旧策略查询无效，episode 保留已获得的一手
记录，但 `measurement_status=incomplete`，主 trajectory KL 与均值均为
缺失值，不能使用局部和冒充完整结果。

### 10.14 Schema 前向兼容

所有事件包含公共字段：

```text
schema_version
event_id
event_type
run_id
created_at
```

数据格式只做增量扩展，不随意重命名、删除或改变已有字段含义。新字段缺失时表示未提供，不默认填 0。旧版 CI 遇到未知字段时忽略并继续处理；遇到未知事件类型时跳过并告警，不因单个未知事件直接崩溃。只有破坏性变化才升级主 schema version。`summary.json` 是派生快照，`events.jsonl` 是事实源。

### 10.15 当前落地状态

已经落地：

- `BenchmarkSpec`、逐局 `GameResult`、完整性检查、平局半胜计分和缺失 score；
- learning/evaluation/total 预算账本，未知 token/time 不转成 0；
- append-only 事件写入、公共 Schema 字段、act/version 生命周期记录；
- `local_policy_kl_trace`、`occupancy` 原始数据写入口，以及局部 KL、occupancy shift、trajectory 汇总和 AUC 纯计算函数；
- 固定 benchmark case 列表的 `BaseEvalRunner` 执行路径。
- provider-neutral 的 `ProviderAdapter`、`CodingAgentController` 和本地 workspace manifest/hash/diff snapshotter；
- canonical state ID、严格 `ActionSupport`/`PolicyDecision` 契约、在线
  `TrajectoryKLAgent` 对照 session；
- 完整和 incomplete trajectory-KL 一手 JSONL 记录；
- 以 episode trace 求和为主值、局部均值为辅助值的本地报告，以及保留缺口
  的 episode 折线图。

仍由接入方决定的部分：具体 benchmark 测试集内容，以及具体环境 runtime
如何枚举本游戏完整动作支持集、冻结动作 schema、让 RL/HL adapter 提供
严格分布。Generals 当前的简化 `get_legal_actions()` 仍不能声明为完整科研
动作域。Replay-based KL 已记录但暂缓实现。Codex/Claude Code CLI JSONL
adapter、provider artifact、统一 act 生命周期、数据质量诊断和本地/CI
research report 已落地；真实运行仍需要调用方安装并认证对应 CLI。
