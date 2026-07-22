# RL/HL 迭代的科研测量方法总结

> 讨论日期：2026-07-22
>
> 主题：在统一的 MDP、概率和评测框架下，比较 RL 与 HL（heuristic learning，规则 agent 迭代）的行为变化、性能提升和资源效率。

## 1. 研究对象与术语

本研究比较两类 agent 的迭代过程：

- RL：通过参数更新得到新的策略；
- HL：通过规则、技能或规则顺序的修改得到新的策略。

首先区分两种“信息增益”：

1. **策略/行为信息增益（Policy/Behavioral Information Gain）**：衡量新旧策略在相同状态上的行为分布变化；
2. **认知/认识论信息增益（Epistemic Information Gain）**：衡量 agent 对隐藏环境参数或对手类型的不确定性降低。

当前 bench 适合实现第一种。第二种需要隐藏参数 $\Theta$、belief/posterior 和观测更新机制，目前不应把策略 KL 称为严格的 epistemic information gain。

## 2. MDP 与 episode 定义

一个 episode 定义为：

> 从一次 `env.reset()` 开始，到游戏自然终止或被明确截断为止的一条完整交互轨迹。

轨迹写作：

$$
\tau_i=(s_0,a_0,r_0,s_1,\ldots,s_{T_i-1},a_{T_i-1},r_{T_i-1},s_{T_i}).
$$

- $T_i$：第 $i$ 个 episode 的环境决策步数；
- $s_t$：第 $t$ 个动作执行前的状态；
- $a_t$：在 $s_t$ 上执行的动作；
- $s_{T_i}$：终止状态，不再产生动作。

在当前 Generals bench 中：

- 一个完整游戏对应一个 episode；
- 一次 `env.step(action)` 对应一个环境状态转移；
- 一个 action 可以是包含多个 primitive command 的 macro-action；
- macro-action 内的 command 数量不改变 environment step 的计数；
- 当前 step 语义是玩家的一次回合决策，双方行动都属于环境轨迹。

应区分：

- `terminated`：主将被消灭、投降、达到游戏规则最大回合等自然终止；
- `truncated`：外部预算、超时或实验主动停止。

如果 coding agent 是外层研究对象，还需区分：

- `coding_agent_act`：coding agent 的一次完整决策闭环；
- `env_step`：一次游戏环境转移；
- `game_agent_decision_step`：游戏 agent 的一次 `act()`。

这三个量不能混为同一个 step。

## 3. 统一的状态—动作概率空间

RL 与 HL 必须被投影到同一个状态和动作空间中。

### 3.1 决策状态

如果 agent 是严格 Markov 的，使用环境状态 $s$。

如果 HL 的规则或技能模块带有内部记忆，则决策变量应扩展为：

$$
z=(s,m),
$$

其中 $m$ 是规则/技能模块的内部状态。否则同一个可见状态可能对应不同动作，不能严格表示为 $\pi(a\mid s)$。

### 3.2 合法动作集合

定义状态 $s$ 下的合法动作集合：

$$
A(s)=\{a:a\text{ 在状态 }s\text{ 下合法}\}.
$$

RL 和 HL 的策略都必须是同一个 $A(s)$ 上的概率分布：

$$
\pi_X(\cdot\mid z)\in\Delta(A(s)),
\qquad X\in\{RL,HL\}.
$$

Generals 中的 $A(s)$ 应是规范化、完整的合法 macro-action 支持集。简化动作模板不能直接作为严格 KL 的完整支持集。

## 4. 策略信息增益定义

对于策略版本 $k-1\rightarrow k$，在共同的参考状态分布 $\nu$ 上定义局部策略信息增益：

$$
IG_k
=
\mathbb E_{z\sim\nu}
\left[
D_{KL}\left(
\pi^{k}(\cdot\mid z)
\,\middle\|
\pi^{k-1}(\cdot\mid z)
\right)
\right].
$$

使用自然对数，结果单位为 nats。这里的 KL 表示策略行为变化，不表示环境知识增加。

### 4.1 HL 的确定性策略

当前 HL 通常是确定性规则：

$$
a=f_{HL}(s).
$$

直接使用 KL 时，新旧动作不变得到 0，动作改变得到 $+\infty$，因此没有足够的比较区分度。

如果坚持使用 KL，需要统一的测量随机化层：

$$
U_s(a)=\frac{1}{|A(s)|},
$$

$$
\tilde\pi_{HL}(a\mid s)
=
(1-\epsilon)\mathbf 1[a=f_{HL}(s)]
+\epsilon U_s(a).
$$

RL 也使用同一个测量通道：

$$
\tilde\pi_{RL}(a\mid s)
=
(1-\epsilon)\pi_{RL}(a\mid s)
+\epsilon U_s(a).
$$

其中 $\epsilon$ 是统一的概率底噪比例。它是测量约定，不一定改变训练过程；如果只用于指标，应明确称为 epsilon-regularized policy KL，并报告敏感性分析。

如果不希望引入 $\epsilon$ 这一测量约定，可以使用 Jensen–Shannon divergence 或 action disagreement rate。它们能处理确定性 HL，但 JS 不具有 KL 轨迹链式分解性质。

### 4.2 episode 级信息增益

对于第 $i$ 个 episode，使用动作发生前的状态计算：

$$
IG_{k,i}
=
\frac{1}{T_{k,i}}
\sum_{t=0}^{T_{k,i}-1}
D_{KL}\left(
\pi^k(\cdot\mid z_{k,i,t})
\,\middle\|
\pi^{k-1}(\cdot\mid z_{k,i,t})
\right).
$$

终止状态不参与 KL，因为终止状态不再产生动作。

主分析保留每个 episode 的 $IG_{k,i}$；如果一轮有多个 episode，可以绘制散点、均值/中位数和置信区间。不要把所有迭代过程先压成一个全局均值。

## 5. 时间权重与 $M$ 的含义

时间权重不是信息增益的必需组成。$\lambda^t$ 是 discounted occupancy 的一种选择，表达“越早的状态权重越高”。指数形式来自每一步以固定比例衰减的重要性，也可解释为几何随机终止时间。

对于当前有限 horizon 的 Generals，主指标默认不使用时间折扣：每个决策点等权。

若需要折扣版本，可额外报告：

$$
IG_{\lambda}
=
\frac{\sum_t\lambda^t k_t}{\sum_t\lambda^t},
$$

其中 $\lambda$ 不应未经说明地复用 reward 的 $\gamma$。

$M$ 只是 episode 样本数量。如果计算：

$$
\frac{1}{M}\sum_{i=1}^{M}IG_{k,i},
$$

它描述的是评测协议下随机一个 episode 的期望平均策略变化；它是统计汇总，不是学习曲线本身。主曲线应保留 episode/迭代顺序。

## 6. Coding agent 迭代曲线

每次 coding agent act 产生一个新的 workspace/version：

$$
v_0\rightarrow v_1\rightarrow v_2\rightarrow\cdots\rightarrow v_K.
$$

定义：

$$
\text{raw\_score}=\text{score}(v_0),
$$

$$
\text{evo\_score}_k=\text{score}(v_k),
$$

$$
\text{gain}_k=\text{evo\_score}_k-\text{raw\_score}.
$$

每个 act 后的实际 score 都保留，即使性能下降；`evo_score` 不是历史最高分。历史最高分可以作为辅助统计。

coding agent 的主要学习曲线为：

$$
x=\text{coding\_agent\_act\_count},
\qquad
y=\text{benchmark\_score}.
$$

同时可绘制 score 关于累计 episode、env step、token 和 time 的曲线。

## 7. 冻结评测集与性能分

评测集按 opponent 实力递增组织，但不设置通过门槛，也不定义最高通过等级。评测集必须版本化；同一 benchmark version 内固定：

- opponent 集合；
- 地图/任务条件；
- seed 集合；
- 先手/后手安排；
- 每个等级的重复对局协议。

RL 和 HL 的不同版本必须在同一冻结评测集上评测。

主性能分直接使用评测集所有对局的胜局率，平局按半胜计入：

$$
\text{benchmark\_score}
=
\frac{W+0.5D}{W+L+D}.
$$

评测等级只用于保存分层胜率和绘制胜率—难度曲线，不参与主 score 的阈值判断。

## 8. AUC 与预算效率

AUC 定义为 benchmark score 关于累计 budget 的梯形积分：

$$
AUC_x
=
\int \text{benchmark\_score}(x)\,dx.
$$

横轴必须显式命名：

```text
AUC_coding_agent_act
AUC_episode
AUC_env_step
AUC_token
AUC_time
```

不能保存没有横轴单位的裸 `auc`。

当前研究中，coding agent 迭代效率的主横轴是 `coding_agent_act`；episode、env step、token 和 time 是辅助资源横轴。

## 9. 科研解释上的限制

1. 策略 KL 是行为变化，不是 epistemic information gain。
2. HL 的 $\epsilon$-平滑会引入测量假设，必须固定并报告 $\epsilon$。
3. 如果新旧策略在不同状态分布上评测，KL 同时混入 state distribution shift；主分析应使用固定参考状态集或明确的参考 occupancy。
4. trajectory KL 与固定状态集上的局部平均 KL 不是同一个量，不能混用名称。
5. episodic Generals 不应默认使用 stationary distribution；若使用 discounted occupancy，必须明确归一化和 finite-horizon 截断方式。
6. `episode`、`env_step`、`game_agent_decision_step` 和 `coding_agent_act` 是不同层次的量，比较时必须标明横轴。
7. 任何 score、gain 或 AUC 都只能在相同 benchmark version、seed、opponent 和先后手协议下比较。

## 10. 当前 bench 需要补齐的科研数据前提

要实现上述测量，至少需要：

- 保存每个决策前的 observation/state；
- 保存 episode、step、actor 和 termination reason；
- 提供 RL 的 action distribution 接口；
- 为 HL 提供统一随机化分布或可解释的 rule score；
- 提供完整规范化的合法动作集合 $A(s)$；
- 保存策略/version 标识，形成 $v_{k-1}\rightarrow v_k$ 配对；
- 保存冻结评测集版本和每一场原始胜负结果；
- 保存 coding agent act 与 benchmark evaluation 的对应关系。

## 11. Coding agent 数据记录责任

采用分层记录责任：

- `Bench Controller` 负责记录 act 生命周期、`act_id`、开始/结束时间、workspace diff、version、评测结果和预算累计；
- `Provider Adapter` 负责记录 provider 能提供的一手 `model_call`、`tool_call` 和 token usage；
- coding agent 不需要主动调用 bench 记录接口，可以提供可选元数据，但不作为事实来源。

一次 `coding_agent_act` 是 Bench Controller 发起的一次完整 coding-agent invocation：从发送任务上下文开始，到该次调用返回最终结果或异常结束为止。一次 invocation 内部的思考、model call、tool call 和 workspace 修改都属于同一个 act。

长会话由 controller 的 checkpoint 或显式任务完成事件切分；token 或其他 provider 数据缺失时记录为 `unknown`，不能记为 0。

## 12. 评测异常与缺失分数

`act_status`、`version_status` 和 `evaluation_status` 分开记录。agent 崩溃、超时、环境错误或评测无法完成时：

- 仍保存 act 事件和可获得的 workspace/version 快照；
- `evo_score` 和 `gain` 记为缺失值，不记为 0，也不自动记为负局；
- 只有实际完成且结果有效的对局才进入胜、负、和统计；
- CI 不对缺失分数进行静默插值，AUC 对应位置保留缺口或标记为不可计算。

评测集只有在全部固定 case 都完成并得到有效胜、负、和结果时，才生成 aggregate `benchmark_score`。任意 case 因 agent 崩溃、环境错误或基础设施超时而无结果时，`evaluation_status=incomplete`，`benchmark_score`、`evo_score`、`gain` 和对应 AUC 点均为缺失值；已完成的逐局结果仍然保存。游戏规则内部规定的超时按 benchmark 规则记为有效结果，只有外部执行失败才算无效。

## 12.1 Act 与 version 生命周期

每个 coding-agent invocation 都记录一个 act。invocation 结束后，只要 workspace 仍可安全读取，就生成 `version_after`；即使没有文件变化，也生成新的逻辑 version，但允许与前一版本拥有相同的 content hash。

如果 agent 崩溃或超时但 workspace 仍可读取，保存该快照；如果无法完成 snapshot，则 `version_after` 为缺失值，但 act 事件仍然保留。不做 accept/reject，也不自动 rollback。只有存在可运行的 `version_after` 时才启动评测。

## 13. Budget 的主次口径

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

主效率曲线和主 AUC 使用 learning-only budget；total budget 和 evaluation cost 作为辅助报告。这样不会把固定评测成本混入 agent 学习效率，同时仍保留完整实验成本。

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

## 14. Finalized-only 事件日志

采用更简单的 append-only 方案：事件只有在数据完整、最终确定后才写入 `events.jsonl`。正常协议不引入 `correction event`，也不修改或删除已经写入的事件。

如果事后发现严重记录错误，保留原始 run，另生成修正版 run 或报告；修正版必须注明来源和修正原因，不能静默覆盖旧结果。

## 15. 局部策略 KL 与 occupancy shift（最终修订）

严格来说，这里不再定义“三个并列的 KL”。对于版本对 `v_{k-1} -> v_k`，只保留两个核心对象：

```text
local_policy_kl_trace
occupancy_shift
```

`local_policy_kl_trace` 在每个实际发生的目标 agent 决策上下文 `z` 上记录：

```text
local_policy_kl_k(z)
= KL(π_k(·|z) || π_{k-1}(·|z))
```

每个 episode 保存按决策顺序排列的原始 trace：

```text
local_policy_kl_trace = [local_policy_kl_k(z_0), ..., local_policy_kl_k(z_{T-1})]
```

`z_t` 只包括该 episode 中真实到达的目标 agent 决策点；不构造不可行的全局测量域，也不把 coding agent 的 act 次数当作策略决策点。CI 再决定按 episode 求和、均值或绘制随 episode 变化的折线。

`occupancy_shift` 描述同一评测上下文下的状态访问分布变化，可以按时间步保存分布差异，也可以保存 rollout 得到的规范化 state-ID 直方图。它反映策略变化经环境动力学和对手交互后的访问变化，不能与局部策略 KL 直接相加。

在固定环境转移核和对手策略时，完整轨迹 KL 只是局部策略 KL trace 的 rollout 加权派生汇总，不是第三个独立测量量：

```text
trajectory_kl_k
= E[Σ_t local_policy_kl_k(z_t)]
```

只有将联合 state-action occupancy 定义为 `q_k(s,a)=d_k(s)π_k(a|s)` 时，才有：

```text
KL(q_k || q_{k-1})
= KL(d_k || d_{k-1})
  + E[s ~ d_k] KL(π_k(·|s) || π_{k-1}(·|s))
```

因此，任意固定参考分布下的 `policy_kl`、occupancy KL 和 trajectory KL 不能被当成三个可独立相加的指标。RL/HL 使用同一合法动作集和概率分布；occupancy 使用规范化 state ID；HL 确定性策略使用 epsilon smoothing。优先保存原始 trace 和 occupancy 数据，trajectory KL 作为可选派生字段；统一按 episode 报告时使用 `nats / episode`。

## 16. Schema 前向兼容

所有事件包含公共字段：

```text
schema_version
event_id
event_type
run_id
created_at
```

数据格式只做增量扩展，不随意重命名、删除或改变已有字段含义。新字段缺失时表示未提供，不默认填 0。旧版 CI 遇到未知字段时忽略并继续处理；遇到未知事件类型时跳过并告警，不因单个未知事件直接崩溃。只有破坏性变化才升级主 schema version。`summary.json` 是派生快照，`events.jsonl` 是事实源。

## 17. 当前落地状态

已经落地：

- `BenchmarkSpec`、逐局 `GameResult`、完整性检查、平局半胜计分和缺失 score；
- learning/evaluation/total 预算账本，未知 token/time 不转成 0；
- append-only 事件写入、公共 Schema 字段、act/version 生命周期记录；
- `local_policy_kl_trace`、`occupancy` 原始数据写入口，以及局部 KL、occupancy shift、trajectory 汇总和 AUC 纯计算函数；
- 固定 benchmark case 列表的 `BaseEvalRunner` 执行路径。
- provider-neutral 的 `ProviderAdapter`、`CodingAgentController` 和本地 workspace manifest/hash/diff snapshotter；
- canonical state ID 与可选 `get_action_distribution(observation, legal_actions)` hook。

尚未由 framework 自己决定的部分：具体 Codex/CC provider 的进程启动和 IPC、完整 CI 图表页面，以及具体环境是否能提供规范化完整动作支持集和更细粒度 state ID 编码。
