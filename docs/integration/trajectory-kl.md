# Trajectory KL 下游接入指南

> **24_miracle 作用域。** 当前接口已统一为 atomic Judge operation、状态
> 局部完整有序的 `ActionSupport + support_id`、`KL(new||old)`、自然对数、
> 双方固定 `epsilon=0.01` 均匀 smoothing、new-policy occupancy、episode
> arithmetic mean 与 `nats / decision`。不存在 KL 大小阈值。参见
> [24_miracle KL contract authority v2](../games/24_miracle_kl_contract_authority.v2.md)。
> 真实运行与权威批准仍是独立边界。

本文面向接入具体 Saiblo 游戏、RL agent、HL（heuristic learning，
rule-based agent iteration）agent 或自定义对局 runner 的开发者。目标是让
下游只提供游戏和策略事实，framework 统一完成校验、KL 计算、一手数据记录
和可视化。

当前方案是**新策略实际 rollout 上的在线 trajectory KL**，不是完整游戏树
KL，也不是 replay-based KL。数学背景和设计取舍见
[信息增益设计](../research/information-gain-design.md)。

## 1. 接入完成后会得到什么

一次接入会为每个目标 agent episode 产生：

- 每个目标决策点的新旧原始策略分布、合法动作 ID、实际动作 ID 和 local KL；
- 主指标 `information_gain`（等于 `mean_local_policy_kl`），单位为 `nats / decision`；
- 独立派生 `local_policy_kl_sum`（兼容字段 `trajectory_kl_episode`），单位为 `nats / episode`；
- episode 的版本、epsilon、对手、seed、先后手和错误信息；
- `events.jsonl` 中的追加式一手记录；
- 本地和 CI 中按 episode 展示且不跨越缺失点的折线图。

严格 estimand 为：

```text
epsilon_regularized_local_kl_sum_under_new_policy_occupancy
```

如果任一目标决策点无法获得完整动作支持集或严格分布，该 episode 仍保存已经
取得的一手数据，但标记为 `incomplete`；主指标和辅助指标都保持缺失，不能用
局部和冒充完整 episode。

## 2. Framework 与下游的责任边界

| 责任 | 负责方 |
|---|---|
| 枚举当前决策点完整、有限的合法动作支持集 | 下游游戏 runtime |
| 为每个动作提供稳定且有语义的 `action_id` | 下游游戏 runtime |
| 在同一次调用中返回实际动作和对应分布 | 新版本 active policy adapter |
| 在同一支持集上只读返回分布 | 旧版本 reference policy adapter |
| 只执行新策略选出的环境动作 | framework 或下游 runner |
| 用每个真实 transition 同步新旧 session | 内置 `Match` 或下游 runner |
| 校验分布、执行 epsilon regularization、计算 local KL 和 episode 标量 | framework |
| 通过 `Run` callback 保存逐决策一手证据 | framework |
| 固定对手调度、seed、先后手、版本身份和 epsilon | 下游实验协议 |
| 从 trace 派生标量、绘图并保留 incomplete 缺口 | framework 和 CI |

下游不需要自行计算 KL、求 episode 总和、计算均值或生成曲线。下游也不应
把 framework 无法观测的游戏内部事实交给 framework 推断。

## 3. 测量数据流与对手的作用

设新策略为 \(\pi_{\mathrm{new}}\)，旧策略为
\(\pi_{\mathrm{old}}\)，对手为 \(\mu\)。实际轨迹的生成规律是：

$$
\tau \sim
P\left(
\tau \mid
\pi_{\mathrm{new}},
\mu,
\rho_0,
P_{\mathrm{env}}
\right).
$$

framework 只在这条实际轨迹中属于目标 agent 的决策点计算：

$$
\operatorname{TrajectoryKL}(\tau)
=
\sum_{t\in D_{\mathrm{target}}(\tau)}
D_{\mathrm{KL}}\left(
\widetilde{\pi}_{\mathrm{new}}(\cdot\mid s_t)
\middle\Vert
\widetilde{\pi}_{\mathrm{old}}(\cdot\mid s_t)
\right).
$$

其中，若当前支持集为 \(A(s_t)\)，则：

$$
\widetilde{\pi}(a\mid s_t)
=
(1-\epsilon)\pi(a\mid s_t)
+\frac{\epsilon}{|A(s_t)|},
\qquad 0<\epsilon<1.
$$

epsilon 只作用于测量层的 local KL，不改变环境实际执行的新策略动作。也就是
说，状态访问来自原始 behavior policy，局部被积函数使用 regularized 分布；
因此严格名称是“epsilon-regularized local-KL sum under new-policy
occupancy”，而不是 regularized policy 之间完整 path distribution 的 KL。

固定一条 trajectory 后，对手不直接进入 local KL 公式。但对手参与生成这条
trajectory，会改变访问状态、目标 agent 决策次数和 episode 长度。因此比较
不同迭代轮次时，应固定对手协议，或者按对手分层分析。

一次目标决策的数据流是：

```text
observation
  -> support_provider(observation)
  -> ActionSupport
  -> active.decide_with_distribution(observation, support)
  -> PolicyDecision(action_id, new_distribution)
  -> reference.distribution_for_measurement(observation, support)
  -> old_distribution
  -> framework validates and computes local KL
  -> environment executes only support.resolve(action_id)
```

旧策略不能把它自己的候选动作提交给环境。旧 session 只用于在新策略真实
轨迹的同一决策点上提供对照分布。

## 4. 接入前提

开始接入前，下游应当已经具备：

1. 一个能够完整运行 episode 的环境；
2. 可冻结并同时加载的新旧两个 agent 版本；
3. 每个目标决策点的完整有限合法动作集合；
4. 能稳定序列化的目标 agent observation；
5. 能表示所有真实环境 step 的 transition；
6. 固定的对手、seed 和先后手调度；
7. 对所有 RL/HL 实验统一使用的 epsilon。

版本身份必须指向不可变 artifact。例如：

- RL：模型权重哈希、推理配置哈希和代码版本；
- HL：规则文件哈希、规则参数哈希和代码版本；
- 混合策略：模型、规则和推理配置的组合哈希。

不要使用会被覆盖的路径（如 `latest.pt`）作为唯一版本身份。

## 5. 提供完整动作支持集

游戏 runtime 提供：

```python
from agentbench_frame.eval import ActionCandidate, ActionSupport


def support_provider(observation) -> ActionSupport:
    legal_actions = enumerate_all_legal_actions(observation)
    candidates = [
        ActionCandidate(
            action_id=encode_stable_action_id(action),
            action=action,
        )
        for action in sorted(legal_actions, key=encode_stable_action_id)
    ]
    return ActionSupport(
        actions=candidates,
        schema_version="my-game-actions-v1",
    )
```

`ActionCandidate` 的两个字段含义不同：

- `action_id`：测量层使用的稳定身份；
- `action`：真正传给 `env.step()` 的游戏 payload。

例如，环境动作可以是：

```python
{"type": "move", "source": [1, 2], "target": [1, 3]}
```

对应的稳定 ID 可以是：

```text
move:1,2->1,3
```

### 5.1 完整性

支持集必须包含当前这一次 `agent.act()` 可以合法提交的**全部原子动作**。
不能只提供：

- 模型 top-k；
- 搜索器已经展开的候选；
- 规则引擎优先级最高的几个动作；
- 本次最终选中的动作；
- 从历史日志中出现过的动作。

如果一个环境 step 接收的是完整组合动作，那么该组合就是原子动作；下游必须
枚举全部合法组合，不能未经声明地把组合拆成另一种决策过程。

若完整枚举在某个游戏中不可实现，当前严格 trajectory KL 在该决策点不可用。
正确处理是让接入失败或 episode 变为 incomplete，而不是静默近似。

### 5.2 稳定身份与顺序

要求：

- 同一动作语义在新旧策略中必须得到相同 `action_id`；
- 同一动作语义跨 episode、seed 和进程必须得到相同 `action_id`；
- 相同合法动作集合必须使用确定性顺序；
- `action_id` 不能包含进程内存地址或随机生成值；
- 修改编码方式或动作语义时升级 `schema_version`。

合法动作集合可以随状态变化。此时 `support_id` 会随有序
`legal_action_ids` 变化，这是预期行为；`schema_version` 不需要随每个状态
变化。

### 5.3 基础校验

`ActionSupport` 会直接拒绝：

- 空支持集；
- 空 `schema_version`；
- 空 `action_id`；
- 重复 `action_id`。

`support.resolve(selected_action_id)` 会拒绝不在当前支持集中的动作 ID。

## 6. 接入新策略 adapter

新策略必须实现：

```python
from agentbench_frame.eval import PolicyDecision


class ActivePolicyAdapter:
    def decide_with_distribution(self, observation, support) -> PolicyDecision:
        probabilities = self.probabilities_on_support(observation, support)
        selected_action_id = self.sample_from(probabilities)
        return PolicyDecision(
            action_id=selected_action_id,
            probabilities=probabilities,
        )

    def reset(self) -> None:
        self.session.reset()

    def observe_transition(self, transition) -> None:
        self.session.observe_transition(transition)
```

动作和分布必须来自**同一次策略决策**。不能先调用 `act()` 改变 session，再
独立查询一个已经处于不同内部状态的分布。

更严格地说，返回值必须描述实际 behavior policy：

- 若动作按 categorical 分布采样，返回被采样的同一分布；
- 若动作由 greedy argmax 确定，实际 behavior policy 是 one-hot，应返回
  one-hot；
- 不能执行 greedy 动作，却把未用于行为选择的 softmax 分布上报为 rollout
  policy；
- temperature、探索率、合法动作 mask 等影响实际行为的处理必须反映在最终
  分布中。

framework 会验证：

- `probabilities` 是 `action_id -> probability` mapping；
- key 集合与 `support.action_ids` 严格相等；
- 每个值有限且非负；
- 概率和在绝对误差 `1e-9` 内等于 1；
- `action_id` 属于当前合法支持集。

framework 能校验数据形状，但不能证明 adapter 上报的是 agent 的真实行为
分布。该真实性属于下游 adapter 的科研责任。

## 7. 接入旧策略对照 session

旧策略必须实现：

```python
class ReferencePolicyAdapter:
    def distribution_for_measurement(self, observation, support):
        return self.session.read_only_distribution(observation, support)

    def reset(self) -> None:
        self.session.reset()

    def observe_transition(self, transition) -> None:
        self.session.observe_transition(transition)
```

`distribution_for_measurement()` 的约束是：

- 在与新策略完全相同的 observation 和 `ActionSupport` 上查询；
- 不向环境提交动作；
- 不因为查询而推进 turn、RNN state、搜索树或 RNG；
- 返回 mapping 的 key 集合与支持集严格一致；
- 不使用旧策略自己的“另一套合法动作列表”替换 runtime support。

stateful 旧策略仍然可以维护内部状态，但只能通过
`reset()` 和实际 `observe_transition()` 推进。这样它始终处于“如果它观察了
新策略实际轨迹，现在会给出什么分布”的同步 session。

如果现有 agent API 的概率查询会修改内部状态，下游需要实现真正的只读查询，
或在查询前后可靠地 snapshot/restore。做不到时应让该决策测量失败，不能用
额外调用一次 `act()` 代替。

## 8. RL adapter 示例

下面示例中的模型 API 和 `masked_softmax()` 由下游实现；最终 mapping 的
定义由 framework 约束：

```python
import random

from agentbench_frame.eval import PolicyDecision


class RLActivePolicyAdapter:
    def __init__(self, model, seed):
        self.model = model
        self.rng = random.Random(seed)

    def probabilities_on_support(self, observation, support):
        logits_by_id = self.model.logits(
            observation,
            support.action_ids,
        )
        probabilities = masked_softmax(
            logits_by_id,
            support.action_ids,
        )
        return {
            action_id: float(probabilities[index])
            for index, action_id in enumerate(support.action_ids)
        }

    def decide_with_distribution(self, observation, support):
        distribution = self.probabilities_on_support(observation, support)
        selected_action_id = self.rng.choices(
            population=list(support.action_ids),
            weights=[
                distribution[action_id]
                for action_id in support.action_ids
            ],
            k=1,
        )[0]
        return PolicyDecision(
            action_id=selected_action_id,
            probabilities=distribution,
        )

    def reset(self):
        self.model.reset()

    def observe_transition(self, transition):
        self.model.observe_transition(transition)
```

对应的旧 RL adapter 只查询分布，不采样动作：

```python
class RLReferencePolicyAdapter:
    def __init__(self, model):
        self.model = model

    def distribution_for_measurement(self, observation, support):
        logits_by_id = self.model.logits_read_only(
            observation,
            support.action_ids,
        )
        probabilities = masked_softmax(
            logits_by_id,
            support.action_ids,
        )
        return {
            action_id: float(probabilities[index])
            for index, action_id in enumerate(support.action_ids)
        }

    def reset(self):
        self.model.reset()

    def observe_transition(self, transition):
        self.model.observe_transition(transition)
```

若实际 RL 评测采用 greedy policy，应返回 greedy 动作的 one-hot，而不是上面
的 soft distribution。RL 与 HL 的量纲相同来自共同的分布契约、epsilon、
对数底和 episode 聚合方式，不来自把确定性行为伪装成随机策略。

## 9. HL adapter 示例

按当前研究协议，HL 在每个决策点返回所选规则动作的 one-hot：

```python
from agentbench_frame.eval import PolicyDecision


def one_hot(selected_action_id, support):
    return {
        action_id: float(action_id == selected_action_id)
        for action_id in support.action_ids
    }


class HLActivePolicyAdapter:
    def __init__(self, rule_engine):
        self.rule_engine = rule_engine

    def decide_with_distribution(self, observation, support):
        selected_action_id = self.rule_engine.select_action_id(
            observation,
            support,
        )
        return PolicyDecision(
            action_id=selected_action_id,
            probabilities=one_hot(selected_action_id, support),
        )

    def reset(self):
        self.rule_engine.reset()

    def observe_transition(self, transition):
        self.rule_engine.observe_transition(transition)
```

旧 HL adapter 同样返回 one-hot，但查询必须只读：

```python
class HLReferencePolicyAdapter:
    def __init__(self, rule_engine):
        self.rule_engine = rule_engine

    def distribution_for_measurement(self, observation, support):
        selected_action_id = self.rule_engine.select_action_id_read_only(
            observation,
            support,
        )
        return one_hot(selected_action_id, support)

    def reset(self):
        self.rule_engine.reset()

    def observe_transition(self, transition):
        self.rule_engine.observe_transition(transition)
```

HL adapter 不需要自行给零概率加平滑。framework 会使用统一 epsilon 对新旧
分布同时进行 regularization，从而保持有限 KL 和 RL/HL 一致的单位。

当前 HL one-hot 契约只适用于：给定完整 measurement context 后，规则选择是
确定性的。固定 RNG seed 只能让随机策略可复现，不能把随机策略变成 one-hot
behavior policy。如果 HL 的随机性实际参与动作选择，下游必须：

1. 在评测模式关闭该随机性，使行为策略真正成为 one-hot；或者
2. 暴露并按真实 categorical 分布采样，此时不得再上报 one-hot。

若两者都做不到，该 HL 版本不能进行严格 trajectory KL 测量。reference 的
只读查询在任何情况下都不能消费 RNG 或改变 session。

## 10. 使用内置 Match

如果环境符合 `BaseEnv`/`Observation` 约定，可以直接使用内置 `Match`：

```python
from agentbench_frame.arena import Match
from agentbench_frame.eval import TrajectoryKLAgent, TrajectoryKLConfig
from agentbench_frame.tracking import Run


run = Run.start(
    game="my-game",
    agent="my-agent",
    run_type="eval",
    data_dir=".",
)

measured_agent = TrajectoryKLAgent(
    active_policy=new_policy_adapter,
    reference_policy=old_policy_adapter,
    support_provider=support_provider,
    config=TrajectoryKLConfig.for_policy_information_gain(
        version_before="artifact-sha256:old",
        version_after="artifact-sha256:new",
        metadata={
            "evaluation_suite": "fixed-suite-v1",
        },
    ),
    on_episode_complete=run.log_trajectory_kl_result,
)

try:
    result = Match(
        env=env,
        agent1=measured_agent,
        agent2=opponent,
        alternate_starts=True,
        seed=42,
    ).run(n_games=100)
except Exception as match_exc:
    try:
        run.finish({
            "evaluation_status": "incomplete",
        })
    except Exception as finish_exc:
        raise finish_exc from match_exc
    raise
else:
    run.finish({
        "evaluation_status": "complete",
        "benchmark_score": result.benchmark_score,
    })
```

内置 `Match` 自动完成：

- 每局开始前给测量 wrapper 注入 `game_index`、seed、`player_id` 和
  `opponent_name`；
- 调用 wrapper 的 `reset()`，进而 reset 新旧两个 session；
- 只有新策略的 `PolicyDecision.action_id` 被解析为环境动作；
- 每一个真实 step 都通知 wrapper，包括对手执行的 step；
- 给新旧 session 分别发送深拷贝后的 transition，避免互相修改；
- 在终局 transition 后完成 episode 测量；
- 在环境异常时调用 `abort_episode()`；
- 在 `MatchResult.game_results[*].trajectory_kl_measurements` 中保留测量结果。

`on_episode_complete=run.log_trajectory_kl_result` 会把 rich result 写入
`events.jsonl`。持久化 callback 如果失败会向上抛出，runner 不会静默声称
测量成功。示例在成功和失败分支都恰好调用一次 `Run.finish()`，以 flush
buffered JSONL 并写出最终运行状态；如果 match 与 finalization 同时失败，
finalization 错误会保留 match 错误作为 cause。

内置 `Match` 还要求环境 observation 至少支持：

- `to_dict()`；
- `player_id`；
- 终局时可从 `observation.state["winner"]` 取得胜者或平局。

如果现有 Saiblo runtime 不符合这些约定，使用下一节的自定义 runner 接入。

## 11. 接入自定义 Saiblo runner

自定义 runner 必须显式维护完整 episode 生命周期。下面代码展示必要调用；
游戏特有的 observation、对手和终局处理由下游替换：

```python
import copy


def run_measured_episode(
    *,
    env,
    measured_agent,
    opponent,
    seed,
    metadata,
):
    measured_agent.set_measurement_episode_metadata(metadata)

    try:
        measured_agent.reset()
        opponent_reset = getattr(opponent, "reset", None)
        if callable(opponent_reset):
            opponent_reset()

        observation = env.reset(seed=seed)
        done = False
        env_step = 0

        while not done:
            previous = observation
            actor = actor_player_id(previous)

            if is_target_turn(previous):
                action = measured_agent.act(
                    observation_to_dict(previous)
                )
            else:
                action = opponent.act(
                    observation_to_dict(previous)
                )

            observation, reward, env_done, info = env.step(action)
            info = info or {}
            env_step += 1
            truncated = bool(info.get("truncated", False))
            terminated = bool(env_done and not truncated)
            done = terminated or truncated

            transition = {
                "observation": observation_to_dict(previous),
                "actor_player_id": actor,
                "action": action,
                "next_observation": observation_to_dict(observation),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "done": done,
                "info": info,
                "env_step": env_step,
            }
            measured_agent.observe_transition(transition)

            opponent_observe = getattr(
                opponent,
                "observe_transition",
                None,
            )
            if callable(opponent_observe):
                opponent_observe(copy.deepcopy(transition))
    except Exception as exc:
        try:
            measured_agent.abort_episode(
                f"{type(exc).__name__}: {exc}"
            )
        except Exception as abort_exc:
            raise abort_exc from exc
        raise

    return measured_agent.latest_trajectory_kl_result
```

必须遵守：

1. 每个 episode 恰好调用一次 `reset()`；
2. 只在目标 agent 回合调用 `measured_agent.act()`；
3. 每个真实环境 step 后都调用一次 `observe_transition()`，包括对手动作；
4. 最后一条 transition 必须设置 `terminated=True` 或 `truncated=True`；
5. runner 异常必须调用 `abort_episode()` 并继续向上抛出原异常；
6. 不要在 terminal 后继续调用 `act()` 或 `observe_transition()`；
7. 不要把未执行的旧策略候选动作伪造成 transition。

`measured_agent.reset()` 和 `env.reset()` 必须位于同一个异常保护区内；只要
measurement episode 已开始，后续 reset、环境初始化或运行错误都必须触发
`abort_episode()`。对手自身的 `reset()` 和 transition 生命周期也由自定义
runner 维护，不能假定 measurement wrapper 会替对手维护状态。

### 11.1 Transition 的最小语义

| 字段 | 含义 |
|---|---|
| `observation` | 动作执行前、对应真实 actor 的可见状态 |
| `actor_player_id` | 本 step 的真实行动方 |
| `action` | 环境实际执行的 payload |
| `next_observation` | 动作执行后的状态 |
| `reward` | runtime 返回的实际 reward |
| `terminated` | 规则终局 |
| `truncated` | 时间、资源或外部约束截断 |
| `done` | runtime 的兼容终局标记 |
| `info` | 原始环境补充信息 |
| `env_step` | 本 episode 从 1 开始的实际环境 step |

stateful reference session 必须看到对手动作，否则它的内部状态会与真实轨迹
失去同步。

### 11.2 Observation 稳定性

framework 用 observation 生成 `context_ref`。建议传入只含以下类型的稳定
结构：

- `dict`、`list`、`tuple`；
- 字符串、整数、布尔值、有限浮点数、`None`；
- 可稳定转换的 dataclass、Enum 或 `to_dict()` 对象。

不要包含非有限浮点数、进程内存地址、未排序随机容器或每次序列化都会变化的
调试字段。`context_ref` 是内容哈希，不等于保存完整 observation；需要完整
状态审计时，下游还应保留环境自己的 trajectory/event artifact。

## 12. 持久化与输出字段

推荐的唯一 rich persistence 接口是：

```python
on_episode_complete=run.log_trajectory_kl_result
```

它写入兼容旧事件类型的 `policy_kl_trace` 事件，同时增量保存 rich 字段。
事实源位于：

```text
{data_dir}/runs/{game}/{agent}/{run_id}/events.jsonl
```

上一节示例使用 `data_dir="."`，所以实际路径为
`./runs/{game}/{agent}/{run_id}/events.jsonl`。

### 12.1 每个决策点的一手字段

| 字段 | 含义 |
|---|---|
| `decision_step` | 目标 agent 在本 episode 的第几次决策 |
| `context_ref` | observation 的稳定内容哈希 |
| `action_schema_version` | 游戏动作 Schema 版本 |
| `support_id` | 当前有序动作支持集的内容 ID |
| `legal_action_ids` | 完整、有序合法动作 ID |
| `selected_action_id` | 新策略实际选择的动作 ID |
| `new_distribution` | 新 adapter 返回的原始 mapping |
| `old_distribution` | 旧 adapter 返回的原始 mapping；query 抛异常或返回非 mapping 时缺失，mapping 校验失败时仍保留 |
| `new_probabilities` | 按支持集顺序对齐并校验后的新概率 |
| `old_probabilities` | 按支持集顺序对齐并校验后的旧概率 |
| `local_policy_kl` | epsilon-regularized local KL；失败时为缺失 |
| `errors` | 本决策的错误列表 |

### 12.2 Episode 字段

| 字段 | 含义 |
|---|---|
| `measurement_status` | `complete` 或 `incomplete` |
| `version_before` | 旧版本不可变身份 |
| `version_after` | 新版本不可变身份 |
| `epsilon` | 本次固定 regularization 参数 |
| `trace` | 与 decisions 一一对齐的 local KL 序列 |
| `decision_steps` | 目标 agent 决策次数 |
| `information_gain` / `mean_local_policy_kl` | 主 episode IG（trace 算术平均），`nats / decision` |
| `local_policy_kl_sum` / `trajectory_kl_episode` | 可选 trace 总和，`nats / episode` |
| `direction` | 固定为 `new||old` |
| `log_base` | 固定为自然对数 `e` |
| `rollout_source` | rich 在线测量为 `new_policy` |
| `estimand` | 严格测量对象名称 |
| `metadata` | seed、对手、先后手和实验协议等 |
| `errors` | episode 级错误列表 |

报告会从有序 decision records 的 old/new distributions 重新计算每个 local
KL、trace、总和与均值，不盲信上传的 local 或预计算 summary。畸形、不对齐、
非有限或 incomplete 事件在图中保留为缺口。

旧的 `Run.log_policy_kl_trace()` 只用于兼容已经计算好的 legacy trace，无法
证明其 rollout 来源；它保留原 trace，但正式 `information_gain` 必须为 null。
新接入应使用 `log_trajectory_kl_result()`。

## 13. 固定实验协议

trajectory KL 具有统一单位，但其数值仍受游戏、动作支持规模、episode 长度
和状态访问影响。`nats / episode` 相同不意味着不同游戏之间可以无条件解释
为相同“学习量”。

在同一游戏内比较 RL 与 HL 或不同迭代轮次时，应固定：

| 项目 | 要求 |
|---|---|
| 对手集合 | 使用版本固定的评测对手或固定评测集 |
| 对手顺序 | 每轮迭代使用同一调度，或保存可重建调度的 case ID |
| seed | 使用同一 seed 列表和 seed 到 episode 的映射 |
| 先后手 | 使用同一交替或配对方案 |
| episode 数 | 每个比较单元使用相同数量 |
| epsilon | RL、HL 和所有迭代使用同一固定值 |
| 动作 Schema | 同一比较中语义一致；变化时明确升级版本 |
| agent artifact | 使用不可变身份，不使用可覆盖别名 |
| 推理设置 | temperature、greedy/stochastic、搜索预算等全部冻结 |

如果评测集对手按实力递增，可以保留每个 episode 的原始折线，但必须同时保存
`opponent_name` 或 benchmark case ID。此时横向变化同时包含：

- 新旧策略在当前访问状态上的差异；
- 对手改变引起的状态访问和 episode 长度变化。

不能把整条曲线的变化全部归因于 learning。更稳妥的做法是每轮迭代重复同一
固定对手序列，并按对手或 case 分层比较。

24_miracle 正式主 epsilon 由无 epsilon 参数的
`TrajectoryKLConfig.for_policy_information_gain()` 固定，必须满足：

```text
epsilon == 0.01
```

通用研究工具仍可用 `TrajectoryKLConfig(..., epsilon=...)` 选择合法的其他
epsilon，但其 `measurement_profile=generic_trajectory_kl`，不能填充正式 IG。

不要让 adapter 自行平滑后又让 framework 二次平滑。

## 14. 错误语义与排障

| 现象 | framework 行为 | 下游排查 |
|---|---|---|
| 支持集为空 | `ActionSupport` 抛出 `ValueError`；episode 被中止 | 检查终局判断或合法动作枚举 |
| `action_id` 重复 | `ActionSupport` 抛出 `ValueError` | 修复稳定编码，不能靠顺序区分 |
| 新动作 ID 不合法 | `support.resolve()` 抛出 `ValueError` | 确保选择来自同一 support |
| 分布缺少或多出 key | 记录该决策错误，episode incomplete | mapping 必须与 `support.action_ids` 严格相等 |
| 概率为负、NaN 或 infinity | 记录该决策错误，episode incomplete | 检查 mask、softmax 和数值转换 |
| 概率和不为 1 | 记录该决策错误，episode incomplete | 在 adapter 中按实际行为策略归一化 |
| 旧查询抛异常或返回非 mapping | 新动作仍可执行；旧原始分布和 local KL 缺失，episode incomplete | 检查旧 artifact、只读 API 和 session 同步 |
| 旧 mapping 校验失败 | 原始 `old_distribution` 保留；对齐概率和 local KL 缺失，episode incomplete | 检查 key、概率范围与归一化 |
| 没有目标决策点 | episode incomplete，主指标缺失 | 检查角色映射、立即终局或 runner 回合判断 |
| terminal 前再次 reset | 旧 episode 先以 incomplete 结束 | 修复 episode 生命周期 |
| transition observer 失败 | 错误写入 episode，主指标缺失 | 检查新旧 session 的 transition 解析 |
| 环境或 runner 异常 | `abort_episode()` 保存已得证据并继续抛出异常 | 修复环境；不要吞掉异常后继续计分 |
| persistence callback 失败 | 写入错误向上抛出 | 检查磁盘、权限和事件序列化 |
| 报告出现缺口 | 不插值，不把缺失当作 0 | 查看对应事件的 `errors` 和 `measurement_status` |

一个决策出现错误后，framework 仍可能让新策略动作继续执行，以保留真实
rollout；但该 episode 不再产生主标量。下游不能从已有 local KL 手工求部分
和并标记为 complete。

一手证据能保留到哪个阶段取决于失败发生的位置：

- 支持集构造失败、active 返回类型错误或所选 action ID 不合法时，当前决策
  record 尚未创建；framework 保留之前已经完成的 decisions 和 episode 级
  abort error；
- 新分布校验失败、旧查询失败或旧分布校验失败时，当前决策 record 已创建，
  会保留当时取得的原始 mapping、可用字段和 errors。

因此“incomplete 会保留一手数据”不表示每一种前置失败都能凭空产生当前决策
的 raw record。

## 15. 下游验收清单

### 15.1 动作域

- [ ] 每个目标决策点都能枚举完整、有限、非空的合法原子动作。
- [ ] 相同动作语义在新旧 adapter 中使用同一 `action_id`。
- [ ] 相同合法集合的 action ID 顺序确定。
- [ ] 动作编码变化会升级 `schema_version`。
- [ ] top-k、搜索候选或历史动作不会冒充完整支持集。

### 15.2 策略 adapter

- [ ] 新策略在一次调用中同时返回 action ID 和 distribution。
- [ ] 新分布描述实际 behavior policy，而不是另一个未执行的 soft policy。
- [ ] 旧分布查询不会提交动作或推进隐藏状态。
- [ ] RL 和 HL 都返回完整的 `action_id -> probability` mapping。
- [ ] 确定性 HL 按协议返回 one-hot；随机 HL 不会把 realized action 谎报为 one-hot。
- [ ] epsilon 只由 framework 添加。
- [ ] 新旧 mapping 的 key 与当前支持集严格相等。

### 15.3 Runner

- [ ] 每个 episode 只 reset 一次。
- [ ] 只有目标 agent 回合调用测量 wrapper 的 `act()`。
- [ ] 每个实际环境 step，包括对手动作，都同步给 wrapper。
- [ ] terminal transition 正确区分 `terminated` 和 `truncated`。
- [ ] runner 异常会调用 `abort_episode()`，且异常继续向上抛出。
- [ ] agent reset 或 env reset 失败也会结束已打开的 measurement episode。
- [ ] `Run.finish()` 在成功和失败路径都恰好执行一次。
- [ ] seed、对手、先后手和 benchmark case ID 写入 metadata。

### 15.4 数据与数值

- [ ] 一个正常 episode 的 decision 数与 trace 长度严格相等。
- [ ] 每个正常 local KL 有限且非负。
- [ ] `information_gain` 等于完整 trace 的算术平均。
- [ ] `local_policy_kl_sum`（及兼容 sum 字段）等于完整 trace 之和。
- [ ] 单位明确为主 `nats / decision`、可选 sum `nats / episode`。
- [ ] 分布/查询失败会保留当前决策已取得的 raw evidence，并使 episode incomplete。
- [ ] support、active 类型或非法动作的前置失败只要求保留先前 decisions 和 abort error。
- [ ] incomplete episode 的总和与均值为缺失，不是 0。
- [ ] `version_before`、`version_after` 和 epsilon 可复现。
- [ ] 本地报告和 CI 都从 trace 派生标量并保留缺口。

### 15.5 最小验收场景

至少运行以下测试场景：

1. 两动作、单决策 episode，人工核对一个 local KL；
2. 多决策完整 episode，核对 trace、总和和均值；
3. 新策略 mapping 缺少一个 action ID，确认 episode incomplete；
4. 旧策略查询抛出异常，确认新动作仍执行且主标量缺失；
5. HL 新旧选择不同动作，确认 one-hot 经 epsilon 后得到有限 KL；
6. 环境中途抛出异常，确认已有 decisions 被保留；
7. persistence callback 失败，确认任务失败而不是生成假成功；
8. agent reset 或 env reset 抛出异常，确认已打开 episode 被中止；
9. match 失败后调用 `Run.finish()`，确认 buffered incomplete 事件被 flush；
10. 用同一 seed、对手和版本重跑，确认协议与 metadata 一致。

## 16. 不属于当前方案的做法

当前接入不包含：

- replay-based KL；
- 对全部历史 rollout 决策点重新查询新旧版本；
- 完整游戏树或全状态空间 KL；
- 用历史动作频率估计策略分布；
- 用 top-k、beam、搜索前沿或候选动作子集替代完整动作支持集；
- 把 occupancy shift 与 policy KL 相加称为单一信息增益；
- 用 episode 平均值替代逐 episode 主曲线；
- 自动设计 benchmark 对手或测试集。

如果未来重新启动 replay-based KL，需要另外定义跨游戏 replay adapter、
stateful agent 恢复、只读 artifact 查询和失败完整性语义；不能复用当前在线
接口后改变已有字段含义。
