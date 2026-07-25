# Trajectory KL Downstream Integration Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a standalone Chinese guide that lets a Saiblo runtime integrate strict online trajectory KL without reading implementation internals.

**Architecture:** Keep mathematical motivation in the research document and put the operational contract in `docs/integration/trajectory-kl.md`. The guide follows the downstream data flow from action-support enumeration through RL/HL adapters, runner wiring, persistence, experiment control, failure semantics, and acceptance checks; it changes no runtime behavior.

**Tech Stack:** Markdown, Python public APIs from `agentbench_frame.eval`, `agentbench_frame.arena.Match`, and `agentbench_frame.tracking.Run`.

## Global Constraints

- Work only in `/home/wkj/projects/AgentBenchFrame/.worktrees/framework` on `worktree/framework`.
- Do not merge or modify local `main`.
- Do not change runtime code, event Schema, CI logic, or benchmark content.
- Use current public signatures from `measurement.py`, `trajectory_kl.py`, `match.py`, and `run.py`.
- HL returns one-hot distributions; RL returns legal-action-masked normalized distributions.
- Never present a candidate subset, historical action frequency, or replay-based KL as the current strict measurement.
- The primary estimand is `epsilon_regularized_local_kl_sum_under_new_policy_occupancy` in `nats / episode`; `mean_local_policy_kl` in `nats / decision` is auxiliary.
- Any invalid target decision makes the episode `incomplete`; primary and auxiliary scalars remain missing.
- Push only `worktree/framework` to its configured `origin/worktree/framework` upstream.

---

### Task 1: Write the standalone downstream guide

**Files:**
- Create: `docs/integration/trajectory-kl.md`
- Reference: `src/agentbench_frame/eval/measurement.py`
- Reference: `src/agentbench_frame/eval/trajectory_kl.py`
- Reference: `src/agentbench_frame/arena/match.py`
- Reference: `src/agentbench_frame/tracking/run.py`

**Interfaces:**
- Consumes: `ActionCandidate(action_id: str, action: Any)`.
- Consumes: `ActionSupport(actions: Sequence[ActionCandidate], schema_version: str)`.
- Consumes: `PolicyDecision(action_id: str, probabilities: Mapping[str, float])`.
- Consumes: `TrajectoryKLConfig(version_before: str, version_after: str, epsilon: float, metadata: Mapping[str, Any] = {})`.
- Consumes: `TrajectoryKLAgent(active_policy, reference_policy, support_provider, config, on_episode_complete=None)`.
- Produces: a self-contained integration guide with executable adapter skeletons and a downstream acceptance checklist.

- [ ] **Step 1: Verify the standalone guide does not already exist**

Run:

```bash
test ! -e docs/integration/trajectory-kl.md
```

Expected: exit code 0.

- [ ] **Step 2: Create the guide with the approved execution-order structure**

Create `docs/integration/trajectory-kl.md` with these top-level sections:

```markdown
# Trajectory KL 下游接入指南

## 1. 接入完成后会得到什么
## 2. Framework 与下游的责任边界
## 3. 测量数据流与对手的作用
## 4. 接入前提
## 5. 提供完整动作支持集
## 6. 接入新策略 adapter
## 7. 接入旧策略对照 session
## 8. RL adapter 示例
## 9. HL adapter 示例
## 10. 使用内置 Match
## 11. 接入自定义 Saiblo runner
## 12. 持久化与输出字段
## 13. 固定实验协议
## 14. 错误语义与排障
## 15. 下游验收清单
## 16. 不属于当前方案的做法
```

The responsibility table must state:

| Responsibility | Owner |
|---|---|
| Enumerate the complete finite legal action support | downstream game runtime |
| Give every action a stable semantic `action_id` | downstream game runtime |
| Return action and distribution coherently | active/new policy adapter |
| Return a read-only distribution on the same support | reference/old policy adapter |
| Run the actual new-policy action | framework/runner |
| Synchronize both sessions with actual transitions | framework `Match`, or custom runner |
| Validate distributions, regularize, compute local KL and episode scalars | framework |
| Persist first-hand decision evidence | framework through `Run` callback |
| Fix opponent schedule, seeds, sides, versions, and epsilon | downstream experiment protocol |
| Derive charts and preserve incomplete gaps | framework/CI |

Explain the trajectory dependence exactly:

```math
\tau \sim P(\tau \mid \pi_{\mathrm{new}}, \mu, \rho_0, P_{\mathrm{env}})
```

and:

```math
\operatorname{TrajectoryKL}(\tau)
=
\sum_{t\in D_{\mathrm{target}}(\tau)}
D_{\mathrm{KL}}\!\left(
\widetilde{\pi}_{\mathrm{new}}(\cdot\mid s_t)
\Vert
\widetilde{\pi}_{\mathrm{old}}(\cdot\mid s_t)
\right).
```

State that the opponent does not enter local KL after a trajectory is fixed,
but changes visited states, target decision count, and episode length.

- [ ] **Step 3: Add a complete action-support example**

The guide must include a game-owned adapter with deterministic ordering:

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

Explain that `action` is the environment payload while `action_id` is the
measurement identity. Dynamic legal sets are allowed, but repeated semantics
must keep the same ID and deterministic order. If all legal atomic actions
cannot be enumerated, strict trajectory KL is unavailable for that decision;
the adapter must not substitute top-k or sampled candidates.

- [ ] **Step 4: Add coherent new-policy and read-only old-policy contracts**

Include these exact skeletons:

```python
from agentbench_frame.eval import PolicyDecision


class ActivePolicyAdapter:
    def decide_with_distribution(self, observation, support) -> PolicyDecision:
        probabilities = self.probabilities_on_support(observation, support)
        selected_action_id = self.sample_or_select(probabilities)
        return PolicyDecision(
            action_id=selected_action_id,
            probabilities=probabilities,
        )

    def reset(self) -> None:
        self.session.reset()

    def observe_transition(self, transition) -> None:
        self.session.observe_transition(transition)


class ReferencePolicyAdapter:
    def distribution_for_measurement(self, observation, support):
        return self.session.read_only_distribution(observation, support)

    def reset(self) -> None:
        self.session.reset()

    def observe_transition(self, transition) -> None:
        self.session.observe_transition(transition)
```

State that `distribution_for_measurement()` cannot submit an action or advance
hidden state as a side effect. Stateful reference sessions advance only from
the actual transition stream. Every mapping must have exactly the support IDs,
finite non-negative values, and sum to 1 within absolute tolerance `1e-9`.

- [ ] **Step 5: Add separate RL and HL examples**

For RL, show legal-action masking, normalization, and selection from the same
probability map:

```python
def probabilities_on_support(self, observation, support):
    logits_by_id = self.model.logits(observation, support.action_ids)
    probabilities = masked_softmax(logits_by_id, support.action_ids)
    return {
        action_id: float(probabilities[index])
        for index, action_id in enumerate(support.action_ids)
    }
```

For deterministic HL, show the agreed one-hot representation:

```python
def one_hot(selected_action_id, support):
    return {
        action_id: float(action_id == selected_action_id)
        for action_id in support.action_ids
    }


def decide_with_distribution(self, observation, support):
    selected_action_id = self.rule_engine.select(observation, support)
    return PolicyDecision(
        action_id=selected_action_id,
        probabilities=one_hot(selected_action_id, support),
    )
```

The old HL adapter must use the same one-hot helper from its selected rule
action. Explain that epsilon regularization is performed by the framework, not
by the HL adapter.

- [ ] **Step 6: Add built-in `Match` wiring and persistence**

Use this complete wiring:

```python
from agentbench_frame.arena import Match
from agentbench_frame.eval import TrajectoryKLAgent, TrajectoryKLConfig
from agentbench_frame.tracking.run import Run


run = Run.start(game="my-game", agent="my-agent", data_dir="runs")
measured_agent = TrajectoryKLAgent(
    active_policy=new_policy_adapter,
    reference_policy=old_policy_adapter,
    support_provider=support_provider,
    config=TrajectoryKLConfig(
        version_before="artifact-sha256:old",
        version_after="artifact-sha256:new",
        epsilon=0.01,
        metadata={"evaluation_suite": "fixed-suite-v1"},
    ),
    on_episode_complete=run.log_trajectory_kl_result,
)

result = Match(
    env=env,
    agent1=measured_agent,
    agent2=opponent,
    alternate_starts=True,
    seed=42,
).run(n_games=100)
run.finish()
```

Explain that `Match` adds game index, seed, player side, and opponent name;
only the active policy chooses environment actions; callback persistence
failures propagate rather than being silently ignored.

- [ ] **Step 7: Add a complete custom-runner lifecycle**

Show pseudocode that performs these operations in order:

```python
measured_agent.set_measurement_episode_metadata(metadata)
measured_agent.reset()
observation = env.reset(seed=seed)

try:
    while not done:
        if is_target_turn(observation):
            action = measured_agent.act(observation_to_dict(observation))
        else:
            action = opponent.act(observation_to_dict(observation))

        previous = observation
        observation, reward, done, info = env.step(action)
        measured_agent.observe_transition({
            "observation": observation_to_dict(previous),
            "actor_player_id": actor_player_id(previous),
            "action": action,
            "next_observation": observation_to_dict(observation),
            "reward": reward,
            "terminated": bool(done),
            "truncated": bool(info.get("truncated", False)),
            "done": bool(done),
            "info": info,
            "env_step": env_step,
        })
except Exception as exc:
    measured_agent.abort_episode(f"{type(exc).__name__}: {exc}")
    raise
```

State that every real environment step, including opponent actions, must be
observed so stateful sessions remain synchronized. Observations must serialize
deterministically and cannot contain non-finite floats.

- [ ] **Step 8: Document evidence, failure semantics, and acceptance**

List the first-hand decision fields:

```text
legal_action_ids
selected_action_id
new_distribution
old_distribution
new_probabilities
old_probabilities
support_id
context_ref
local_policy_kl
errors
```

List episode-level fields and units:

```text
measurement_status
trajectory_kl_episode       # nats / episode, primary
mean_local_policy_kl        # nats / decision, auxiliary
trace
decision_steps
epsilon
direction                   # new||old
rollout_source              # new_policy
estimand
metadata
errors
```

The troubleshooting table must cover invalid/empty support, duplicate IDs,
missing/extra distribution keys, negative/non-finite probabilities, sum not
equal to one, reference query failure, no target decisions, reset before
terminal, environment failure, and persistence failure. The acceptance
checklist must verify:

1. The selected environment action resolves from the returned action ID.
2. RL and HL both expose exactly aligned mappings.
3. New and old sessions receive every actual transition without shared
   mutable objects.
4. A valid episode produces one finite non-negative trace value per target
   decision and a sum in `nats / episode`.
5. Any invalid decision preserves raw evidence but produces an incomplete gap.
6. Events contain immutable version IDs, epsilon, opponent, seed, and side.
7. Repeating the fixed protocol preserves the intended opponent/seed schedule.
8. Local and CI reports derive scalars from traces rather than trusting stale
   precomputed fields.

### Task 2: Link, verify, commit, and publish the guide

**Files:**
- Modify: `docs/research/information-gain-design.md`
- Verify: `docs/integration/trajectory-kl.md`
- Verify: `docs/superpowers/specs/2026-07-25-trajectory-kl-downstream-integration-doc-design.md`

**Interfaces:**
- Consumes: the standalone guide produced by Task 1.
- Produces: a discoverable, verified documentation change published on `origin/worktree/framework`.

- [ ] **Step 1: Add a discoverability link from the research document**

Immediately after the opening paragraph of section `10.13.2 在线测量接口`,
add:

```markdown
具体游戏、RL/HL agent 和自定义 runner 的完整接入步骤见
[Trajectory KL 下游接入指南](../integration/trajectory-kl.md)。本节只保留
科研定义和公共接口摘要。
```

- [ ] **Step 2: Verify structure, forbidden claims, and exact public names**

Run:

```bash
rg -n "^## ([1-9]|1[0-6])\\." docs/integration/trajectory-kl.md
rg -n "ActionCandidate|ActionSupport|PolicyDecision|TrajectoryKLConfig|TrajectoryKLAgent|log_trajectory_kl_result" docs/integration/trajectory-kl.md
rg -n "top-k|候选动作子集|历史动作频率|replay-based" docs/integration/trajectory-kl.md
rg -n "Trajectory KL 下游接入指南" docs/research/information-gain-design.md
```

Expected:

- The first command reports all 16 numbered sections.
- The second command reports every public interface.
- The third command reports only explicit prohibitions/non-goals, never a
  recommended measurement path.
- The fourth command reports the new relative link.

- [ ] **Step 3: Check Markdown and repository cleanliness**

Run:

```bash
git diff --check
rg -n "T[B]D|T[O]DO|待定|稍后补充|implement later" docs/integration/trajectory-kl.md docs/research/information-gain-design.md
```

Expected: `git diff --check` exits 0; the placeholder scan has no matches.

- [ ] **Step 4: Run the full framework test suite**

Run:

```bash
uv run --extra report --with pytest pytest -q
```

Expected: all framework tests pass.

- [ ] **Step 5: Review the final diff against the approved design**

Run:

```bash
git diff -- docs/integration/trajectory-kl.md docs/research/information-gain-design.md
git status --short
```

Expected: only the standalone guide and research cross-link are uncommitted;
the previously committed spec and plan remain unchanged.

- [ ] **Step 6: Commit the guide**

Run:

```bash
git add docs/integration/trajectory-kl.md docs/research/information-gain-design.md
git commit -m "docs: add trajectory KL downstream integration guide"
```

Expected: one documentation commit on `worktree/framework`.

- [ ] **Step 7: Push the independent framework branch**

Run:

```bash
git push origin worktree/framework
```

Expected: `origin/worktree/framework` advances to the guide commit without
checking out, merging, or modifying local `main`.
