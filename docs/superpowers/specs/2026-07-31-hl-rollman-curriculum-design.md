# 29 Rollman 逐个击破课程模式设计

## 1. 目标

在 AgentBenchFramework 的 HL 闭环中增加可复现的动态课程模式。课程从指定的已认证代码快照建立独立运行，逐一选择尚未通过且榜单排名最低的人类对手作为学习目标。每次模型 rollout 只接收当前目标的合法比赛回放并修改可解释策略；候选通过当前目标后必须接受完整人类池认证，且不得丢失已经锁定的通过结果。

29 Rollman 实验满足以下终止条件：

- 来源代码快照：`run-20260731-gpt55-sota/v000001`
- 首个学习目标：`rank15`
- 后续学习目标：根据认证结果动态选择，预期为 `rank14`
- 成功标准：16 个冻结人类对手全部达到规定通过率
- 模型候选数初值：`k=1`
- 成功前不设置总 act 上限

## 2. 非目标

- 不读取、搜索或推断人类对手源码。
- 不把人工战术标签引入原子决策空间。
- 不以无依据的参数枚举或 grid search 代替回放诊断。
- 不改变固定 rank1 模式的默认语义。
- 不部署网站或服务器任务系统。
- 不自动修改 `k`；`k` 作为消融参数由运行配置控制。

## 3. 配置接口

HL 配置增加两个严格校验的区段：

```yaml
origin:
  mode: imported_version
  source_run: ".agentbench/29_rollman/runs/run-20260731-gpt55-sota"
  source_version: "v000001"
  reset_session: true
  reset_experience: true

curriculum:
  mode: weakest_failed
  target_order: lowest_rank_first
  preserve_passed_opponents: true
  required_human_opponents: 16
  stagnation_patience: 4
```

`origin.mode` 支持：

- `model_bootstrap`：由编码模型从规则生成 origin，保持固定 rank1 实验的默认行为。
- `imported_version`：校验来源 manifest 和内容对象，将来源代码复制到候选 workspace，并在新运行中快照为 `v000000`。

`curriculum.mode` 支持：

- `fixed`：使用 `evaluation.learning_opponent`，保持默认实验行为。
- `weakest_failed`：从最近一次完整认证中选择未通过且排名数值最大的对手。

配置中不得保存 API key。运行快照记录完整的无密钥配置、来源内容哈希与上下文哈希。
`source_run` 的相对路径以配置文件所属仓库根目录解析，冻结快照保存解析后的绝对路径和来源 run id。

## 4. 来源快照与独立运行

导入过程执行以下校验：

1. 来源运行目录、版本 manifest 和内容对象必须存在。
2. 内容对象的文件列表与 SHA-256 必须和 manifest 一致。
3. 来源 workspace 中的 `.agentbench`、缓存和字节码不进入代码快照。
4. 新候选 workspace 精确恢复来源代码。
5. 新运行的首个版本为 `v000000`，`edit_type` 为 `imported_origin`。
6. `origin_imported` 事件记录来源 run id、来源版本、来源内容哈希和新版本。
7. 首次模型调用前，对导入代码重新执行当前目标 gate 和 16 人完整认证；课程状态只使用新运行产生的比赛结果，不直接复用来源事件中的成绩。

`reset_session: true` 创建新的 provider 会话，避免 rank1 迭代上下文影响课程学习。`reset_experience: true` 创建新的 Experience Skill；来源代码本身保留可解释结构，课程经验仅由 rank15、rank14 的合法回放生成。

## 5. 课程状态

课程状态由事件日志重建，不依赖可变的旁路状态文件。核心状态包括：

- `active_target`：当前学习对手。
- `locked_opponents`：阶段基线在认证种子上已通过的对手。
- `stage_origin_version_id`：当前阶段的安全回滚点。
- `stage_best_version_id`：当前目标 gate 成绩最好的版本。
- `stage_best_score`：当前目标的固定种子得分。
- `stagnation_count`：连续未提高阶段最佳成绩的 act 数。

新事件类型：

- `origin_imported`
- `curriculum_started`
- `curriculum_target_selected`
- `curriculum_stage_promoted`
- `curriculum_candidate_rejected`
- `curriculum_stagnated`

每个事件均包含 run id、事件 id、时间、版本 id 和足以重建状态的结构化字段。resume 必须从事件日志恢复同一目标、锁定集合、阶段冠军和停滞计数。

## 6. 目标选择

对每个对手，将完整认证中的胜记为 1、平记为 0.5、负记为 0。平均值达到 `required_win_rate` 即为通过。

`weakest_failed` 选择规则：

1. 只使用状态为 `complete` 的认证。
2. 计算每个人类对手的固定认证种子通过率。
3. 收集通过率低于阈值的对手。
4. 选择 `opponent_rank` 最大者。
5. 没有失败对手时，运行以 16/16 成功结束。

来源版本的完整认证结果为课程初始依据。`v000001` 通过 14 个对手，未通过 `rank14`、`rank15`，因此首个目标为 `rank15`。

## 7. 单阶段数据流

一个课程阶段按以下顺序运行：

1. Framework 恢复阶段父版本到候选 workspace。
2. Framework 使用当前目标的固定 gate 种子运行比赛。
3. Replay Skill 将比赛 JSON 翻译为可引用的状态、动作、事件和失败节点。
4. 增量 prompt 包含当前目标、父版本、目标 gate 测量、目标回放路径和 Experience Skill 路径。
5. Codex provider 在可恢复会话中生成一个或 `k` 个候选。
6. 每个候选独立进行静态检查、smoke test 和当前目标固定种子 gate。
7. `k>1` 时，按当前目标 gate 得分选择候选；同分按确定性分数差、代码规模与 branch index 依次打破平局。
8. gate 未通过的候选保留为历史版本，其回放用于下一次诊断。
9. gate 通过的候选执行 16 人完整认证。
10. 认证结果满足阶段晋级规则时锁定候选并选择下一个目标；否则回滚到阶段安全版本。

规则、决策空间和 Replay Skill 由内容哈希固定并在会话中通过文件引用复用。每个 act 只发送增量测量和新增回放索引，避免重复发送长篇静态上下文。

## 8. Prompt 约束

课程 prompt 必须明确：

- 当前只优化 `active_target`。
- 至少引用一个具体 level、round 或事件作为因果诊断证据。
- 允许可解释的 `if/else` 情形处理、状态机、路径规划、图搜索、有限记忆和结构化启发式。
- 情形分支必须对应可观测状态和合法原子动作。
- 新规则应整合进已有策略层次，替代失效逻辑，避免无限追加特殊分支。
- 不得针对 seed、固定回放坐标或人类身份硬编码。
- 不得在缺乏回放证据时枚举阈值或执行 grid search。
- 不得访问认证对手源码。
- Experience 更新必须区分稳定知识、失败假设、回放证据和待验证问题。

候选只允许读取 Framework 提供的 context、candidate workspace、Experience Skill 和目标比赛产物。API key、其他对手源码和来源运行中的 provider 会话不可见。

## 9. 晋级、选择与回滚

### 9.1 阶段内选择

阶段内成绩仅与同一 `active_target` 比较。目标切换时建立新的阶段基准，避免把 rank15 分数和 rank14 分数当作同一标尺。

候选提高当前目标 gate 得分时成为 `stage_best_version_id`。持续退化达到 rollback patience 时，下一次 act 从阶段最佳版本继续。

### 9.2 全池防退化

候选必须同时满足：

- 当前目标在完整认证中达到通过阈值。
- `locked_opponents` 中每个对手仍达到通过阈值。

满足条件时：

- 候选成为新的 `stage_origin_version_id`。
- 当前目标加入 `locked_opponents`。
- Framework 根据认证结果选择下一个最低排名失败者。
- provider 会话可继续复用，静态上下文不重复发送；prompt 显式声明阶段切换。

不满足条件时：

- 记录 `curriculum_candidate_rejected`，包含丢失的锁定对手和对应结果。
- Experience 仅保留经回放支持的失败诊断，不把候选代码设为父版本。
- workspace 回滚到 `stage_origin_version_id`。

## 10. 停滞监督

运行没有成功前的总 act 上限，但不得无限消耗模型 token。

连续 `stagnation_patience` 个 act 未提高当前目标的 `stage_best_score` 时：

1. 写入 `curriculum_stagnated`。
2. 保存 checkpoint、会话 id、目标回放、Experience 和所有版本。
3. 退出模型调用循环并返回 `stagnated` 状态。
4. 保留 resume 能力。

停滞不是科研终止条件。监督者检查回放、代码复杂度、失败假设和 token 使用后，可通过参数继续 `k=1`，或建立仅改变 `candidates_per_act` 的消融运行。Provider 失败、评测不完整和本地基础设施错误同样立即暂停，禁止付费重试死循环。

## 11. 指标与报告

每个候选继续记录：

- 原子动作局部策略 KL
- episode 级 KL
- 状态占用差
- 当前目标固定 gate 胜率
- 完整认证总胜率
- 每个对手的认证通过率
- Rollman Elo
- token 使用
- 代码内容哈希和版本父子关系

报告接口输出：

- `curves.csv`
- `matches.csv`
- `curriculum.csv`
- `curves.png`
- `curves.svg`

图表至少包含：

1. information gain 指标：局部策略 KL 与状态占用差。
2. 当前目标及其 gate 胜率。
3. 完整人类池通过人数和整体胜率。
4. Elo。
5. prompt、cached input、completion 和总 token。

目标切换、晋级、拒绝、回滚和停滞以竖线或标记叠加在曲线上。`curriculum.csv` 每行记录 act、版本、当前目标、锁定人数、目标 gate 分数、认证通过人数和课程事件。

## 12. CLI 行为

本地工作流保持统一：

```bash
agentbench hl validate --config configs/hl/29_rollman-curriculum.yaml
agentbench hl run --config configs/hl/29_rollman-curriculum.yaml \
  --run-dir .agentbench/29_rollman/runs/run-<id>
agentbench hl resume --config configs/hl/29_rollman-curriculum.yaml \
  --run-dir .agentbench/29_rollman/runs/run-<id>
agentbench hl report --run-dir .agentbench/29_rollman/runs/run-<id>
```

`validate` 必须在任何模型调用前验证来源版本、冻结后端、SDK、完整人类池、配置约束和 API key 变量名。`--dry-run` 完成同样校验并生成上下文预览，但不读取 API key、不创建 provider 会话。

## 13. 测试要求

单元测试覆盖：

- 严格配置解析和非法组合拒绝。
- 来源 manifest、文件列表和内容哈希校验。
- 导入 origin 不调用模型。
- `weakest_failed` 从 `rank14`、`rank15` 中选择 `rank15`。
- 目标通过后选择 `rank14`。
- 16/16 后写入唯一成功事件并停止。
- gate 失败不触发完整认证。
- 候选丢失锁定对手时拒绝并回滚。
- 目标切换后阶段分数基准重置。
- 停滞计数、暂停和 resume 状态一致。
- prompt 仅引用当前目标的比赛证据。
- 固定模式保持原有 rank1 行为。
- curriculum 事件生成稳定的 CSV 和曲线数据。

集成验证覆盖：

- 全量测试套件。
- 冻结来源审计。
- 课程配置 dry-run。
- 从 `v000001` 导入的新运行 origin 与来源内容哈希一致。
- origin 的本地 rank15 gate 和 16 人认证结果可重现。
- 启动首个付费 act 前，不存在 provider 调用记录。

## 14. 完成判定

Framework 仅在最新完整认证满足以下条件时写入：

```json
{
  "event_type": "run_completed",
  "reason": "all_human_opponents_defeated",
  "passing_human_opponents": 16
}
```

任何停滞、回滚、provider 失败或评测异常均不得伪装为成功。成功版本、完整认证回放、事件日志、来源哈希、冻结配置、Experience、版本对象和报告共同构成实验复现包。
