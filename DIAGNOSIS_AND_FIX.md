# HL 迭代闭环诊断 + 修复方案

> 诊断对象：`AgentBenchFramework/src/agentbench_frame/hl/`（分支 `liuzhuo/lostspace`）
> 参照对象：`origin/zhongkaiyu` 的 `snakego/`（同框架、另一游戏 26_snakego 的 HL 实现）
> 检查表：5 项目标（闭环 / 规则文档+回放验证 / obs+action 契约 / IG 数据+缺失原因 / 有效策略更新+曲线）
> 日期：2026-08-01
> 状态复核：2026-08-02（对照 HEAD 之后提交复核；Fix-C 已落地、total_tokens 已修，见 §0.5）

---

## 0.5 状态快照（2026-08-02 复核）

| Fix | 状态 | 证据 |
|---|---|---|
| **A** policy_kl 诚实性 | ❌ 未做 | `local_policy_kl_trace` 仍返回扁平 `List[float]`，`None` 仍喂进 `epsilon_smoothed_distribution` → uniform → KL=0（`controller.py:392-393`、`distribution.py:236`） |
| **B** decision_space 契约文件 | ❌ 未做 | 无契约文件，Q3 仍散在 `distribution.py` |
| **C** ν 从 instrumentation 记录 | ✅ 已做 | `reference_recorder.py`（c17189d）解析真实 `trace.jsonl` → 带 `transcript` 的 `ReferenceSample`；probe 重放 transcript（7eb3ab6）；`reference_seed.py` 降级为 TEST FIXTURE（79bc6b6） |
| **D** REPLAY_SKILL act | ❌ 未做 | `GAME_RULES.md` 仍只注入 4 行 blurb（`context.py:42-47`） |
| **E** 评测池+自动出图 | ❌ 未做 | `cli.py` 不自动调 `plot_curves`；`win_rate` 仍恒 0 |
| **F** files_touched / total_tokens | 🟡 半成 | `total_tokens` 两个 runner 已填充（旧“恒 null”说法**过时**）；`files_touched` 仍恒 `[]` |

**文档遗漏的根因（比手写 ν 更直接）：** 未提交的 `adapter.py` diff 加了 `Path(dest).resolve()` —— Windows `CreateProcess` 把相对 `argv[1]` 按 cwd 解析，相对 `dest` 嵌套成 `.hl_codebase/stage/.hl_codebase/...`，candidate 起不来，probe 把 OSError 吞成 `None` → uniform → **KL=0 掩盖**。该修复已于 50af612 提交。`record_real_nu.py`/`diag_emissions.py` 移入 `tools/hl/`。

---

## 0. 一句话结论

`liuzhuo/lostspace` 的 HL 模块**是一个真实可跑的迭代骨架**（act→snapshot→eval→probe→emit 全链路实现并多次跑过真实 LostSpace logic），**但 5 项目标每一条都有明确缺口**；`zhongkaiyu` 的 SnakeGo 实现把同样的研究目标做得**更干净**，主要因为它选了一个动作空间小且上下文无关的游戏（6 个 macro-action，固定 SUPPORT），从而绕开了 LostSpace 那条线最痛的“A(s) 不完整 / ν 手写 / KL 被静默折 0”三连坑。修复策略：**把 zhongkaiyu 的契约式做法移植回 LostSpace，并对 LostSpace 动作空间大的现实做显式降级与诚实标注。**

---

## 1. 现状诊断（5 项目标逐条）

### Q1. 闭环：官方逻辑 → 对战 → replay → 改 → 存版本 → 再评测（日志可追溯）

**已实现且跑通**（`controller.py:108-279`，`cli.py:443-447`）：
- `agent_act` → 跑编码 agent（`ApiCodingRunner`/`ClaudeCodeRunner`）→ `HLCodebase.snapshot()`（`codebase.py:105-125`，sha256 内容哈希）→ 发 `version` → 冻结 `BenchmarkSpec` 评测（`save_replays=True`，`cli.py:219`，写 `runs/25_lostspace/<run_id>/<run_id>/`）→ 发 `eval` → probe 两版本 → 发 `policy_kl`/`occupancy_shift` → 发 `budget`。
- 全部事件经 `HLEventWriter`（`events.py:64-102`）带公共 schema 追加进 `events.jsonl`。
- 实测：`hl-run-0730-fix/events.jsonl` 有 10 个完整 act 链 + 6 个真实 replay JSON。

**缺口：**
1. **win_rate 恒为 0.0**（所有 run）——“再评测”在跑但从未体现分数提升。
2. 多个 act 是 `noop` + `failure_reason:"claude CLI timed out"`（撞 600s `--claude-timeout`）。
3. `version` 事件里 `files_touched` 恒为 `[]`（只有 `FakeRunner` 填，`runner.py:144-150`）。
4. ~~claude-CLI 路径 `total_tokens` 恒为 `null`（`runner.py:287`）~~ —— **已修复**：`ClaudeCodeRunner`（`runner.py:285`）与 `ApiCodingRunner`（`runner.py:370-373`）现在都填充 `total_tokens`。仅 `files_touched` 仍恒 `[]`（Fix-F 未完）。

### Q2. 人工写清规则/回放字段/关键事件/常见误读，再由 Agent 润色 + 一次真实回放解析验证

**已实现：**
- 人工文档 `lostspace/GAME_RULES.md`（带 file:line 指向 logic 源）+ `lostspace/replay_format.md`。
- `context.py` 注入浓缩 blurb：`_GAME_RULES_BLURB`、`_DATA_SCHEMA_BLURB`（含 interprops-int-coded 关键警告）、`_PLAYBACK_RECIPE`。
- agent 可读真实 replay（`read_replay` 工具 + 座位 0 digest）。
- `EXPERIENCE.md`（`experience.py`）是 agent 自写的策略教训，真实存在。

**缺口（本目标最大问题）：**
1. **没有“agent 润色规则/回放文档本身”这一步**——`GAME_RULES.md`/`replay_format.md` 静态、agent 只读不改。
2. **没有专门的“一次实际回放解析验证”act**——agent 只在策略 act 里顺带读 replay。
3. **`GAME_RULES.md` 没接到 HL prompt**——`context.py` 只给 4 行 blurb，agent 甚至不被指向它。

### Q3. 明确 observation、合法 macro-action、action mask、终止条件、动作支持集

**已实现**（`distribution.py`、`reference.py`）：
- Observation = `roundbegin` 帧字段；`enumerate_legal_actions` 镜像 `Player.get_legal_actions()`；终止条件按 `status` 分流（`DIED/ESCAPED/SKIP/ERROR`→空集）；`epsilon_smoothed_distribution` = `(1-ε)·onehot + ε·U(A(s))`。

**缺口：**
1. ~~ν 是手写 8 个合成决策点~~ —— **已修复（Fix-C）**：`reference_recorder.py` 从真实 match trace 解析出带 `transcript` 的 ν，`reference_seed.py` 降级为测试 fixture。**“真实编辑却 KL=0”的真正根因**是 `adapter.py` 的 Windows 相对路径 bug（`Path(dest).resolve()` 修复，50af612），见 §0.5。
2. macro-action 投影是粗糙 v1（`distribution.py:171-178` 自述）；per-primitive 分布**不捕捉一回合内 primitive 间相关性**（`distribution.py:25-27`）——同 primitive 分布但不同回合编排的两 agent 会 KL≈0。
3. `round_state.py` 名不副实（仅回合计数器）。
4. `get_action_distribution` 协议（`distribution.py:17-20`）写了但**无人实现、无代码路径检查**。
5. `out_of_support` flag 在 probe/distribution 都算了但**没进 `policy_kl` 事件**（见 Q4）。

### Q4. 统一口径保存逐 episode/iteration IG；无法算严格 KL 时明确记缺失原因，不冒充

**已实现：**
- `policy_kl` 事件带 `local_policy_kl_trace`（原始 per-decision 列表）+ `epsilon`，原始存储不预聚合（`distribution.py:266-290`）。
- 双方都发射时严格 KL 总可算（ε 平滑保证 q>0）。真实 KL>0 出现（act7 `[0,0,0,1.37,0,1.85,0,0]`）。

**缺口（本目标的诚实性漏洞）：**
1. **缺失 KL 没有显式 reason 字段**——`version_after=None` 时**不发 `policy_kl` 事件**，缺失仅靠“事件缺席”隐式表达。
2. **【最严重】trace 内单点 `None` 被静默折成 KL=0.0**——某版本在某 sample 上 `None`（`controller.py:392-393`）→ `epsilon_smoothed_distribution(None)`→uniform→`KL(uniform||uniform)=0.0`。一条全零 trace 既可能“两版本选择相同”也可能“两版本都没响应”，events.jsonl **无法区分**。（补充：大批 `None` 的另一个来源是 adapter 相对路径 bug 导致 candidate 起不来——已修，但单点 None 静默折 0 本身仍未修，见 Fix-A。）
3. `out_of_support` flag 算了被丢弃，没进 `policy_kl` 事件。`failure_reason` 在 `version`/`budget` 上诚实填充，但 `policy_kl` 上无对应字段。

### Q5. 至少一次有效策略更新 + 版本对齐 score/IG 曲线；失败/缺失/不完整如实保留

**已实现：**
- 有效策略更新（`edit_type≠noop` 且某点 KL>0）实测发生（act7 等）。
- `plot_curves.py` 出 `score_iteration.png` + `ig_iteration.png`，按 act 尾数 join；**诚实保留**：`win_rate=None`→gap 不补 0；失败 act→红色 `✗`；act1 无 IG→gap。`plot_curves.py` 是纯读取器，不造假。
- 实测跑通：`plot_curves --name hl-run-0730-fix` → “10 acts · 0 incomplete · 5 failed”。

**缺口：**
1. **无任何 run 的 win_rate 非 0**——有效更新仅在 IG 层，从未转化为分数提升。score 曲线是 0 处一条平线加红叉。
2. run 不自动生成曲线，要手动 `plot_curves`。
3. `figures/hl-v1_iteration_curves.png` 是旧的无源不可复现产物（`PLAN_hl_checklist.md` 已点名）。

---

## 2. zhongkaiyu 的可借鉴做法（按目标映射）

zhongkaiyu 的 `snakego/decision_space.py` 开头就明文写：**“Defines the observation, the legal macro-action set, the action mask, the termination conditions, and the action-support set used for information-gain computation.”**——它把 Q3 当成一个独立契约文件来写，而不是散落在 `distribution.py` 里。这是最值得抄的结构。

| 目标 | zhongkaiyu 做法 | lostspace 现状 | 借鉴点 |
|---|---|---|---|
| Q1 闭环 | `loop.py`：step()=play→save replay→learn→eval→accept/reject→snapshot→IG→log；**多对手轮转**（rank06/09/11/13/15）+ **冻结评测池**（rank13/15 × 3 seed）+ **3 次拒绝后随机扰动逃局部最优** | 单/少对手、ν 手写、win_rate 恒 0 | 多对手冻结评测池 + 拒绝扰动；但 LostSpace 对手是 16 rank 人类，更难 |
| Q2 规则+回放+验证 | `docs/RULES.md`（人工写清）+ `docs/REPLAY.md`（字段表）+ **`docs/REPLAY_SKILL.md`**：明确“人工写清…由 Agent 润色，并用一次真实回放解析验证（见末尾「验证」一节）” | blurb 注入，无润色/无验证 act | 把 REPLAY_SKILL 模式移植：人工写规则文档 → 作为 act 0 让 agent 润色 → 跑一局真实 replay → agent 解析并对照 score_dic 自验 |
| Q3 契约 | `decision_space.py` 独立文件：`MACRO_ACTIONS={1..6}`、`SUPPORT=ACTION_IDS`（固定、与 mask 同源）、`Observation` dataclass（冻结）、`compute_mask(eng)`、终止条件 | 散在 `distribution.py`，ν 手写，`round_state.py` 名实不符 | 立 `decision_space.py` 式独立契约文件；`SUPPORT` 与 mask 同源 |
| Q4 IG+缺失原因 | `ig.py`：`kl_to_human` 不完整时显式 `{"status":"incomplete_replay","reason":...,"kl":None}` 且 **“do NOT substitute any other metric”** | 缺失靠事件缺席；单点 None 静默折 0 | **`policy_kl` 事件加 `per_sample_status` 字段**，None 发射记 `reason="no_emission"`，不折 0 |
| Q5 曲线+诚实保留 | `curves.py` 双序列对齐 + `reports/README.md` 三类分组（对照/HL 主线/被拒）+ `curves.json` 三组分开 + 红叉保留 | `plot_curves.py` 已诚实，但 run 不自动出图、无三类分组 | 自动出图 + 报告分类（对照=纯调参 IG≈0 / HL 主线 / 被拒） |

**zhongkaiyu 也没解决 score 卡死**——v9 卡在 ~0.41 局部最优，`ITERATION_REPORT_7` 诚实记录 v12 五变体均未超过 v9，并诊断是 harness 阻塞点（评测集太小、对手是自家 scorer 而非真人）。所以**两条线都缺“真正涨分”**，这不是本修复能单独解决的，但 zhongkaiyu 的诚实保留+三类分组让“没涨分”这件事**可见且诚实**。

---

## 3. 修复方案（按优先级，每项标注目标号 + 是否需要改 logic）

### Fix-A【Q4，最高优先，诚实性】`policy_kl` 事件携带 per-sample 缺失原因，禁止静默折 0

**问题**：单点 `None` 被折成 KL=0.0，无法与“两版本真选同”区分（诊断 §1.Q4 缺口 2）。

**改法**：
1. `distribution.py` 的 `local_policy_kl_trace` 返回**结构化 trace**，每点带 `{kl, status, reason}`：
   - `status="ok"`：双方都发射且 `chosen∈A(s)`，`kl` 为真实值。
   - `status="out_of_support"`：`chosen∉A(s)`，`kl=None`，`reason="chosen_not_in_support"`（已有 `out_of_support` flag，只需透传）。
   - `status="no_emission"`：某版本返回 `None`，`kl=None`，`reason="version_X_unresponsive"`。
2. `controller.py:371-402` `_measure_policy_kl`：不再把 `None` 当 uniform 喂进 KL；改为按上面分状态记录。
3. `events.py` 的 `policy_kl` 事件加字段：`per_sample_status: [...]`、`n_ok`、`n_missing`、`missing_reasons: [...]`、`kl_mean`（仅对 `status="ok"` 求平均，缺失不进分母）。
4. `plot_curves.py`：IG 曲线的 y 值用 `kl_mean`（ok-only）；`n_missing>0` 的点画半透明 + caption 标注“N samples missing”。

**借鉴**：`ig.py:47-58` 的 `{"status":"incomplete_replay","reason":...,"kl":None}` 模式。

**不改 logic**：纯 HL 模块改动。

### Fix-B【Q3】立 `lostspace/decision_space.py` 独立契约文件

**问题**：Q3 契约散在 `distribution.py`，ν 手写，`round_state.py` 名实不符（诊断 §1.Q3 缺口 1-3）。

**改法**：新建 `src/agentbench_frame/lostspace/decision_space.py`，照搬 zhongkaiyu 的结构：
- `MACRO_ACTIONS`：LostSpace 顶层 `play()` 返回的 macro 集（move/attack/interact/use_tool/place_trap/detect/escape/finish）——**显式声明这是 macro 投影，非完整 primitive 空间**（诚实降级，对应诊断 §1.Q3 缺口 2）。
- `SUPPORT`：固定有序 macro id 列表，**与 `compute_mask` 同源**，跨版本不变。
- `Observation` dataclass：冻结 `roundbegin` 字段（现有 `ReferenceSample.observation` 升格）。
- `compute_mask(state) -> ActionMask`：返回每个 macro 的 ✅❌（现有 `enumerate_legal_actions` 重构）。
- `termination(state) -> Termination`：把 `status` 分流逻辑集中于此。
- ~~删除/重命名 `round_state.py`~~ —— **改判**：`round_state.py` 是 auto-naming 用的持久化单调回合计数器（`cli.py:340` 调用 `next_round`），是合法功能，**保留原名不改**。

**借鉴**：`decision_space.py` 开头契约声明 + `SUPPORT=ACTION_IDS`。

**不改 logic**：纯框架层。

### Fix-C【Q3】ν 从冻结 reference roll instrumentation 记录，替代手写 8 点

**状态：✅ 已做（fe8a5d1 / c17189d / 7eb3ab6 / 79bc6b6）**。`reference_recorder.py` 解析真实 match trace → 带 `transcript` 的 ν；probe 重放 transcript；`reference_seed.py` 降级为测试 fixture。残留：README §10 的“not yet built”若仍存在则删（随 recorded-reference-set 归档）。

**补充根因**：手写 ν 只是次要因素。“真实编辑却 KL=0”的直接原因是 `adapter.py` 相对路径在 Windows 下嵌套、candidate 起不来、probe 吞 OSError 为 `None` → uniform → KL=0。已修（`Path(dest).resolve()`，50af612）。

**不改 logic**：instrumentation 通过子类化 / monkeypatch `Player` 完成，不修改 logic 源（遵守 CLAUDE.md “不改 match.py/ladder.py/evaluator.py”精神）。

### Fix-D【Q2】REPLAY_SKILL act：人工写规则 → agent 润色 → 一次真实回放解析验证

**问题**：无润色步骤、无验证 act、`GAME_RULES.md` 未接 prompt（诊断 §1.Q2 全部缺口）。

**改法**：
1. 把 `lostspace/GAME_RULES.md` + `replay_format.md` 合并升格为 `lostspace/docs/REPLAY_SKILL.md`，照 zhongkaiyu 格式：字段表 + op 动作码 + 关键事件 + **常见误读** + **末尾「验证」一节**。
2. 在 `context.py` 把 `GAME_RULES.md` 全文接进 prompt（不再只给 4 行 blurb）。
3. 新增 act 类型 `rules_validation`（act 0）：
   - 给 agent 规则文档 + 一局真实 replay 路径。
   - agent 任务：解析该 replay 的某回合，写出每个字段含义，对照 `score_dic` 验证自己的解析正确。
   - 产出 `RULES_VALIDATION.md`（agent 写）+ 验证结果（pass/fail + 证据）。
4. `events.py` 加 `rules_validation` 事件类型，记 `validation_status`、`fields_checked`、`mismatches`。

**借鉴**：`docs/REPLAY_SKILL.md` 的“末尾验证一节”结构。

**不改 logic**。

### Fix-E【Q1/Q5】多对手冻结评测池 + 自动出图 + 三类报告分组

**问题**：win_rate 恒 0、曲线不自动生成、无三类分组（诊断 §1.Q1 缺口 1、§1.Q5 缺口 2-3）。

**改法**：
1. `cli.py` `--ladder-opponent` 已支持多对手；固化一个“冻结评测池”配置（例如 rank 1/3/6/9 各 2 seat），写进 `BenchmarkSpec`，跨版本不变。
2. `controller.py` 评测后自动调 `plot_curves` 出图到 `<round>/figures/`，不再要手动。
3. `compare.py` 扩展为三类分组：`hl_main`（结构化 edit）/ `control_weight_tuning`（纯调参，IG≈0）/ `rejected_hl`（edit 了但 win_rate 没涨/回退）。借鉴 `reports/README.md`。
4. 保留旧 `figures/hl-v1_iteration_curves.png` 的“无源不可复现”标注，或删除。

**借鉴**：`loop.py` 多对手轮转 + `curves.py` 自动双序列 + `reports/README.md` 三类分组。

**注意**：**Fix-E 不能保证 win_rate 真涨**（zhongkaiyu 也没解决）。它的目标是让“没涨分”可见且诚实，而非强行涨分。这要在文档里写明，不要假承诺。

**不改 logic**。

### Fix-F【Q1 小修】`files_touched` 真实填充（total_tokens 已修）

**问题**：`files_touched` 恒 `[]`（诊断 §1.Q1 缺口 3）。`total_tokens` 半——两个 runner 现在都填充，不再需要回填逻辑，保留“缺失→null 不造假 0”原则即可。

**改法**：
1. `ApiCodingRunner`/`ClaudeCodeRunner` 在 `str_replace`/Write 工具调用时记录被改文件路径，填进 `AgentRunResult.files_touched`。

**不改 logic**。

---

## 4. 修复优先级与依赖

```
Fix-A (Q4 诚实性)        ← 最高优先，独立，先做
Fix-B (Q3 契约文件)      ← 独立，可与 A 并行
   ↓
Fix-C (ν instrumentation) ← 依赖 B 的 SUPPORT 定义
   ↓
Fix-D (REPLAY_SKILL act) ← 独立，可与 C 并行
   ↓
Fix-E (评测池+自动出图+分组) ← 依赖 A 的 per-sample status（IG 曲线要用）
Fix-F (小修)              ← 随时可做
```

**TDD**（按 CLAUDE.md 工作流）：每项先写 `tests/hl/test_*.py`，再实现。`Fix-A`/`Fix-B`/`Fix-C` 应各有一个端到端测试：跑一个 fake act，断言 `policy_kl` 事件含 `per_sample_status` 且 `None` 发射不折 0（A）；断言 `SUPPORT` 与 `compute_mask` 同源（B）；断言两次 `record_reference_states` 产出相同 states（C）。

## 5. 不解决但要诚实记录的（non-goal，借鉴 zhongkaiyu ITERATION_REPORT_7）

- **win_rate 真正提升**：两条线都卡住。zhongkaiyu 诚实记录 v12 五变体未超 v9 并诊断 harness 阻塞点。LostSpace 也应如此：在 `EXPERIENCE.md`/报告里诚实写“win_rate=0 持续，诊断为 X”（评测池弱 / agent 编辑没触到得分关键路径 / LostSpace 动作空间过大），而非回避。
- **完整 primitive A(s)**：本修复显式声明 macro 投影降级（Fix-B），不假装是完整 primitive 空间。
- **RL 侧对称**：`get_action_distribution` 协议仍无 agent 实现，本轮不补，记为 follow-up。
