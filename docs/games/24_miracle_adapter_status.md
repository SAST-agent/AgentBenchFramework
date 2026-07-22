# 24_miracle 接入状态

> 本文件是 AgentBench 第24届 Miracle 接入工程的滚动状态记录。
> 一切以磁盘文件、Git 状态、结果文件为准，不依赖聊天记忆。
> 维护依据：`SKILL.md`（接入原则与停止条件）。

## 资产来源（只读，不得修改）

| 角色 | 路径 | 性质 |
|---|---|---|
| Miracle 原始裁判语料 | `C:\Users\gongh\Documents\agentbench\backend_sources\corpus\24_miracle` | 受保护真实 Git 克隆内，禁止 reset/覆盖/stash |
| 16 份决赛人类策略 | `C:\Users\gongh\Documents\agentbench\top_algorithms\corpus\24_miracle_final` | 同上；含 `MANIFEST.tsv` |
| 高翔遗留工程 | `C:\Users\gongh\Desktop\AgentBench-gaoxiang\AgentBench-gaoxiang` | 无 `.git` ZIP 副本，只读参考 |
| 高翔关键文件 | `tools\miracle\run_match.py`、`analyze_trace.py`、`trajectory_digest.py`、`corridor_digest.py`、`reachability_digest.py`、`tools\miracle\ifelse_bot\main.py` | 复用 `run_match.py` 作为第一阶段比赛执行器 |
| Framework（本工作仓库） | `C:\Users\gongh\Documents\AgentBenchFramework` | 真实 Git 克隆，分支 `gongheng/24-miracle-adapter` |
| Results 聚合器（消费端权威） | `C:\Users\gongh\Desktop\AgentBenchResults-main\...` | 无 `.git` ZIP 副本，只读；`scripts\aggregate.py`、`report_builder.py` |

## Miracle 忠实比赛结构

```
原始 Judge 进程
  ├── AI 进程 A（策略入口）
  └── AI 进程 B（策略入口）
```

第一阶段**复用高翔 `run_match.py`**，不把历史选手改写成框架内部 `RuleBasedAgent`。

## 阶段2：数据契约 + Framework 风险核验（已完成，源码级）

### Results 数据契约（`aggregate.py` / `report_builder.py` / `aggregate.yml`）

1. **目录布局**：`<data_root>/runs/<game>/<agent>/<run_id>/{run.toml,summary.json,events.jsonl}`。`aggregate.find_runs` 遍历三层目录，**无 `run.toml` 则跳过该 run**（`aggregate.py:34-35`）。
2. **win_rate 来源**：`aggregate` 与 `report_builder` **都不读 `events.jsonl`**。win_rate / h2h / elo 全部取自 `summary.json`（`aggregate.py:64`、`report_builder.py:39,41-43`）。→ **summary.json 的 win_rate 必须按换边归一化算对，否则网页直接错。events.jsonl 仅作审计交叉核对。**
3. **CI schema 校验（风险#8，已确认）**：`aggregate.yml` 只跑 `aggregate.py` + `report_builder.py` + 部署 Pages，**无 jsonschema / pydantic / 字段断言**。字段缺失被静默 coerce（`_get_meta_field` 默认 `""`、`s.get(...)` 默认 `None`→TOML 空串）。**CI 绿 ≠ 数据正确。**
4. **Framework 侧 `agentbench data check`（cli.py:116-163）**：是唯一"schema 校验"，但仅做**字段存在性**检查——run.toml 须有 `[run]` 的 `run_id,game,agent,type,created`；summary.json 须有 `run_id,game,agent,wall_hours,total_steps,win_rate`。**不**校验 h2h 方向、win_rate 取值、events 内容、Replay、normalized_result。
5. **data_root 约定**：`<data_root>` 是**包含** `runs/` 的目录。`Run.start` 会在 `data_dir` 下再拼一层 `runs/`（`run.py:67`），故必须 `Run.start(data_dir=<data_root>)`，**不能**传 `./runs`（否则双重 runs，见风险#6）。

### Framework 9 项已知风险核验

| # | 风险 | 判定 | 证据 (file:line) |
|---|---|---|---|
| 1 | `ENV_REGISTRY` 可能为空 | **CONFIRMED** | `env/registry.py:10` 起步 `{}`，仅显式 `register_env` 填充 |
| 2 | `Match` 换边后按原始 player 编号统计 | **CONFIRMED** | `arena/match.py:91-96` 按 `raw_winner==0/1` 计 agent1/agent2 胜，忽略自算的 `winner_agent`（line 146-152） |
| 3 | `Run` 只把 `winner==0` 计为胜利 | **CONFIRMED** | `tracking/run.py:164` `wins=sum(1 for w in episode_winners if w==0)` |
| 4 | `BaseRunner` 未传 `run_type` | **CONFIRMED** | `runner/base.py:55-60` 调 `Run.start(...)` 不传 run_type → 恒 `"eval"` |
| 5 | summary 先写盘后合并 result | **CONFIRMED** | `runner/base.py:83-84` `finish()` 写 summary.json 在前，`summary.update(result)` 在后 → `_execute` 返回值不落盘 |
| 6 | CLI 双重 `runs/` | **CONFIRMED** | `cli.py:205,214,222` 默认 `--data-dir ./runs`；`run.py:67` 再拼 `runs/` → `./runs/runs/...`；而 `data check` 默认 `./agentbench_data`（cli.py:244）→ 产出与校验目录互不相交 |
| 7 | `cpu_percent`/`cpu_pct` 不一致 | **CONFIRMED** | `sampler.py:87` 写 `cpu_percent`；`run.py:189` 读 `cpu_pct` → summary 中 CPU 恒 0（rss 一致，内存正常） |
| 8 | Results CI 无严格 schema 校验 | **CONFIRMED** | 见上 §Results.3 |
| 9 | 无自动化测试 | **CONFIRMED** | 仓库无 `tests/`、无 pytest 配置、无任何测试文件 |

**追加实锤**：`BaseRuleRunner`（rule_runner.py:43-97）在 `_execute` 内**正确**做了换边感知 `wins`（line 81-83），但 `log_episode` 传原始 winner（line 78）+ 风险#5 → **磁盘 summary.json 的 win_rate 在换边下错误，仅内存返回值正确**。`BaseEvalRunner` 不调 `log_episode` 且用 `Match`（风险#2）→ **磁盘 summary win_rate 恒为 0.0**。

### 其他关键源码事实

- **外部进程支持**：`env/stdio_protocol.py` 的 `SubprocessGameRunner`（line 181-266）可 spawn 子进程、4 字节长度前缀 + JSON 收发、`terminate→wait(5s)→kill`。**但不捕获 exit code、不强制 timeout**——适配层需自行 `process.returncode` 与超时控制。Miracle 的 Judge 内部自行管理两个 AI 进程（由高翔 `run_match.py` 编排）。
- **Elo 语义**（`arena/rating.py`）：`update(p1,p2,winner)`，winner=0/1/-1（draw）→ draw 计 0.5/0.5（line 69-74），与 skill 一致。eval 无 Elo 时留 `null`（Run 已含字段，不省略）。
- **events.jsonl**：`JSONLWriter`（`tracking/writer.py`）以 `open(path,"w")` 截断打开，buffer_size=64，flush/close 落盘；`Run.write(**kw)` 追加一行。崩溃在 buffer 中未 flush 的事件会丢——适配层需逐局 flush 或在异常路径 flush。
- **summary.json 字段**：`Run._build_summary`（run.py:166-184）产出的字段集合**恰好**等于 skill 要求的 summary 字段集，可直接复用。

## 适配层架构决策（被源码风险强制；零共享框架修改）

**核心**：不使用 `Match`、不使用 `BaseRunner.run()` / `BaseEvalRunner` / `BaseRuleRunner`。写一个 **Miracle 专用 runner**，直接驱动 `Run`：

1. `Run.start(game="24_miracle", agent=<被评策略>, run_type="eval", data_dir=<data_root>)` —— 显式传 run_type 与 data_root，**绕开风险#4/#6**。
2. 每局调用高翔 `run_match.py`（固定 seed、指定双方 camp、timeout、独立工作目录、stdout/stderr 隔离），捕获：`raw_winner`、双方分数、双方 exit code、Judge exit code、Replay 路径、异常、超时。
3. **归一化**：依据 `evaluated_agent_camp` 把 `raw_winner` 映射为 `normalized_result ∈ {win,loss,draw,error}` 与 `winner_agent`。崩溃/超时/Replay 缺失 → `error`，**不计入有效胜率**（rank03 崩溃属无效证据）。
4. 每局 `run.write("game", **{全量 skill 字段：game_id, seed, 双方策略ID, 双方源码SHA256, 双方camp, raw_winner, winner_agent, evaluated_agent_camp, normalized_result, 双方分数, draw, 起止时间, duration, 双方+Judge exit code, timeout, exception, Replay路径+SHA256, 是否有效, 是否恢复, 是否重跑})`。
5. **喂数据给 Run 使其磁盘 summary 正确**（绕开风险#2/#3/#5）：
   - `run.log_episode(reward=…, steps=<Judge↔AI 轮次数>, winner=<0 若被评方胜, 1 若负, -1 平>)` —— winner 用**归一化**值，使 Run 的 `wins=sum(w==0)` 正确；`steps` 用可审计的轮次计数。
   - `run.log_h2h({<行策略>: {<列策略>: <行胜列的有效胜率>}})` —— 平局不计入分子，draw 数量另行保留。
   - eval 不需要 Elo 则 `best_elo/final_elo/elo_history` 保持 `null`（Run 默认即如此）。
6. `run.finish()` 写出 run.toml / summary.json / events.jsonl。因所有累计量在 finish 前已正确注入，**磁盘 summary 的 win_rate 正确**（绕开风险#5）。

**total_steps 口径**：1 step = Judge 向当前 AI 发一次公开状态并收到一次动作响应。逐局记录该局轮次数，`total_steps = Σ 各局轮次数`。保守、可审计、可解释。

**winner / camp 映射**：永远区分 `raw_winner`（Judge 的 camp/player 编号）、`winner_agent`（映射后策略身份）、`evaluated_agent_camp`（被评方本局阵营）、`normalized_result`。换边后**不**把 `raw_winner==0` 当被评方胜。

**rank03 崩溃**：当前对手程序崩溃，属无效证据，**不计入**被评 Agent 有效胜率，单独标记。

## 阶段进度

- [x] 阶段1：入口与 Git 审计；工作仓库 `Documents\AgentBenchFramework`（分支 `gongheng/24-miracle-adapter`）已建立
- [x] 阶段2：数据契约 + Framework 9 项风险核验（本文档）
- [x] 阶段3：冻结 Miracle 资产身份与 SHA256 → `docs/games/24_miracle_assets.json`（Judge main.py/Data.json、sample_ai、16 份策略 zip、高翔 run_match.py / ifelse_bot / historical_lessons，含上游 archives SHA256SUMS）
- [~] 阶段4：实现外部比赛适配层 — 纯逻辑 + Run 胶水已完成（result.py / driver.py）；子进程包装(run_match)+顶层 runner 待写
- [~] 阶段5：单元测试 — **39 项通过**，覆盖 skill 17 项中的 16 项（仅"无残留进程"待 4b）
- [x] 阶段6：≤4 局 smoke — 已完成 4 局（BOTH_GROUPS_PASS，见下）
- [x] 阶段7：本地 schema + aggregate + 网页交叉验收 — 三向全绿（见下）
- [x] 阶段8：完整接入报告 — 本文档 + docs/games/evidence/

## 阶段4 实现进展

**运行时要求**：Framework/Results 与 Judge 都需 Python 3.11+（`import tomllib`）。本机用 `py -3.13`（3.13.5）；默认 `python` 是 3.10 不可用。pytest/jinja2 已装入 3.13（清华镜像）。

**已实现（工作仓库 `src/agentbench_frame/games/miracle/`）**：
- `result.py`（纯函数，零依赖）：`derive_raw_winner`/`scores_from_end_info`（Judge end_info 语义）、`normalize`/`finalize`（win/loss/draw/error 归一化，崩溃/超时/Replay缺失→error）、`compute_win_rate`（有效胜/有效对局，平局计分母不计胜）、`compute_h2h`（行胜列，平局计入分母使两者可不等于1）、`read_replay_header`（从 Replay 头读 map_type/day_time 作 seed provenance）、`sha256_file`、`select_games_to_run`/`would_rerun_successful`（可恢复不重复）、`to_event_record`（events 全字段）。
- `driver.py`（击败风险 #2/#3/#5 的关键）：`feed_outcomes_to_run` 把**归一化**胜负喂给 `Run.log_episode`、error 局不记 episode、全量 `game` 事件入 events、h2h 入 summary；磁盘 summary.json 的 win_rate 在换边下正确。

**测试（`tests/miracle/`，39 项通过）**：`test_result.py`(29) + `test_driver.py`(7) + `test_results_pipeline.py`(3)。覆盖 skill 第 1–4、5–9、10、11、12、13、14、15、16 项；含"磁盘换边 win_rate 正确""无双重 runs""events↔summary↔网页三向一致"。

**待写**：子进程包装（拷贝高翔 run_match.py + `MIRACLE_JUDGE_DIR` env 覆盖 + 调用 + trace 解析 + Replay 哈希 + exit code 捕获 + 进程清理）、顶层 `MiracleEvalRunner`、第 17 项"无残留进程"测试。

## 已确认的 Results 上游缺陷（跨平台 bug）

`AgentBenchResults/scripts/aggregate.py` 写 `path = str(run_dir.relative_to(data_dir))` → Windows 上为反斜杠 `runs\24_miracle\...`，在 TOML 字符串里非法 → `registry.toml` 解析失败 → `report_builder.load_registry` 崩溃。Linux CI 用正斜杠故从未暴露（"CI 绿 ≠ 数据正确"的实证）。

处理（最小、透明、不伪造绿灯）：
- 工作仓库 `vendor/results_local/aggregate.py`：逐字副本 + 一行 `as_posix()` 补丁（零语义改动，仅路径分隔符），仅供本地 Windows 验证；上游仍为权威实现。
- `test_results_pipeline.py::test_upstream_aggregate_emits_invalid_toml_on_windows`：**证据测试**钉住上游 bug（断言上游 registry.toml 在 Windows 上确为非法 TOML），不删除、不放宽。
- Linux 生产 CI 不受影响；report_builder.py 无需改动（registry.toml 合法后即可工作）。

## 阶段6 smoke 结果（session 20260721-172308_e026da，4 局，BOTH_GROUPS_PASS）

驱动 `tools/miracle_smoke.py`（session 隔离、不删旧/拒重名、严格门槛、独立 PID+create_time 残留核验、UTF-8 全量日志）。Judge=受保护仓库权威 `judge_dev_logic`；sampleA/sampleB=同源不同逻辑ID；if-else=高翔冻结 bot（只读运行）。授权上限 4 局，未追加；上一条 smoke 命令曾被 429/工具拒绝、**未启动任何对局**（不记为结果）。

- **GROUP1**（sampleA vs sampleB，换边）：
  - g1_00 camp0：raw_winner=0 → **win**；scores {0:30003,1:26003}；steps=166；map=(mt=0,dt=0)。
  - g1_01 camp1：raw_winner=0 → **loss**（换边归一化正确：raw_winner=0 不总等于被评方胜）；与 g1_00 同分/同 steps 系同源确定性 AI 在相同随机地图上的可重现结果。
  - summary：attempted=2 valid=2 invalid=0，win_rate=0.5（1W/1L）。
- **GROUP2**（miracle_ifelse vs sampleB，换边；G1 严格通过后才运行）：
  - g2_00 camp0：raw_winner=0 → **win**；scores {0:30019,1:1017}；steps=232；map=(mt=0,dt=1)。
  - g2_01 camp1：raw_winner=1 → **win**；scores {0:28,1:30060}；steps=630；map=(mt=1,dt=1)。
  - summary：attempted=2 valid=2 invalid=0，win_rate=1.0（2W）。两局 map/分数/steps 均不同 → 真实独立对局。
- 每局 `requested_seed/effective_seed=null`、`reproducible_from_seed=false`、`realized_randomization={map_type,day_time}`（**不表述为 seed**）。
- 独立 PID 残留核验：每局 3 个受管进程（judge/ai0/ai1）全部 clean，0 residual / 0 reused；最终全局检查 CLEAN。
- evidence 全量：`.smoke/sessions/20260721-172308_e026da/`（manifest.json / smoke.full.log / group1.json / group2.json / repair_note.txt / data/runs/.../{run.toml,summary.json,events.jsonl} / work/.../{trace,replay,result.json,stdout,stderr}）。

## 阶段7 三向交叉验收（在 smoke 真实数据上，全绿）

1. Framework schema check（`agentbench data check`）：**2 valid, 0 invalid**。
2. vendor（Windows 补丁）aggregate：2 runs → registry.toml。
3. Results report_builder：2 runs / 1 game → site/data.json。
4. 三向一致：sampleA `summary_wr=0.5 == web_wr=0.5 == events 重算 0.5`；miracle_ifelse `1.0 == 1.0 == 1.0`；每局 seed 字段（requested/effective=null, reproducible=false）OK。
5. Windows 路径补丁边界：`test_vendor_boundary.py` AST 等价（仅路径分隔符，不改指标/h2h/发现逻辑）。

## 新发现的 Framework 缺陷（验收时暴露，已最小修复 + 失败测试先行）

`tracking/run.py::Run._write_toml` 写字符串值时不转义反斜杠/引号 → 当 `[config]` 含 Windows 路径（judge_dir_resolved）时 run.toml 非法 TOML（`Invalid hex value`，`\U` 被当 unicode 转义）→ `agentbench data check` 失败。与 aggregate 路径 bug 同类（风险清单 9 项之外的**新缺陷**）。

修复（最小、失败测试先行、报告单列、不推送）：`Run._write_toml` 增 `q()` 转义（`\`→`\\`、`"`→`\"`）用于所有字符串值；`test_driver.py::test_run_toml_valid_with_windows_path_in_config` 先红后绿。smoke 既有 run.toml 用修复后的 writer 从权威 summary.json 重新序列化（events/replay/summary 指标未动；见 session/`repair_note.txt`）。

## 双版本测试证据

Python 3.13.5 + 3.11.15（uv 托管 `cpython-3.11.15`）各 **93 passed / exit 0**。证据（完整未截断、UTF-8）：`docs/games/evidence/{py_interpreters.txt, py313_tests.txt, py311_tests.txt, proctree_redlight.txt, match_runner_redlight.txt, smoke_schema_check.txt, smoke_three_way.txt}`。

## 下一阶段门槛（未授权不跨越）

- 完整 16 人矩阵、16×16 round-robin、RL 训练、hidden evaluation、push、PR、上传 Results、合并 main、删除失败实验——**均需用户明确授权**。
- 策略研究（rank13/rank16 Archer lattice）在完整评测协议预注册 + 用户授权前不恢复。
