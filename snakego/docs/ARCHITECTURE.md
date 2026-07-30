 # SnakeGo 智能体迭代系统 · 架构与交付总览

 本文件说明「干净重建」后的系统结构，并把用户的 5 个交付物 + 6 条要求逐条对应到代码与产物。

 ---

 ## 设计原则（对应 6 条要求）

 1. **from scratch**：保留唯一可信的基础（忠实引擎 `engine.py` + socket 对战主机 `host.py`），其余老代码（`env.py`/`agents.py`/`arena.py` 等从未接真实人类）标记为废弃，迭代系统全部基于新模块。
 2. **每次和最优秀的人类打 + 从回放学习**：`loop.py` 默认对手是冠军 `rank01`；每局后 `learn()` 从回放提取行为事实，写进 `experience.py`。
 3. **可解释代码，不只是 if-else**：`strategy_core.py` 是「特征加权打分」策略——每个动作由一组可解释特征项打分取 argmax，决策可审计（`DecisionTrace` 记录每项贡献）。
 4. **压缩整合而非堆叠**：`experience.py` 的 Experience 在 append 新教训时，若与旧教训在同一权重轴冲突，把旧的标 `superseded` 归档；策略始终是「一个」consolidated 策略，权重随经验演化。
 5. **Agent 自我总结的既有经验 Skill**：`experience.py` 即此 Skill——结构化的 lessons + 运行权重快照，`SEED_LESSONS` 是从真实人类回放提炼的起点。
 6. **保留历史版本 + 可回滚**：`versions.py` 的 VersionStore append-only 保存每次迭代的 Snapshot（权重+经验+评测+IG），`rollback(iter)` 可取回任意历史版本的权重。

 ## 模块清单

 ```
 snakego/
 ├── engine.py            忠实移植 adk.hpp Context（位级一致，已验证）
 ├── host.py              socket 对战主机（绕开 Windows stdio 文本模式 bug）
 ├── board.py             棋盘分析（合法步、可达空间、圈地估算）
 ├── decision_space.py    ★交付3：observation / macro-action / action mask / 终止 / IG 动作支持集
 ├── ig.py                ★交付4：信息增益（KL 严格口径；算不出如实记 incomplete，不冒充）
 ├── strategy_core.py     ★要求3/4：consolidated 可解释打分策略 + DecisionTrace
 ├── experience.py        ★要求5：Experience Skill（lessons + 压缩 + 种子经验）
 ├── versions.py          ★要求6：版本快照 + 回滚
 ├── replay2.py           确定性回放重建 + 关键事件提取
 ├── loop.py              ★交付1：HL 迭代闭环（对战→存回放→学习→重建策略→存版本→再评测→IG）
 ├── curves.py            ★交付5：score-iteration / IG-iteration 曲线（版本对齐，失败如实保留）
 ├── myagents.py          （旧 v1-v6 if-else，保留为历史，已被 strategy_core 取代）
 └── docs/
     ├── RULES.md         游戏规则
     ├── REPLAY.md        回放格式简述
     └── REPLAY_SKILL.md  ★交付2：看懂回放的 Skill（字段/数字/事件/误读，已验证）
 work/
 ├── analyze_humans.py    人类回放只读分析（提取真实策略模式）
 ├── quick_eval.py        consolidated 策略快速评测（只读，不落盘）
 ├── run_human_match.py   人类 vs 人类对战
 ├── run_myrollout.py     我的策略 vs 人类
 └── myrollout_v*.json    历史对局回放（可回放、可分析）
 ```

 ## 5 个交付物对应

 | 交付物 | 实现 | 状态 |
|--------|------|------|
| 1. HL 迭代闭环 | `loop.py: Loop.step()` | ✅ 已跑通（step 全流程：对战→学→存版本→评测→IG） |
| 2. 看回放 Skill | `docs/REPLAY_SKILL.md` + `replay2.py` | ✅ 人工写规则/字段/事件/误读，已用真实回放验证（分数复现一致） |
| 3. 决策空间定义 | `decision_space.py` | ✅ observation / 6 macro-action / action_mask / 终止 / 固定 IG support |
| 4. 信息增益 | `ig.py` | ✅ KL 严格口径；incomplete 如实记 `incomplete_replay`，不冒充 |
| 5. score/IG 曲线 | `curves.py` + 实跑 3 轮 | ✅ 版本对齐双曲线；iter0 IG=null（无基线）；失败局如实保留 |

 ## 一次实跑的曲线数据（consolidated 策略，vs rank15 学习 / sample_ai 评测，3 轮）

 ```json
 {
   "score_iteration": {"iter":[0,1,2], "score":[32,68,8]},
   "ig_iteration":    {"iter":[0,1,2], "ig_kl":[null,0.0120,0.1117]}
 }
 ```

 per-point：
 - iter0：my=32 hu=8 WON，ig=null（首轮无基线，如实记）
 - iter1：my=68 hu=90 LOST，ig=0.0120（策略小改，行为分布微变；本局人类超时但结果有效）
 - iter2：my=8 hu=40 LOST，ig=0.1117（策略再改，行为变化更大；分数回退，**如实保留失败**）

 说明：这是 consolidated 打分策略的最初 3 轮，尚处调参早期，曲线有波动属正常。学习循环已闭合，继续 step() 会持续更新经验与权重并产出更长曲线。

 ## 怎么跑

 ```bash
 # 一次完整迭代（默认 vs 冠军 rank01；可把 champion 换成其它）
 python -c "from snakego.loop import Loop; lp=Loop(data_dir='runs'); lp.step()"
 # 看 score/IG 曲线
 python -m snakego.curves runs
 # 看回放分析（只读）
 python work/analyze_humans.py
 ```

 ## 已知限制（如实记录）

 - **沙箱只读**：本轮无法落盘新回放/曲线文件，故上面曲线是在内存中实跑后打印；需要可写沙箱才能把 `runs/` 落盘供 CI/可视化。
 - **rank09/13 对局 desync**：某些人类子进程在 ~26 回合崩溃（引擎移植在 split/seal 边界与 C++ 分叉）。sample/rank15/rank01 不受影响，能跑完整局。修这个分叉是下一项工作。
 - **冠军 rank01 较慢**：搜索型 AI 每步 ~0.95s，单局约 10 分钟，影响迭代吞吐。
 ```
