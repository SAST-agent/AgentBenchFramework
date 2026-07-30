 # SnakeGo 迭代报告索引

 所有报告分三类，对应 `curves.json` 的数据分组。

 ## 对照组（权重调参，非 HL）— Policy Ablation

 这几轮只改权重数值（"seal more / split more"），没有结构性代码变化。
 IG ≈ 0 证实它们是"手动 RL"，不计入 HL 主线。保留作 Policy Ablation 对照。

 | 报告 | 对应版本 | 性质 |
 |------|----------|------|
 | `ITERATION_REPORT_1.md` | v1→v2→v3 | 纯调权重（survival gate / big seal / early split） |
 | `ITERATION_REPORT_2.md` | v4→v5→v6 | 纯调权重（U-turn / kill edge），研究"为何打不过 rank15" |
 | `ITERATION_REPORT_3.md` | 基础设施 | 修 bug + 跑通持久化循环（非策略迭代） |
 | `ITERATION_REPORT_4.md` | v7 | 回放分析根因，但落地仍是调权重 |

 经验数据：`runs/experience_control.json`（全部是 `weight_delta`）。

 ## HL 主线（结构化代码变化）

 每一轮都是真正的决策结构重写（相态机 / 新决策模块），不是权重微调。
 这才是被计入 HL 迭代曲线的主线。

 | 报告 | HL 迭代 | 版本 | 改动 | ratio | IG |
 |------|---------|------|------|-------|-----|
 | `ITERATION_REPORT_5.md` | HL-1 | v8 | GROW→SPLIT→HUNT_LOOP→SEAL 相态机 | 0.418 | 1.87 |
 | `ITERATION_REPORT_6.md` | HL-2 | v9 | space-max GROW + 生存回退 + 软敌方惩罚 | 0.408 | 0.34 |

 经验数据：`runs/experience.json`（结构化经验，`structural_change`）。

 ## 被拒 HL 尝试（诚实保留）

 结构化尝试但回退或无定论，按科研协议保留。

 | 报告 | 版本 | 结果 | 原因 |
 |------|------|------|------|
 | `ITERATION_REPORT_7.md` | v12 (a-e) | 0.397 | 5 个变体均未干净超过 v9；暴露评测 harness 两个阻塞点 |

 ## 曲线

 迭代图：`runs/iteration_chart.png`。x 轴 `HL-1 / HL-2` 为主线，
 对照组（灰色淡线）和被拒版本（红叉）单独呈现，互不混淆。
 数据源：`runs/curves.json`（`hl_main` / `control_weight_tuning` / `rejected_hl`）。
