---
name: miracle-harness
description: Miracle（24 届）迭代闭环标准操作流程：发起对战/看回放/改策略/存版本/迭代 SOP，供外部 LLM/Agent 使用
---

# Miracle harness 使用 SOP（面向外部 LLM/Agent）

在 `AgentBenchFramework` 仓库内操作 Miracle（24 届）迭代闭环的标准流程。
所有命令在仓库根目录执行；产物统一落在 `agentbench_data/`（gitignore，但可追溯）。

## 0. 模块速查（`src/agentbench_frame/miracle/`）

| 模块 | 职责 |
|---|---|
| `match.py` | `run_match(agent_a, agent_b, seed, tag)` → `MatchResult`（官方逻辑子进程对战，落 `.mrc` + `.trace.jsonl`） |
| `agent_bridge.py` | `MiracleAgent` 接口；`EndRoundAgent`/`SampleAgent`/`SampleV2Agent` |
| `decision_space.py` | obs schema、`action_mask`、终止条件、动作支持集（官方仲裁口径） |
| `ig.py` | 严格 KL(new‖old) 逐决策点→episode→iteration；缺失如实记录 |
| `replay.py` | `.mrc` 二进制 → 事件时间线（`parse_replay`/`summarize`） |
| `iterate.py` | 评测→IG→导出→曲线 管线（`demo`/`build_curves`） |
| `versions.py` / `loop.py` | 版本快照 / 迭代编排（events.jsonl） |
| `cli.py` | `python -m agentbench_frame.miracle.cli <子命令>` |

## 1. 发起对战（要求 1）

```bash
python -m agentbench_frame.miracle.cli match --agent sample --seed 11
# 或代码内：run_match(SampleAgent(), EndRoundAgent(), seed=11, tag="...")
```

- 结果：`agentbench_data/replays/24_miracle/match_<tag>_<ts>_seed<N>.mrc` + `.trace.jsonl`
- `MatchResult`：`winner`、`scores`（30000=胜负标记，非真实分）、`rounds`、`terminated_by`

## 2. 看回放（要求 2）

先调用 inline skill `miracle-replay-reader` 了解字段含义，再：

```bash
python -m agentbench_frame.miracle.cli replay --path <xxx.mrc> [--jsonl out.jsonl]
```

- 二进制 `.mrc` 用 `replay.py` 解析为事件时间线；JSON 逐帧用 `.trace.jsonl`。
- 决策点 obs 提取：trace 中 content 含 `"map"` 的 from_logic 帧（`state` 字段是 round 号，
  `content[0]` 前 6 位是长度前缀）。

## 3. 改策略（要求 1 后半）

1. 在 `agent_bridge.py` 新增 `MiracleAgent` 子类（可继承 `SampleAgent` 覆盖 `summon_order` 等）；
2. 在 `iterate.AGENTS` 注册 `{"名字": 类}`（注册即"保存新版本"）；
3. 用 `cli evaluate --agent <名字> --iteration N` 单独评测，或直接进迭代。

## 4. 存版本（版本对齐）

```python
from agentbench_frame.miracle.versions import snapshot_version
snapshot_version("sample_v2", iteration=1, seed=11, match_summary={...})
```
落 `agentbench_data/versions/24_miracle/iter<N>_<agent>/`（源码副本 + meta.json：
git commit + 时间戳 + 评测摘要）。

## 5. 迭代 SOP（要求 5：score–iteration / IG–iteration 曲线）

```bash
python -m agentbench_frame.miracle.cli iterate --agents sample,sample_v2 --seed 11
python -m agentbench_frame.miracle.cli curves --agent sample [--json curves.json]
```

1. 跑 `iterate`：对每个版本真实评测 + 版本快照 + IG（新‖旧，ε-soft 分布口径）
   + 导出 `agentbench_data/runs/24_miracle/<族>/<iterN_ver_seedN>/`（AgentBenchResults
   契约：`run.toml` + `summary.json` + `ig.json`）；
2. 全程事件在 `agentbench_data/events/24_miracle/iterations.jsonl`（evaluate/snapshot/ig
   逐条可追溯）；
3. `curves` 子命令输出版本对齐的 score/IG ASCII 表；JSON 可接入 AgentBenchResults 的
   `compare.toml` 可视化。

## 6. 数据口径红线（务必遵守）

- score 30000 = 神迹被毁/全灭的胜负标记，不是真实得分；
- IG 用**严格 KL(new‖old)**；支撑集不相交/状态不可比 → 记 `missing_reason`，
  **不得用其他指标冒充**；
- 失败、缺失、不完整评测如实保留（如 v2 输 0 分、IG=1.9474 如实上曲线）。
