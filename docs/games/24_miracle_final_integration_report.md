# 24_miracle 最终集成报告

## 一、基础设施接入过程（阶段1–9B 摘要）
1. **入口审计/契约/风险**：建立工作仓库 `Documents\AgentBenchFramework`（分支 `gongheng/24-miracle-adapter`）；源码级核验 Framework 9 项已知风险（全部 CONFIRMED）+ Results CI 无 schema 校验。
2. **适配层（绕开框架 bug，零共享修改）**：`result.py`（纯逻辑归一化/win_rate/h2h/realized_randomization）、`driver.py`（归一化胜负喂 Run，击败风险 #2/#3/#5）、`proctree.py`（跨平台精确 PID+create_time 进程树清理）、`match_runner.py`（三向交叉 result-json/trace/Replay + 时间线分类）、`entry.py`（main.exe 绝对路径入口适配）、`matrix.py`+`matrix_runner.py`（32 局编排）、`smoke_audit.py`（session 隔离/独立残留核验）。
3. **vendor 补丁**：`vendor/results_local/aggregate.py`（Windows 路径 `as_posix` 一行补丁，AST 等价上游）、`vendor/miracle_local/run_match.py`（跨平台进程清理+线程化超时读+`--result-json`，协议零改动）。
4. **框架最小修复（失败测试先行）**：`Run._write_toml` 转义反斜杠/引号（Windows 路径 config 致 run.toml 非法 TOML）；`pyproject.toml` 增 `miracle` extra。
5. **资产冻结**：Judge/sample/16 对手 archive/13 C++ 构建哈希全部冻结（`24_miracle_assets.json` + 协议 v0.3）。
6. **smoke（4 局，基础设施验证）**：sample-vs-sample + if-else-vs-sample，`BOTH_GROUPS_PASS`，三向一致。
7. **C++ 兼容适配（阶段9A/9B）**：13 C++ 策略隔离编译 13/13 COMPILE_PASS（rank16 经副本内建空 `build/`，源码/Makefile 哈希编译前后一致）；无 Judge 启动预检 `PROTOCOL_NOT_VALIDATED`。
8. **正式矩阵（A 方案 32 局）**：单后台 runner，32/32 COMPLETE，四向一致。

## 二、双版本测试最终数量
- **Python 3.13.5：142 passed / exit 0**。
- **Python 3.11.15（uv 托管）：142 passed / exit 0**。
- 证据 `docs/games/evidence/{py313,py311}_tests.txt`（完整未截断）。测试覆盖：纯逻辑(result/driver 36) + 进程树(9) + 依赖(2) + match_runner(20) + Results 管线(3) + runner(3) + matrix(19) + matrix_runner(13) + vendor 边界(3) + entry(12) + smoke_audit(12) + 其他。
- 红灯证据：`proctree_redlight.txt`、`match_runner_redlight.txt`、`matrix_redlight.txt`、`matrix_runner_redlight.txt`、`entry_redlight.txt`。

## 三、smoke 与正式矩阵的区别
| | smoke（4 局） | 正式矩阵（32 局） |
|---|---|---|
| 目的 | 基础设施验证 | 正式竞争力评测 |
| 对手 | sample AI（同源） | rank01–16 决赛策略 |
| 授权 | 默认允许（≤4 局） | 用户正式授权（A 方案）|
| 结论用途 | **不作竞争力结论** | if-else 战力评估 |
| session | `.smoke/sessions/20260721-172308_e026da` | `.smoke/matrix/20260722-001929_127b74` |

## 四、32 局逐局战绩
见 `24_miracle_final_battle_record.md`。**总：32 attempt / 20 valid / 12 invalid / 2W–18L / win_rate=10%（=2/20 valid）**。

## 五、12 局 invalid 准确归因
见 `24_miracle_final_evidence_audit.md`。全部 12 局异常在**对手 AI 方**（10 ai_timeout + 2 ai_crash），**0 if-else 异常**，无误分类，8s 与 v0.3 一致。Judge 经 player_error 判 if-else 胜但**未计分**。

## 六、Results 四向一致
events 重算 0.1 == summary 0.1 == aggregate 0.1 == web 0.1（`FOUR_WAY_CONSISTENT`）；schema `1 valid, 0 invalid`。

## 七、唯一 runner / 0 重跑 / 0 残留
- **唯一 runner**：单后台 `uv run --python 3.11` 进程（任务 `b9kiarq1v`），exit 0，无第二 runner。
- **0 重跑**：32 unique game_id，progress 与 events 均无重复。
- **0 残留**：矩阵完成后精确 cmdline 检查 0 个 Judge/AI/runner 进程。

## 八、限制与诚实结论
- **小样本**：32 局含 12 invalid；win_rate=0.1 仅基于 20 valid。
- **camp 不对称**：2 胜均在 camp0，camp1 全败 → 显著先手依赖。
- **Windows 运行限制**：5 个 C++ 对手（r01/r07/r08/r11/r15）每步 >8s 触发 ai_timeout（本机 8s/步预算下未能在 Judge 协议内响应）；r03 已知崩溃。这些 invalid 反映"协议未验证/Windows 运行限制"，**非 Agent 能力胜利**。
- **if-else 竞争力明显落后**：20 有效对局仅 2 胜（10%），分差重度落后（mean -13978）。
- **未修改策略/Judge/协议**：if-else `98199fae…`、Judge `104f77bf…`、16 对手 archive、13 C++ 构建哈希、协议 v0.3 `866696fd…` 均未改；协议状态字段保持 `PRE_REGISTERED_NOT_AUTHORIZED`（赛前未执行态），正式授权由 execution manifest 独立记录，完成状态由矩阵 progress/result 记录。
