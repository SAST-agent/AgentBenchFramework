# 阶段9A 总结报告：C++ 兼容适配、隔离编译、无对局启动预检

> 本轮：兼容适配 + 编译预检。**未启动 Judge / 未运行任何真实对局 / 未跑矩阵 / 未 push/PR**。
> 用户已正式选择 **A 方案**（if-else 对 rank01–16，每对手换阵营各 1 局，共 32 局，未来另行授权）。

## 1. 入口审计【已实际验证】
分支 `gongheng/24-miracle-adapter`；0 暂存；0 残留 match 进程；smoke 4 局完整（summary 哈希 `1d984655…` 未变）；16 archive 哈希 ALL_MATCH；受保护仓库 11 处未触碰。

## 2. main.exe 入口适配（TDD）【已实际验证】
- **根因**：Windows `CreateProcess` 对 `Popen(["main.exe"], cwd=dir)` **不搜索 cwd 参数目录** → `FileNotFoundError`。原 `resolve_ai_command` 返回裸名 → 13 个 C++ 策略在 Windows 无法启动。
- **修复**：新增 `src/agentbench_frame/games/miracle/entry.py::resolve_ai_command`，返回**绝对路径**的入口；跨平台（Win: main.exe>main.py>main；POSIX: main>main.py>main.exe）；explicit 优先；路径含空格保持单元素（不拆）；main.exe+main 共存确定性（Win 选 main.exe，POSIX 选 main）；缺入口明确报错；不改策略源码。vendor `run_match.py` 委托给它。
- **测试**：`tests/miracle/test_entry.py` 红灯→绿灯，**12 项**（Win main.exe / POSIX main / main.py / explicit 优先 / 空格不拆 / 共存 / 缺入口 / 不改目录 / 返回绝对路径）。`entry_redlight.txt` 存证。

## 3. 13 个 C++ 策略隔离编译【已实际验证】
脚本 `tools/miracle_precheck.py`：复制受保护源到唯一全新 session（不动原策略；session 已存在则拒绝；不删历史）→ 用策略自带 makefile 原样编译（不改逻辑/优化）→ 记录源哈希/命令/完整 stdout+stderr+exit/产物类型+大小+SHA256/PE/DLL 依赖（objdump 静态）。逐策略独立，失败不掩盖。

| rank | 策略 | COMPILE | main.exe SHA256（前16） | 备注 |
|---|---|---|---|---|
| 01 | Maiev | PASS | b451d4f99b694c4a | PE32, deps libgcc/libstdc++/KERNEL32/msvcrt |
| 02 | 碧海潮生曲 | PASS | 5c731a794eb8282f | 同上 |
| 03 | 深浅值藏的第六分块 | PASS | 84cdb1344104060e | **invalid_now**（运行时崩溃风险保留）|
| 06 | Mooncell | PASS | b10de17f886ca930 | |
| 08 | yyy | PASS | 249b9c6f7bcea4b3 | |
| 09 | タチコマ | PASS | 3c7c822cec8cf613 | |
| 10 | AK | PASS | ae0c24618b1fdc3e | |
| 11 | Halcyon | PASS | 6c8c169c1f23dc87 | |
| 12 | 星之梦 | PASS | e92fe57054968b7b | |
| 13 | 啥都没改 | PASS | d2e8ff65eaed3c97 | Archer lattice 断点 |
| 14 | sample | PASS | bb0f1b171251edc3 | |
| 15 | a_idoit | PASS | c570d1807402b9fc | |
| 16 | 起床大失败 | **FAIL** | — | makefile 需 `./build/`，解包缺失；未改其 makefile（不绕过）|

**12/13 COMPILE_PASS，1 FAIL（rank16）**。所有产物为 PE32，依赖 MinGW 运行时 DLL（libgcc_s_seh-1.dll / libstdc++-6.dll，位于 mingw/bin，本机 PATH 可用）。

## 4. 无对局启动预检【已实际验证，PROTOCOL_NOT_VALIDATED】
对每个编译产物：`entry.resolve_ai_command` 构造命令（与正式 runner 一致）→ 短时 `Popen`（stdin=DEVNULL，**不接 Judge**）→ PID+create_time 精确清理。结果分类：

| 分类 | 12 个编译成功策略 |
|---|---|
| COMPILE_PASS | ✅ |
| OS_SPAWN_PASS | ✅（rc=3：AI 读 stdin EOF 后退出，符合"无 Judge"预期）|
| COMMAND_RESOLUTION_PASS | ✅（绝对路径命令解析成功）|
| PROTOCOL_NOT_VALIDATED | ✅（**未启动 Judge**，不证明协议/比赛可用）|

- rank16：COMPILE_FAIL → 未进入启动预检。
- Python 策略 rank04/05/07：静态复核入口（main.py 在）+ 命令解析（entry 返回 [python, abs main.py]），不启动比赛。
- **进程清理**：每局 spawn 后 `proctree.cleanup_all` 精确 PID+create_time 清理，cleanup_succeeded=True；全局残留检查 **0** Judge/AI 进程。
- **诚实声明**：编译/进程创建成功 ≠ 策略已成功完成比赛；无 Judge 启动，AI 协议/比赛可用性**未验证**。

## 5. 双版本完整回归 + 完整性【已实际验证】
- Python 3.13.5：**110 passed / exit 0**；Python 3.11.15（uv 托管）：**110 passed / exit 0**。证据 `docs/games/evidence/{py313,py311}_tests.txt`（完整未截断）。
- 残留进程 0；smoke 4 局 summary 哈希未变（`1d984655…`）；precheck 未生成任何比赛 replay/events/match-summary（仅预检汇总）。
- 既有 Miracle 测试与其他游戏：entry 改动为新增模块 + vendor 委托，`run.py`/`pyproject` 仅本轮既有最小修复，未破坏（110 passed 含既有 98）。

## 6. 32 局协议（PRE_REGISTERED_NOT_AUTHORIZED）
- `24_miracle_evaluation_protocol.draft.{json,md}` 状态升级为 **PRE_REGISTERED_NOT_AUTHORIZED**，Plan A 冻结：if-else（SHA `98199fae…`）对 rank01–16，每对手 camp0+camp1 各 1 局，**总 attempt 严格 32**。
- 12 个 C++ 构建产物 SHA256 入 `frozen_identities.build_artifacts_win64_mingw`；rank16=null（COMPILE_FAIL）；rank03 标 invalid_now。
- seed 口径：requested/effective=null、reproducible=false、仅 realized_randomization；同分 player1 胜；win_rate 分母仅 valid；成功局不重跑；恢复需新 game_id+recovery_of+另行授权。
- **未创建正式矩阵结果目录/progress/空 results 文件**。

## 7. 阻塞项
1. **rank16 COMPILE_FAIL**（makefile 需 `./build/`）→ NEEDS_USER_DECISION：`mkdir build` 是否算可接受构建环境准备（不改策略源/makefile 语义）；否则 rank16 的 2 局无法在 Windows 运行。
2. rank03 invalid_now（崩溃），其 2 局即使崩溃也保留原始证据、不补跑、不计 Agent 有效胜利。
3. 协议 PROTOCOL_NOT_VALIDATED：无 Judge 对局未验证 AI 协议/比赛可用性（需正式矩阵授权后真跑才验证）。
4. 正式 32 局需用户**单独授权**。

## 8. 是否具备申请正式矩阵授权的条件
**基础设施 + 兼容适配侧：基本具备**（main.exe 入口适配 + 12/13 编译 + 启动预检 + 110 双版本测试 + 协议冻结）。
**执行侧：尚有 2 个阻塞**（rank16 编译、rank03 无效），且正式 32 局须单独授权。本轮**未启动**矩阵。

## 9. 本轮禁止事项（均未违反）
未启动 Judge/真实对局；未跑 32 局矩阵/smoke/补充对局；未改 if-else/人类策略；未跑 RL/Round81/hidden；未删/重跑 smoke；未 git add/commit/push/PR/上传；未把编译/启动成功写成竞争力结论。

## 证据位置
- 预检全量证据：`.smoke/precheck/20260721-184652_da899d/`（manifest/precheck.full.log/reports/rankNN.json/summary.json/strategies/<rank>/main.exe+源）。
- 历史失败迭代 session 一并保留：`20260721-183916_f0266d`（os_spawn 裸名 bug）、`20260721-184610_f052ac`（spawn_check 下标 bug）。
- 测试证据：`docs/games/evidence/{py313,py311,entry_redlight,...}.txt`。
