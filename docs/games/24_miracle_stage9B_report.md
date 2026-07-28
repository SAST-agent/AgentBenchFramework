# 阶段9B 总结报告：rank16 隔离构建修复 + 最终矩阵就绪审计

> 本轮：仅 rank16 隔离构建修复（副本内建空 `build/`）+ 协议 v0.3 收尾 + 最终审计。
> **未重新编译 rank16（沿用唯一成功 session）、未重跑启动预检、未启动任何真实对局/矩阵/smoke；未 push/PR。**
> 正式 32 局矩阵**仍未授权**。

## 1. rank16 隔离构建（唯一成功 session）
- session：`.smoke/rank16build/20260721-195738_baff71`（唯一；未重跑）。
- 操作：从受保护源复制 → 仅在**副本**内创建空 `build/`（Makefile 假设其存在）→ 用原 Makefile 编译一次。**未改源码/Makefile/编译参数/优化**。
- 源码+Makefile 哈希编译前后一致：`source_integrity_ok=True`（13 个源文件哈希 before==after）。
- 新增文件仅构建产物：`build/{ai-sample,ai_client,calculator,gameunit}.o` + `main.exe`（空 build/ 为构建环境准备）。
- 产物：`main.exe`，PE32，SHA256 `ed92b36ba411cd52c1408762bbb6cedfc54a7a62ecb16f7d2d98cb95458372db`，依赖 libgcc/libstdc++/KERNEL32/msvcrt。
- `rank16_report.json` SHA256 `4e7e80f7105338ddcfb38733b69f2cffdcef855a38a6bbd82b214251ce97effd`。
- **rank16 compiler warning**：`control reaches end of non-void function`（ai-sample.cpp 等）→ 属运行风险，**不修复/不隐瞒/不改源码**（见 protocol `known_runtime_risks`）。
- 此前 rank16 失败证据（9A `.smoke/precheck/20260721-184652_da899d/`）**保留未删**。

## 2. rank16 无对局启动预检
沿用 9A session 已记录结果（本轮未重跑）：
- `COMPILE_PASS` ✅、`OS_SPAWN_PASS` ✅（rc=3，AI 读 stdin EOF 退出，符合"无 Judge"）、`COMMAND_RESOLUTION_PASS` ✅（绝对路径命令解析）、cleanup_succeeded ✅。
- **`PROTOCOL_NOT_VALIDATED`**：无 Judge 启动，不证明协议/比赛可用。
- **未进行真实比赛**；编译/启动成功 ≠ 竞争力结论。

## 3. 协议版本化（不静默覆盖 9A）
- **保留 v0.2**（rank16=null 历史）：`24_miracle_evaluation_protocol.v0.2.json` SHA256 `e484ca8f2b30519a8e1708e2d94086322c845620d0dc056ac5e06ce575e91424`；`.md` `f552ae50…`。
- **新建 v0.3**：`v0.3.json` `866696fd9e094da85e3f2c04dc5ba20d0500faf8461531a242362323c4efe0b3`；`.md` `7658e310064eda5b41029d438ada34786358287dd0a0175ee46b69924c7ee282`。
- 变更清单：`24_miracle_protocol_changelog_v0.2_to_v0.3.json` SHA256 `9758118490af79381c1f7b6817c397ac48e9f7f66503d8d20821caef4cb3e77b`。
- 变更**仅含**：用户选 A 方案 32 局、main.exe 入口适配完成、rank16 隔离构建成功 + 哈希补入、已解决阻塞转完成、rank03/rank16-warning/general-warning 转 `known_runtime_risks`。其余字段不变。两 JSON 均经 Python `encoding=utf-8` 解析通过。
- v0.3 内容：`status=PRE_REGISTERED_NOT_AUTHORIZED`；`matrix_meaning.total_attempts=32, games_per_opponent=2, camps=[0,1]`；`blockers_before_matrix=["FORMAL_32_GAME_MATRIX_NOT_AUTHORIZED"]`；`known_runtime_risks`=[rank03_crash, rank16_compiler_warning, general_cpp_warnings]；13 C++ 构建哈希 ALL_PRESENT。
- **rank03**：保留在 32 attempt 中；正式对局若 end_info 前崩溃→invalid+保留证据+不补跑+不计 Agent 有效胜率（不进分母）。
- **未创建正式矩阵目录/progress/results/空结果文件**。

## 4. 双版本完整回归 + 最终审计
- Python 3.13.5：**110 passed / exit 0**；Python 3.11.15（uv 托管）：**110 passed / exit 0**。完整证据 `docs/games/evidence/{py313,py311}_tests.txt`（未截断）。
- 残留进程：**0**（Judge/AI/runner/编译）。
- 13/13 C++ COMPILE_PASS（9A=12 + rank16=1）；13 个 C++ 产物均有冻结哈希（v0.3 build_artifacts）。
- rank04/05/07 Python 入口静态确认（main.py，entry 绝对路径解析）。
- 原 4 局 smoke summary 哈希 `1d984655…` **未变**；无新 Replay/events/match-summary。
- rank16 唯一成功 session（无第二次构建）；受保护仓库 11 处未触碰；git 暂存区 **0**（未 add/commit/push）。

## 5. 429 说明
本轮协议文档收尾期间发生过一次工具 429（`glm-5.2 temporarily unavailable`）——发生在**协议 JSON 编辑阶段**，**不是** rank16 构建或游戏失败，也未启动任何对局；恢复后从磁盘已落盘的合法 draft 继续，未回滚、未重编译、未重跑预检。阶段9A 失败证据与 9B 成功证据均完整保留。

## 6. 是否已不存在正式矩阵执行阻塞
**基础设施阻塞全部解除**：matrix 含义(Plan A)、main.exe 入口、13 C++ 编译+冻结哈希、rank16 构建均已就绪。
**唯一剩余项**：`FORMAL_32_GAME_MATRIX_NOT_AUTHORIZED`（正式 32 局矩阵尚未授权）。
**已知运行风险（非阻塞，不阻止启动）**：rank03 崩溃、rank16/general 编译警告——正式对局按 invalid 分类处理并保留证据。

## 7. 禁止事项（均未违反）
未启动 Judge/真实对局；未重编译 rank16/重跑预检；未跑 32 局矩阵/额外 smoke；未改 if-else/人类策略源码/Makefile；未跑 RL/Round81/hidden；未删/重跑 smoke；未 git add/commit/push/PR/上传；未静默覆盖 v0.2。

## 证据位置
- rank16 构建：`.smoke/rank16build/20260721-195738_baff71/`（rank16_report.json/full.log/manifest/rank16_copy/main.exe+源+build/）。
- 9A 编译：`.smoke/precheck/20260721-184652_da899d/`（12 策略 + rank16 历史失败）。
- 协议：`docs/games/24_miracle_evaluation_protocol.{v0.2,v0.3,draft}.{json,md}` + changelog。
- 测试：`docs/games/evidence/{py313,py311}_tests.txt`。
