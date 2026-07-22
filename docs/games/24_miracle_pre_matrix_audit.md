# 24_miracle 矩阵前审计（Pre-Matrix Audit）

> 本轮：正式矩阵前审计 + 评测协议预注册。**未启动任何新游戏/矩阵/RL/push/PR。**
> 状态分级：【已实际验证】/【仅静态确认】/【尚未验证】/【需用户决定】/【阻塞项】。

## 1. 恢复与冻结检查【已实际验证】
- 分支 `gongheng/24-miracle-adapter`；工作树仅本轮新增（未跟踪）+ `M pyproject.toml` + `M src/agentbench_frame/tracking/run.py`；**未 commit/push**。
- 残留 match 进程：**0**（非审计自身）。
- smoke session `20260721-172308_e026da`：2 run 目录 / 4 game 标签，run.toml+summary.json+events.jsonl+trace+replay+result.json 哈希已冻结（只读未改写）。g1 两局 Replay 哈希相同（同源确定性 AI 在相同随机地图上的可重现结果，符合预期）。
- 适配器/vendor/Judge/sample/if-else SHA256 已记录（见 `24_miracle_assets.json` 与会话证据）。
- 受保护仓库仍 11 处未提交（既有 DOTO，**未触碰**）；高翔工程只读。

## 2. 公共框架修改审计【已实际验证】

### 2.1 `Run._write_toml` 转义修复（tracking/run.py）
- 缺陷：写 `[config]` 字符串值时不转义反斜杠/引号 → Windows 路径使 run.toml 非法 TOML。
- 修复：增 `q()` 转义（`\`→`\\`、`"`→`\"`）用于所有字符串值；`encoding="utf-8"`。
- **回归测试**（`test_driver.py`，5 项，全绿）：Windows 路径、双引号、反斜杠、混合类型（str/bool/int/float）、普通串不过度转义、`[run]` 段回环。
- **对其他游戏影响**：`Run` 是所有游戏共用的基类设施；转义对正常字符串零改动（无反斜杠/引号时输出逐字节不变），仅修复含特殊字符的值 → **向后兼容，不破坏其他游戏**。

### 2.2 `pyproject.toml`
- 新增 `miracle = ["psutil"]` extra（`tracking` 已有 psutil；`miracle` 为显式声明）。缺失 psutil 时 `proctree` 给可操作错误（`test_dependencies.py` 覆盖）。

### 2.3 全仓库双版本测试【已实际验证】
- Python 3.13.5：**98 passed / exit 0**；Python 3.11.15（uv 托管）：**98 passed / exit 0**。
- 全仓库测试 = `tests/miracle/`（Framework 原无测试，风险#9 不变）。
- 证据（完整未截断 UTF-8）：`docs/games/evidence/{py_interpreters.txt, py313_tests.txt, py311_tests.txt}`。

### 2.4 vendor 最小 diff【已实际验证】
- `vendor/results_local/aggregate.py` vs 上游：**仅 header 注释 + 1 行**（`str(...)` → `.as_posix()`），AST 等价（`test_vendor_boundary.py`）。证据 `evidence/vendor_aggregate_diff.txt`。
- `vendor/miracle_local/run_match.py` vs 上游高翔：5 处功能性补丁（JUDGE_DIR env、proctree 跨平台清理、spawn flags、线程化超时读、`--result-json`）+ 结构重排；摘要见文件头。证据 `evidence/vendor_run_match_diff.txt`。

### 2.5 提交范围（不擅自 stage）
- 见 `24_miracle_submission_scope.md`。`.smoke/`、生成网页、临时数据、资产副本**不应**提交。

## 3. 16 策略运行时兼容性【仅静态确认】
见 `24_miracle_roster_manifest.json`。摘要：
- **Python（rank04/05/07）**：Windows 原生可跑（main.py），可按现 runner 协议启动。
- **C++ 源码（13 个：rank01/02/03/06/08–16）**：可移植 makefile（`g++ -std=c++11`），本机有 g++15.2/Make4.4，但**无预编译二进制**；MinGW 编译产出 `main.exe`，而 `resolve_ai_command` 仅识别 `./main` → **当前协议下无法启动**（需显式 `--cmd main.exe` / 适配层补 main.exe 探测 / WSL-Linux）。**均未编译未运行**（verification_status=static_only）。
- **rank03**：SKILL.md 记载当前崩溃，无效证据。

## 4. "完整矩阵"含义【需用户决定】
NEEDS_USER_DECISION（见协议草案 §2）。最可能为 A（if-else vs 16 换边），但材料未唯一确定；B（16×16 round-robin）为独立更大规模任务。**不自行等同，不启动。**

## 5. Results 兼容性预审（仅用 4 局 smoke）【已实际验证】
- Framework schema check：`2 valid, 0 invalid`。
- vendor aggregate：2 runs → registry.toml；report_builder：2 runs/1 game → site/data.json。
- 三向一致：sampleA `summary 0.5 == web 0.5 == events 重算 0.5`；miracle_ifelse `1.0==1.0==1.0`；seed 字段 OK。
- 输出与 AgentBenchResults CI 目录契约对齐：`runs/<game>/<agent>/<run_id>/{run.toml,summary.json,events.jsonl}` + `registry.toml`。
- 上传所需文件/目标目录见 `24_miracle_submission_scope.md`（**未复制到 Results 仓库、未 push、未上传**）。
- Windows vendor 补丁边界：仅路径分隔符；Linux CI 上游不受影响（`test_vendor_boundary.py` AST 等价 + 上游 bug 证据测试）。

## 6. 阻塞项【阻塞项】
1. matrix_meaning 未定（NEEDS_USER_DECISION）。
2. 13 个 C++ 策略的 Windows 启动缺口（main.exe 探测 / --cmd / WSL）。
3. rank03 无效（崩溃）。
4. 正式矩阵需用户明确授权。

## 7. 是否具备申请正式矩阵授权的条件
**基础设施侧：具备**（适配层 + 98 项双版本测试 + 4 局 smoke + 三向验收 + 无残留 + 资产冻结 + 协议草案）。
**执行侧：尚不具备**——需先解决 §6 阻塞项（矩阵定义 + C++ 启动缺口 + 授权）。本轮**未启动**矩阵。
