# 24_miracle 正式评测协议（草案）

> **状态：`PRE_REGISTERED_NOT_AUTHORIZED`**（用户已正式选择 A 方案）。结构已冻结但**未授权执行**；未创建正式矩阵目录/progress/空 results 文件，以免造成"已经开跑"的假象。仅当用户明确授权后才启动 32 局。
> 配套机器可读版：`24_miracle_evaluation_protocol.draft.json`（含 12 个 C++ 构建产物 SHA256）。

## 1. 冻结身份
- 被评 Agent：`miracle_ifelse`（高翔 `ifelse_bot/main.py`，SHA256 `98199fae…`，**只读不修改**）。
- Judge：受保护仓库 `judge_dev_logic`（`main.py` SHA256 `104f77bf…`，**不修改**）。
- 16 个对手：见 `24_miracle_roster_manifest.json`（每个 archive_sha256 已冻结）。

## 2. "完整矩阵"含义（已定：Plan A）
用户已正式选择 **A 方案**：if-else Agent 对 rank01–16，每对手交换阵营各 1 局，**总计划 attempt 严格 32**。
- 历史报告（高翔 consolidated lessons）研究目标"if-else bot vs 每位人类决赛选手"与此一致。
- B（16×16 round-robin）为独立更大任务，本轮不涉及。
- **rank16**：阶段9B 隔离构建成功（副本内建空 `build/`，源码/Makefile 哈希编译前后一致，未改策略；main.exe SHA256 `ed92b36b…`）；编译警告 `control reaches end of non-void function` 保留为运行风险（不修复/不隐瞒）。
- **rank03**：编译/启动成功，但保留历史运行时崩溃风险（invalid_now），即使崩溃也保留原始证据、不补跑、不算 Agent 有效胜利。

## 3. 随机性口径（非 seed）
Judge 用 `random.randint` 选 map_type/day_time、不读外部 seed。每局记录：
`requested_seed=null, effective_seed=null, deterministic_seed_supported=false, reproducible_from_seed=false, realized_randomization={map_type, day_time}`（**不表述为 seed**）。

## 4. 超时
- 单步 AI 操作超时：8s。
- 单局 wrapper 超时：120s（MAX_ROUND=100；if-else 局可更长，仍限内）。

## 5. 有效 / 无效定义
- **valid**：合法 end_info + result-json 与 trace 分数一致 + raw_winner 符合 Judge 规则 + end_info 前无 ai_error/ai_timeout + Judge 未在 end_info 前崩溃 + Replay 存在且合法 + cleanup 完成。
- **invalid**：AI crash / AI timeout / Judge crash / wrapper timeout / evidence_mismatch / replay_missing / replay_corrupt / result_json_missing / result_json_corrupt。
- **rank03**：当前对手程序崩溃，**无效证据**，不计入有效胜率。

## 6. 异常分类（按时间线，非仅 returncode）
- AI crash：end_info 前自然异常退出 或 trace `ai_error`。
- AI timeout：trace `ai_timeout` 或动作期限内未响应。
- **正常 cleanup 非零 returncode**：end_info 后 runner 主动终止 AI 的非零 returncode → **不判 AI 崩溃**。
- Judge crash：end_info 前异常退出 或 无合法 end_info。
- wrapper timeout：vendor+Judge+AI 树未在 wrapper 超时内结束。
- infra_failure：result-json 缺失/损坏、证据三向矛盾、Replay 损坏。

## 7. 计分
- raw_winner：`0 if s0>s1 else 1`。
- **同分**：score0==score1 时 Judge 判 player1 获胜（`judge_tiebreak_applied=true`），记 win/loss，**非 draw**。
- **win_rate = wins / valid_games**（分母仅 valid games）；`valid_games==0` → `win_rate=null`, `evaluation_status=NO_VALID_GAMES`（不报 0%）。
- draw：仅异常输入/未来协议防御；当前 Judge 不产生 draw。

## 8. 执行规则
- **成功局绝不重跑**。
- **失败恢复**：必须用新 game_id + `recovery_of=<原 game_id>`；本轮不自动恢复。
- **分批**：每批设停止门槛（残留进程/证据矛盾/winner 映射不一致/数据损坏 → 立即停，不进下一批）。
- **换边**：每对手双方各占 camp0/camp1。
- **PID+create_time 独立残留检查**：每局后按精确 PID+psutil create_time 核验 judge/ai0/ai1；PID 复用不杀；不按名称批量杀。
- **六处交叉验证**：Replay 头/哈希、trace、result-json、events、summary、网页可相互追溯；任一权威字段冲突 → `evidence_mismatch`。
- **原始失败证据永久保留**。
- **smoke ≠ 竞争力结论**：4 局 smoke 仅为基础设施验证。

## 9. 预计规模
- 时间：A 方案 32 局≈0.5–1.5h，160 局≈2.5–7h；B 方案 480 局≈4–12h，2400 局≈1–2 天（含 C++ 编译）。
- 存储：A 方案<50MB；B 方案可达 GB 级。
- 最坏：总时间≈总局数×wrapper_per_game_s。

## 10. 启动前阻塞项
1. `FORMAL_32_GAME_MATRIX_NOT_AUTHORIZED`——正式 32 局矩阵尚未授权。
（阶段9A/9B 已解决：matrix 含义=Plan A；main.exe 绝对路径入口适配完成；13 个 C++ 策略编译通过 + 冻结哈希；rank16 隔离构建成功。）

## 10b. 已知运行风险（非基础设施阻塞，不阻止启动）
- **rank03**：保持原策略；正式对局若在 end_info 前崩溃 → 记 invalid 并保留原始证据；不补跑；不计入 Agent 有效胜率（不进 win_rate 分母）。
- **rank16**：编译警告 `control reaches end of non-void function`，属运行风险，不修复/不隐瞒/不改源码。
- 多个 C++ 策略的 `-Wreturn-type` 等警告（非错误）仅记录；运行时异常按分类规则处理。

## 11. 未授权项
完整 16 人矩阵执行、16×16 round-robin、RL/Round81/hidden eval、修改 if-else 策略、push/PR/上传 Results/合并 main、删除或重跑 smoke 证据。
