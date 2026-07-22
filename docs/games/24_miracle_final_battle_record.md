# 24_miracle 正式矩阵战绩记录（A 方案 32 局）

> 权威数据：`24_miracle_final_evidence_audit.json`。矩阵 session：`.smoke/matrix/20260722-001929_127b74/`。
> 单后台 runner（Python 3.11.15，单 `uv run` 进程）`state=COMPLETE`，elapsed 244.5s，exit 0。

## 总战绩
- **总 attempt：32**（rank01–16，每对手 camp0/camp1 各 1）。
- **valid：20｜invalid：12**。
- **有效 W–L：2–18**。
- **正式胜率（win_rate）= 有效胜局 / 有效对局 = 2 / 20 = 10%**。

### 胜率口径（重要，不得误用）
- **胜率 = 2/20 = 10%**（分母为 valid games）。
- `2/32 = 6.25%` 是"有效胜局占全部 attempt 的比例"，**不是胜率**，仅在统计脚注中提及。
- **invalid 既不是 win 也不是 loss**，不进 W/L、不进胜率分母。
- **禁止写成"2 胜 30 负"**（12 invalid 不是 12 负）。正确表述："2 胜 18 负（20 有效对局），另有 12 局 invalid"。

## 逐 rank 战绩
| rank | valid/inv | W-L | win_rate | 备注 |
|---|---|---|---|---|
| r01 | 0/2 | 0-0 | null | 对手 ai_timeout×2（>8s/步）|
| r02 | 2/0 | 0-2 | 0.00 | |
| r03 | 0/2 | 0-0 | null | 对手 ai_crash×2（已知运行时崩溃风险）|
| r04 | 2/0 | 1-1 | 0.50 | if-else 取 1 胜 |
| r05 | 2/0 | 0-2 | 0.00 | |
| r06 | 2/0 | 0-2 | 0.00 | |
| r07 | 0/2 | 0-0 | null | 对手 ai_timeout×2 |
| r08 | 0/2 | 0-0 | null | 对手 ai_timeout×2 |
| r09 | 2/0 | 1-1 | 0.50 | if-else 取 1 胜 |
| r10 | 2/0 | 0-2 | 0.00 | |
| r11 | 0/2 | 0-0 | null | 对手 ai_timeout×2 |
| r12 | 2/0 | 0-2 | 0.00 | |
| r13 | 2/0 | 0-2 | 0.00 | |
| r14 | 2/0 | 0-2 | 0.00 | |
| r15 | 0/2 | 0-0 | null | 对手 ai_timeout×2 |
| r16 | 2/0 | 0-2 | 0.00 | |
| **合计** | **20/12** | **2-18** | **0.10** | |

## camp 拆分
- **camp0**（if-else 先手）：valid=10，wins=2，win_rate=0.20。
- **camp1**（if-else 后手）：valid=10，wins=0，win_rate=0.00。
- 结论：if-else 两胜均在 camp0；**存在明显先手依赖**。

## 其他统计（20 valid）
- steps：total=18144，mean≈907.2，median≈785.5。
- 分差（if-else 视角）：mean≈-13978.2，median≈-20996.0（重度落后）。
- map_type 分布：{0:10, 1:10}；day_time 分布：{0:12, 1:8}。
- invalid 原因分布：ai_timeout×10、ai_crash×2。

## 12 局 invalid 归因（详见 final_evidence_audit.md）
全部 12 局的 AI 异常（timeout/crash）均发生在**对手 AI 方**，**无一发生在 if-else**。Judge 经 player_error 路径判 if-else 胜，但依协议**不计入有效 W/L**（对手崩溃/超时不算 Agent 能力胜利）。
