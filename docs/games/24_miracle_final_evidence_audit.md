# 24_miracle 最终证据审计（12 局 invalid 逐局 + 四向一致性）

> 权威机器可读：`24_miracle_final_evidence_audit.json`（SHA256 `0e182ab67601c62d919a6eb245e7783b8dca7f6b9b9f467f48fe6fde26077aea`）。
> 只读审计，未重跑、未改证据。矩阵 session `.smoke/matrix/20260722-001929_127b74/`。

## 一、12 局 invalid 逐局归因

| game_id | rank | camp | error_type | 异常方 | anomaly 在 match_end 前 | Judge 原始胜负 | 计为能力胜？ | cleanup |
|---|---|---|---|---|---|---|---|---|
| m_rank01_camp0 | 01 | 0 | ai_timeout | OPPONENT(ai1) | 是 | p0=IFELSE（未计分）| 否 | True |
| m_rank01_camp1 | 01 | 1 | ai_timeout | OPPONENT(ai0) | 是 | p1=IFELSE（未计分）| 否 | True |
| m_rank03_camp0 | 03 | 0 | ai_crash | OPPONENT(ai1) | 是 | p0=IFELSE（未计分）| 否 | True |
| m_rank03_camp1 | 03 | 1 | ai_crash | OPPONENT(ai0) | 是 | p1=IFELSE（未计分）| 否 | True |
| m_rank07_camp0 | 07 | 0 | ai_timeout | OPPONENT(ai1) | 是 | p0=IFELSE（未计分）| 否 | True |
| m_rank07_camp1 | 07 | 1 | ai_timeout | OPPONENT(ai0) | 是 | p1=IFELSE（未计分）| 否 | True |
| m_rank08_camp0 | 08 | 0 | ai_timeout | OPPONENT(ai1) | 是 | p0=IFELSE（未计分）| 否 | True |
| m_rank08_camp1 | 08 | 1 | ai_timeout | OPPONENT(ai0) | 是 | p1=IFELSE（未计分）| 否 | True |
| m_rank11_camp0 | 11 | 0 | ai_timeout | OPPONENT(ai1) | 是 | p0=IFELSE（未计分）| 否 | True |
| m_rank11_camp1 | 11 | 1 | ai_timeout | OPPONENT(ai0) | 是 | p1=IFELSE（未计分）| 否 | True |
| m_rank15_camp0 | 15 | 0 | ai_timeout | OPPONENT(ai1) | 是 | p0=IFELSE（未计分）| 否 | True |
| m_rank15_camp1 | 15 | 1 | ai_timeout | OPPONENT(ai0) | 是 | p1=IFELSE（未计分）| 否 | True |

- **anomaly 方**：12/12 全为 **OPPONENT**（anomaly_player 随 camp 正确翻转：camp0→ai1，camp1→ai0）。
- **if-else 异常：0**（无 if-else 超时/崩溃）。
- **时序**：12/12 anomaly 事件在 trace 中出现在 `match_end` 之前（AI 先失败 → Judge 经 player_error 终止）。
- **Judge 原始胜负**：12/12 Judge 判 **if-else 胜**（对手 0 分），但依协议 `counted_as_capability_win=false` → **不计入有效 W/L**。
- **cleanup**：12/12 `cleanup_all_succeeded=true`（proctree 精确 PID+create_time 清理）。

### 必须确认项（全部成立）
- r01/r07/r08/r11/r15 的 10 次 timeout **确实来自对手 AI**（`run_match` 线程+queue 8s 超时读 AI stdout；AI 未在 8s 内响应）。✅
- r03 的 2 次 crash **确实来自对手**（AI 进程读异常/崩溃）。✅
- **没有 if-else 超时或崩溃**。✅
- **没有 Windows 管道/runner/Judge 错误被误分类为 AI 异常**（矩阵 `state=COMPLETE`，无 wrapper_timeout/Judge crash/result_json_missing/evidence_mismatch/replay 问题触发 HALT；分类源为 vendor run_match 的 AI 响应读取层）。✅
- **8 秒门槛与 v0.3 一致**（manifest `timeout=8.0`）。✅

## 二、关键文件 SHA256（每局证据，节选；全量见 JSON）
| game_id | result.json | replay | trace(.jsonl) |
|---|---|---|---|
| m_rank01_camp0 | 6516a13f… | 81947e57… | 6402a571… |
| m_rank03_camp0 | 1a0a7499… | bff72728… | e67ecd58… |
| m_rank15_camp1 | bd198049… | 3655e0d6… | efd0f860… |
（其余 9 invalid + 20 valid 局的 SHA256 在 `final_evidence_audit.json` per_game 字段。）

## 三、独立重算（从 events.jsonl 逐行）
- attempted=32，valid=20，invalid=12，wins=2，losses=18。
- **win_rate（valid）= 2/20 = 0.10**。
- camp0：valid=10, wins=2；camp1：valid=10, wins=0。

## 四、Results 四向一致性（GREEN）
- Framework schema：`1 valid, 0 invalid`。
- vendor aggregate：1 run → registry.toml；report_builder：1 run/1 game → site。
- **events 重算 0.1 == summary 0.1 == web 0.1（FOUR_WAY_CONSISTENT）**。
- 32 局 seed 字段 requested_seed/effective_seed=null 全部 OK。

## 五、断言（final_evidence_audit.json.assertions，全部 true）
- all_12_invalid_anomaly_on_opponent: **true**
- zero_ifelse_anomaly: **true**
- rank03_crash_is_opponent: **true**
- timeouts_are_opponent: **true**
- no_misclassification_no_infra_halt: **true**
- win_rate_is_2_of_20_not_2_of_32: **true**

## 六、完整性
- progress：32 done / 0 running / 32 unique game_id（**0 重跑**）。
- events：32 行 / 32 unique。
- 16 个 rank audit 文件齐；Run-compatible 输出（run.toml/summary.json/events.jsonl）落盘。
- 0 残留进程（精确 cmdline 检查）。
