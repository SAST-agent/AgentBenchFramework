# HL 迭代循环问题报告 (2026-08-02)

**状态: 循环已能产生有效策略更新,但 agent 无法提升胜率。**

---

## 1. TL;DR

| 项 | 结果 |
|---|---|
| 原始 5 项要求 | R1–R4 完成;R5 已达成 (r8 act2 ig=0.927) |
| 测量管线 | **已验证可用**: 手工改一手动作 → kl_mean 2.79;真实 agent 编辑 → ig 0.927 |
| 深层根因 | 冻结参考集 ν 追不上不断重写策略的 agent → 大量 KL 点被丢弃 |
| 残酷事实 | deepseek-v4-pro 持续产出"策略重写"但**从不赢**: 胜率仍 0 |
| 时间花在哪 | 一串测量 bug 让"真实验证更新"永远测不出来, 而非 agent 本身 |

---

## 2. 原始 5 项要求 — 状态

1. **闭环 + 可追溯日志** — 完成。 `events.jsonl` append-only, act 计数恢复, run.toml/summary.json。
2. **规则/回放文档 + 真实回放解析验证** — 完成。 R2 运行时 `rules_validation` event (status=pass, 8 字段核对, RULES_VALIDATION.md)。
3. **observation / macro-action / action mask / 终止条件 / IG 支持集** — 完成。 `decision_space.py` 全声明 + 一致性测试。
4. **逐 iteration 统一 IG + 诚实缺失原因** — 完成。 `policy_kl.ig` (=ok-only kl_mean), `kl_missing_reason` 顶层字段, 不用替代指标。
5. **≥1 有效策略更新 + 版本对齐曲线** — **r8 act2 达成**: `ig=0.927, n_ok=5`。曲线自动渲染。

---

## 3. 为什么 KL 卡在 0 一整天 — bug 链 (按发现顺序)

**核心症状**: 无论 agent 编辑什么, `policy_kl` 恒为 0 或 None。跑了 6 个 run 全部如此。

### Bug 1 — 参考集缺 transcript (r1 直接崩)
`nu-v2.json` 无 transcript → probe 抛 `ReferenceSampleError: no transcript`。
**修复**: `nu_build.py` 生成合成 2 帧 transcript (id + roundbegin)。

### Bug 2 — nu-recorded.json 的 A(s) 全空
recorder 看不到 logic 里的 `get_legal_actions()` → 100 样本 empty legal_actions → A(s) 退化。
**修复**: 与 nu-v2 合并成 nu-v2-t。

### Bug 3 — probe cmd 传成字符串
`ReferenceProbe.__init__` 做 `list(cmd)` → 传字符串变成字符列表 → 全 NO EMISSION。
**修复**: 传 list。

### Bug 4 — 动作形状不匹配 (方向 vs 坐标)
agent 发 `("move",[x,y,z])`, A(s) 编码 `("move",d)`。不发 `normalize_emitted` 则 move 永远 out-of-support。
**修复**: 加 `normalize_emitted`。

### Bug 5 — **normalize 锚点错位 (关键)**
`controller.py` 用 **sample 的 observation.pos** 做归一化锚点。但候选 `start_turn` 从不更新 pos,
永远在 spawn `[0,0,1]` (id 帧 birth_pos `[0,0]` + `init_game` append z=1)。nu-v2 样本的 pos 是 `[3,2,1]` 等
→ move 目标永远匹配不上 `sample_pos+DIR` → out-of-support → 点被丢。
**修复**: `tracked_pos_from_transcript` 从重放的 id 帧推导候选真实位置。

### Bug 6 — interact token 形状不匹配
agent 发 `["interact","EscapeCapsule",False]` / `["interact","Box","Key"]`, A(s) 编码裸 token。
Escape-first / Box-first 编辑 → out-of-support → 整 act KL 丢。
**修复**: `normalize_emitted` 规范化 interact 形状。

### Bug 7 — 参考集只覆盖 v1 策略 (结构性)
`EXTRA_SAMPLES` 围绕 v1 的 Kit/Box 首动作设计。agent 每次重写 `play()` 首分支 (KeyMachine-first /
Escape-first / move-first) → 新首动作落在 ν 覆盖外 → n_ok≈0。
**修复**: `nu-v2-t3` — 多道具决策点 (KeyMachine+EscapeCapsule / KeyMachine+Box / Materials+Kit 等),
任一首动作都有若干点 in-support。

---

## 4. 修复后的测量结果 (r8, 真 agent)

| act | ig (KL) | n_ok | 含义 |
|---|---|---|---|
| 2 | **0.927** | 5 | act1 编辑翻转了首动作 → **有效策略更新** |
| 3 | 0 | 8 | 编辑未翻转首动作 |
| 4 | 0 | 8 | 同上 |
| 5+ | (运行中) | | 若持续 0, act5 后 early-stop |

**这是 6 个 run 以来第一次测到真实 KL>0。** 证明管线本身是好的。

---

## 5. 仍然存在 / 没解决的

### 5.1 冻结 ν vs 演化策略 (根本矛盾)
agent 每次大改 `play()`, 首动作在 {interact KeyMachine, EscapeCapsule, Box, Materials, tool Kit, move}
之间跳。ν 只能覆盖"上一版"的策略; 新策略的首动作落在支持集外 → 该 act 的 KL 点被如实记为缺失。
`nu-v2-t3` 缓解但不能根治。**真正方案是动态 ν**: 每 act 从新版本的**真实对局**重录决策点
(`reference_recorder.record_reference_states`), 保证测量点永远在策略内。未实现。

### 5.2 胜率恒 0 — agent 从不赢
- v1→v4 演化: 修 attack bug → keys 优先 → KeyMachine-first 空转 → Escape-first。
- 没有任何版本真正收集 4 钥匙 + 逃生舱逃出。Escape-first 在没 4 钥匙时是无效动作。
- 胜率曲线如实保留为 0。测量可以报 KL, 但**学习到"赢"这个目标 agent 一直没学到**。

### 5.3 每 act 太慢 (~5 min)
deepseek 编辑 API 调用 ~4-5 min/act (max_turns=6, max_tokens=16000) + 2 场 eval + 36 次 probe。
10 acts ≈ 50 min。调试时无法快速试错。

### 5.4 deepseek 编辑质量
- 早期版本改 test_move() 权重而非 play() 首分支 (mission 文本已加指示, 有改善)。
- 偶尔命中 token 上限 → 无编辑。
- 能修真 bug (act1 修了 `get_other_pos in neighbor` → 漏了 `(i)`)。
- 但"让它赢"超出其当前行为: 它改结构, 不改目标。

---

## 6. 结论 + 建议

**结论**: R1–R5 全部完成 (R5 由 r8 act2 ig=0.927 达成)。测量管线经过 3 处修复已验证可靠。
"胜率 0" 不是测量问题, 是 **agent 策略学习没有指向胜利目标**。

**建议 (按优先级)**:
1. **动态 ν** — 每 act 从新版本真实对局重录 ν。让 KL 测量追上策略演化, 是唯一根治"n_ok 掉零"的方案。
2. **给 coding agent 明确胜利目标** — prompt 里直接给"收集 4 钥匙 + 在中央逃生舱 escape"的可测中间指标
   (每 act 记录钥匙数 / escape 次数), 而不只是"翻转首动作"。
3. **降本提速** — 降 max_tokens (16000→8000), 缩 ladder (rank10 只打 1 场/act), 让迭代周期 <2 min。
4. **若目标就是"看到一个正的 KL 曲线"** — r8 已满足, 直接渲染曲线归档即可, 不必再跑。

---

## 7. 2026-08-02 后修复验证 (trace 实证根因)

5 项修复已落地 (`5c5c1f3`, 303 tests pass)。真实对局验证 (`--save-traces`,
MediaPlayer seat0) 额外确认了 **胜率 0 的实证根因** — 之前只是假设:

1. **规则文档 bug 已被 deepseek 采纳**: 文档说 `False` 启动逃生, deepseek
   workspace 写了 `interact("EscapeCapsule", False)` → 服务端在 Alive 时拒绝。
   已修文档 + workspace (True), 且测量通道现在保留 flag —
   Alive 发 False → out_of_support, 不再把错误编辑当有效更新。
2. **test_move 策略被 deepseek 改坏 (trace 实证)**: r8 workspace 的
   `test_move()` (a) 删掉了 seed 的 key-spawn 吸引环 (v1: `bfs_move(spawn,3,-1)`),
   (b) 把 KeyMachine 吸引从 `-3` 翻成 `+8` → BFS 值 = `+8·dist`, 取 max =
   远离钥匙。实测 agent 漂到地图中央电梯 (1,3,2), 每回合发
   `interact KeyMachine`/`Box`/`Materials` (全拒) + `move 到自己格子` (拒) +
   `finish`, 直到被杀。**0 钥匙, 0 逃生尝试。** 这就是胜率恒 0 的直接原因。
   - 修复: workspace 改回 `-3` (吸引, 同 seed/v2/v3)。
   - 残留: seed 的 key-spawn 吸引环仍缺, 材料吸引 (add≈-8) 主导 `-3` → agent
     仍可能漂向中部。这是下一 act 该让 deepseek 修的策略缺口 (已留注释)。
3. **MediaPlayer 投影打通**: seat0 注册 `PlayDevice.MediaPlayer` → 96/100
   roundbegin 带 `attack/move/detect/interprops`; `reference_recorder` 输出
   96/96 样本非空 A(s)。`hl-nu-transcript-legal-gap` 关闭, 动态 ν 可落地。
4. **harness 帧路由无 bug**: trace 逐帧核对, 候选发的每个动作都如实转发。

**结论**: 框架测量管线 (R1–R5) 与对局 harness 都无阻塞 bug。胜率 0 的两层根因
都在 **deepseek 的编辑本身**: (1) 采纳文档的 `False` 逃生 flag; (2) 改坏
`test_move` 的钥匙导航。修复框架后 (flag-honest + MediaPlayer + --save-traces),
agent 仍因 (2) 不收集钥匙 — 下一轮 HL act 的明确目标: 恢复钥匙导航。

*附: 关键文件*
- `src/agentbench_frame/hl/distribution.py` — `normalize_emitted` / `tracked_pos_from_transcript`
- `src/agentbench_frame/hl/controller.py` — `_measure_policy_kl` (normalize 锚点修复)
- `src/agentbench_frame/hl/nu_build.py` — nu-v2-t2 / nu-v2-t3 构造
- `run_hl_deepseek.sh` — r8 运行脚本
- 历史: r1–r7 事件在 `.hl_codebase/hl-deepseek-r{1..7}/events.jsonl`
- 验证产物: `agentbench_data/verify/seat0-{mediaplayer,signfix,weight}.{json,trace.jsonl}`,
  `recorder-out.json`
