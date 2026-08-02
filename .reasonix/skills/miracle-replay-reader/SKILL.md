---
name: miracle-replay-reader
description: 读取 Miracle（24 届）对战回放：trace.jsonl 逐帧格式、obs 字段数字含义、关键事件、常见误读清单与解析工作流
---

# Agent 看游戏回放：Miracle（24 届）

读取并解读 Miracle 对战回放。回放由 `run_match()`（`agentbench_frame/miracle/match.py`）落盘在 `agentbench_data/replays/24_miracle/`，一份对局产生两个文件：

| 文件 | 格式 | 内容 |
|---|---|---|
| `*.mrc` | **二进制**（官方 magic replay，非 JSON） | 官方格式完整对局，给官方客户端/裁判用 |
| `*.mrc.trace.jsonl` | JSONL 逐帧 | 本框架的**可读 trace**，逐帧记录 host 与官方逻辑的往返消息（含完整 obs 与操作） |

> 常见误读①：`.mrc` 不是 JSON，别用 `json.load` 读；分析一律用 `.trace.jsonl`。

## 1. 游戏规则速览（24 届 Miracle）

- 六边形立方坐标地图（`x+y+z=0`），每方一座神迹（camp0 `(-7,7,0)`、camp1 `(7,-7,0)`，HP 30）。
- 选卡：开局各选 1 神器 + 3 生物卡组（样例 `HolyLight` + `Archer/Swordsman/BlackBat`）。
- 回合制，`MAX_ROUND=100`；每回合 mana +1（上限 12），回合开始重置 `can_move/can_atk`。
- 生物：召唤（需 mana、容量、召唤点）→ 移动（≤max_move）→ 攻击（射程内）→ `endround`。
- 地图上有 4 个固定驻扎点（Barrack），占领后该方多 3 个召唤点。
- 固定障碍：`Abyss`（39 格，地面单位不可过、飞行可过）、2 个 Miracle 障碍、地图边界 `MAPBORDER`（66 格）。**这些不在 obs 里**，见 `decision_space.py` 常量。

## 2. 胜负与计分（终局帧 state=3 的 `content`）

官方终局消息形如：`{"map": {...}, "players": [...], "round": N, "result": {...}}`。

- `score` 30000 = 该方神迹被毁或一方全灭（**30000 是"胜/负"标记，不是真实分数**）。
- 正常打完：`score = 神迹剩余 HP`（可能一胜一负）。
- `winner`：分数高者；**平局时后手（camp1）+1 分**。
- 终止条件分类见 `decision_space.termination_reason`：`max_round` / `miracle_destroyed` / `ai_timeout` / `abnormal`。

> 常见误读②：看到 score=30000 会以为"拿了 30000 分"——它是胜负标记。

## 3. trace.jsonl 行格式

每行 JSON，字段：`kind`、`state`、`player`、`seq`、`payload`、`summary`、`ts`。

`summary` 是 `payload` 的 JSON 截断（host 记录时截到 200 字符），**别拿 summary 当完整内容**，解析一律用 `payload`。

| kind | state | 含义 |
|---|---|---|
| `init` | - | 对局开始 |
| `from_logic` | 0 | 官方请求选卡（`content` 为 JSON 字符串） |
| `to_logic` | - | Agent → 官方操作（`content` 是 JSON 字符串，即官方操作 dict） |
| `from_logic` | 1 | 官方返回玩家列表消息 |
| `from_logic` | 2 | **回合消息**：`content[0]` = 长度前缀 + obs JSON |
| `from_logic` | 3 | **终局帧**：完整 obs + result |

`summary` 是 `payload` 的快捷视图（同内容）；`content` 里每个元素是 `NNNNNN{...}` 形式——**前 6 位是长度前缀**，后面才是 JSON。

> 常见误读③：直接 `json.loads(content[0])` 会失败——先去掉前 6 位长度前缀。

## 4. Obs 字段与数字含义

obs JSON：`{"map": {"units": [[18 字段],...], "miracles": [hp0,hp1], "barracks": [4 个 camp]}, "players": [[5 字段],...], "round": N, "camp": 0/1}`。

**units[i] 18 字段**（索引见 `decision_space.UNIT`）：

| 索引 | 含义 |
|---|---|
| 0 `ID` | 单位 id（对局内唯一，官方全局计数器分配） |
| 1 `CAMP` | 0/1 |
| 2 `TYPE` | **全局编号**：Archer=0, Swordsman=1, BlackBat=2, Priest=3, VolcanoDragon=4, Inferno=5, FrostDragon=6 |
| 3-6 | `COST`/`ATK`/`MAX_HP`/`HP` |
| 7 `ATK_RANGE` | `[min,max]` |
| 8 `MAX_MOVE` | 移动力 |
| 9 `COOL_DOWN` | 召唤冷却 |
| 10 `POS` | `[x,y,z]` |
| 11-15 | `LEVEL`/`FLYING`/`ATK_FLYING`/`AGILITY`/`HOLY_SHIELD` |
| 16 `CAN_ATK` / 17 `CAN_MOVE` | **本回合是否可行动**（0=已行动/刚召唤/冷却） |

**players[camp] 5 字段**（`decision_space.PLAYER`）：`[artifacts, mana, max_mana, capacities, newly_summoned]`；`capacities = [[卡组序号, 上限, [已召唤 id]], ...]`。

> 常见误读④：`CAN_MOVE=0` 不是"坏单位"，是新召唤或本回合已行动；下一回合开始自动恢复。
> 常见误读⑤：`TYPE` 是全局编号，不是卡组顺序（Archer 永远是 0）。
> 常见误读⑥：obs 里没有障碍/边界信息；判断移动合法性要用 `decision_space` 的 `MAPBORDER/ABYSS/MIRACLE_OBSTACLES`。

## 5. 操作格式（to_logic 的 content）

官方操作 dict：`{"player": 0, "round": N, "operation_type": "summon|move|attack|endround|init", "operation_parameters": {...}}`。

- `summon`: `{type: 名, level: 1-3, position: [x,y,z]}`
- `move`: `{mover: id, position: [x,y,z]}`
- `attack`: `{attacker: id, target: id}`（target 可为敌方单位 id 或**敌方神迹 id=camp**）
- `endround`: 空参数

> 常见误读⑦：`attack` 的 `target` 传 `camp`（0/1）表示攻击神迹，不是单位 id。

## 6. 常见误读汇总（先查这里）

1. `.mrc` 是二进制，别 `json.load` → 用 `.trace.jsonl`。
2. score 30000 = 胜负标记，不是真实得分。
3. `content` 前 6 位是长度前缀，先切掉再 `json.loads`。
4. `CAN_MOVE/CAN_ATK=0` 是"本回合已行动/刚召唤"，非故障。
5. `TYPE` 是全局编号（Archer=0…FrostDragon=6）。
6. 障碍/边界不在 obs 里，看 `decision_space` 常量。
7. `attack target=camp` 是打神迹。
8. 平局后手 +1 分（winner 判定）。
9. 飞行单位可与地面单位同格（官方 `get_unit_at` 按 flying 过滤）。
10. 新召唤单位**当回合不能行动**（can_move/can_atk=0），下一回合才行。

## 7. 解析工作流（验证回放自洽）

1. 取 `*.mrc.trace.jsonl`，逐行解析。
2. 收集：开局选卡（state=0 的 to_logic）、每回合 obs（state=2 from_logic）、双方操作序列（to_logic）、终局（state=3）。
3. 用 `decision_space.action_mask(obs, camp)` 核对每个操作是否在合法集合内；官方接受与否以 `Parser.check_legality` 为准。
4. 用 `ig.py` 可对逐决策点做新旧策略 KL（要求 4 口径）。
5. 对照终局 `result` 与 `termination_reason` 验证终止分类。
