# Miracle 决策空间定义（要求 3）

> 实现：`src/agentbench_frame/miracle/decision_space.py`
> 一致性测试：`tests/miracle/test_decision_space.py`（官方 `Parser.check_legality` 仲裁抽查）

## 1. Observation

对局中 host 转发给 Agent 的每条回合消息 `content` 为 JSON 字符串，解析后：

```jsonc
{
  "map": {
    "units":    [[18 个字段], ...],
    "miracles": [hp0, hp1],        // 双方神迹 HP
    "barracks": [camp, camp, camp, camp]  // 4 个驻扎点归属（-1 未占领）
  },
  "players": [
    [artifacts, mana, max_mana, capacities, newly_summoned],  // camp 0
    [同结构]                                                  // camp 1
  ],
  "round": 0,   // 当前回合
  "camp": 0     // 当前行动方
}
```

### `map.units[i]` 字段索引（`UNIT.*` 常量）

| 索引 | 常量 | 含义 |
|---|---|---|
| 0 | `ID` | 单位 id（官方模块级计数器分配，对局内唯一） |
| 1 | `CAMP` | 0/1 |
| 2 | `TYPE` | 类型编号（`UNIT_NAME_PARSED`：Archer=0, Swordsman=1, BlackBat=2, Priest=3, VolcanoDragon=4, Inferno=5, FrostDragon=6） |
| 3 | `COST` | 当前等级费用 |
| 4 | `ATK` | 攻击力 |
| 5 | `MAX_HP` / 6 `HP` | 生命 |
| 7 | `ATK_RANGE` | `[min, max]` 射程 |
| 8 | `MAX_MOVE` | 移动力 |
| 9 | `COOL_DOWN` | 冷却 |
| 10 | `POS` | `[x, y, z]` 立方坐标（`x+y+z=0`） |
| 11 | `LEVEL` | 1..3 |
| 12 | `FLYING` | 1=飞行（可越 Abyss、与地面单位同格） |
| 13 | `ATK_FLYING` | 1=可对空 |
| 14 | `AGILITY` | 1=敏捷（行动后补一刀） |
| 15 | `HOLY_SHIELD` | 1=圣盾（挡一次伤害） |
| 16 | `CAN_ATK` / 17 `CAN_MOVE` | 本回合是否已行动（0=已行动/冷却中） |

### `players[camp]` 字段索引（`PLAYER.*` 常量）

| 索引 | 常量 | 含义 |
|---|---|---|
| 0 | `ARTIFACTS` | 已装备神器（列表） |
| 1 | `MANA` / 2 `MAX_MANA` | 法力 / 上限（每回合 +1，上限 12） |
| 3 | `CAPACITIES` | `[[type_idx, limit, [已召唤 id]], ...]`（按卡组顺序） |
| 4 | `NEWLY_SUMMONED` | 本回合新召唤单位 id 列表 |

## 2. 合法 macro-action

官方 `Parser` 接受的操作帧（host 发送格式）：

```json
{"player": 0, "round": 0, "operation_type": "...", "operation_parameters": {...}}
```

| 类型 | 参数 | 说明 |
|---|---|---|
| `init` | `artifacts: [名字]`, `creatures: [名字×3]` | 仅选卡阶段（state=0） |
| `summon` | `type`（名）, `level`（1..3）, `position` | 需 mana、容量、合法召唤点 |
| `move` | `mover`（单位 id）, `position` | 需 can_move、距离 ≤ max_move、路径可行 |
| `attack` | `attacker`, `target` | 需 can_atk、射程内、目标为敌单位或神迹（id=camp） |
| `endround` | 无 | 结束己方回合，恒合法 |

编码辅助：`encode_action(type, **params)` / `Action(type, params).to_dict()`。

## 3. Action mask（`action_mask(obs, camp, deck)`）

基于 obs 可见信息的规则过滤，产出当前回合全部合法候选动作：

- **endround**：恒合法。
- **summon**：类型在卡组内、容量未满（`len(cap[2]) < cap[1]`）、`mana >= cost`、
  位置 ∈ 神迹召唤点（5 个）+ 已占领 barrack 召唤点（3×n）、
  且该 flying 维度无单位占用（官方 `get_unit_at` 口径：飞行/地面互不阻挡）。
- **move**：己方 `can_move=1` 的单位，一步可达邻居（六边形 6 向），
  过滤 MAPBORDER（66 格）、Miracle 障碍、Abyss（地面单位）、同 flying 占用。
- **attack**：己方 `can_atk=1` 的单位，射程 `[lo, hi]` 内的敌方单位（hp>0）与敌方神迹。

> 口径说明：mask 是**规则近似超集**——官方逻辑是最终仲裁
> （`Parser.check_legality`），个别场景（弹道/寻路细节）以官方接受/拒绝为准。
> 测试保证：mask 全量候选 → 官方全部接受；构造的非法动作 → 官方拒绝且不在 mask。

## 4. 终止条件

| 条件 | 判定 |
|---|---|
| `max_round` | 打满 100 回合（官方 MAX_ROUND），平局后手 +1 分 |
| `miracle_destroyed` | 神迹 HP ≤ 0 或一方无单位，官方 score 记 30000 |
| `ai_timeout` | 玩家超时/卡死，官方发 error 终局帧 |

`termination_reason(terminated_by, rounds, scores, errors)` 把对局结果映射到上述条件。

## 5. 动作支持集（信息增益用）

`support_set(obs, camp)` 返回当前 obs 可选的 macro-action **类型**集合
（`{summon, move, attack, endround}` 的子集）；`action_mask` 返回带参数的完整候选。
信息增益（`ig.py`）以 mask 候选为支撑集、以 `Action.signature()` 为动作标识。
