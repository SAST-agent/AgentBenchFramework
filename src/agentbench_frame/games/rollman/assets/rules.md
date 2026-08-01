# 29_rollman 冻结游戏规则

## 1. 权威契约

科研运行使用 AgentBench commit `b581bca3ba3d2d7d58a2f8c6bbddd060fc7fdc87` 中的冻结后端：

```text
backend_sources/corpus/29_rollman/logic/gamecode_logic/PacmanLogic
```

规则解释优先级：

1. 冻结后端；
2. PacmanLogic 与 Logic-core 固定提交；
3. 官方游戏规则、通信文档与 SDK 文档；
4. 开发引导示例。

固定源码：

- PacmanLogic `81d0468d177089cefe1f08ed1cffe78beb0d27e9`
- Logic-core `b293c04746fc3bf9b67a00130a2ca15fc38691bc`
- PacmanSDK-python `7bd36b9938570ad0cc0dcbd80d1a9d21efbfa539`
- GhostsSDK-python `da80f17c0341e4bfbe464e64f21a379c5c988961`

冻结后端 `core` 与 Logic-core 固定提交逐文件哈希一致。

## 2. 角色、关卡与胜负

一场比赛有两位选手：

- Rollman（卷王，role 0）控制一个角色；
- Ghosts（幽灵，role 1）同时控制三个幽灵。

比赛包含三关：

| 关卡 | 地图大小 | 最大轮数 | 传送门 |
|---|---:|---:|---|
| 1 | 41×41 | 500 | 第 60 轮结算后开放 |
| 2 | 32×32 | 400 | 第 50 轮结算后开放 |
| 3 | 22×22 | 300 | 无下一关传送门 |

传送门在开放后的下一轮才能进入。关卡通过以下任一条件结束：

- Rollman 在碰撞结算后仍有效，并且本轮路径经过已开放传送门；
- Rollman 吃完本关全部知识金币和双倍荣耀币；
- 达到本关最大轮数。

第三关结束后比赛结束。Rollman 总分与三个幽灵总分比较：

- Rollman 分数更高：Rollman 胜；
- Ghosts 分数更高：Rollman 负；
- 分数相同：平局。

AI 初始化时限 20 秒，每次操作时限 1 秒，空间上限 512 MB。协议错误、非法操作、超时或运行错误依冻结后端异常分处理。

## 3. 操作

Rollman 每轮提交一个整数，Ghosts 每轮提交三个有序整数：

| 操作码 | 动作 | 坐标变化 |
|---:|---|---|
| 0 | STAY | (0, 0) |
| 1 | UP | (+1, 0) |
| 2 | LEFT | (0, -1) |
| 3 | DOWN | (-1, 0) |
| 4 | RIGHT | (0, +1) |

五个操作码在协议层均合法。动作指向墙壁时，角色停留在最后一个有效坐标；该动作不会从标准决策空间中删除。

Rollman 拥有疾风之翼时，同一个方向在一轮内执行两次，形成两个子步。每个子步都可能拾取物品。

## 4. 地图元素

| 数值 | 枚举 | 含义 |
|---:|---|---|
| 0 | WALL | 墙 |
| 1 | EMPTY | 空地 |
| 2 | REGULAR_BEAN | 知识金币，基础分 1 |
| 3 | BONUS_BEAN | 双倍荣耀币，基础分 2 |
| 4 | SPEED_BEAN | 疾风之翼 |
| 5 | MAGNET_BEAN | 智引磁石 |
| 6 | SHIELD_BEAN | 护学之盾 |
| 7 | DOUBLE_BEAN | 智慧圣典 |
| 8 | FROZE_BEAN | 时间宝石 |
| 9 | PORTAL | 传送门 |

每关地图随机生成障碍、物品和四角出生位。关卡 1、2、3 的障碍物数量分别为 16、9、4。

## 5. 道具与生效时序

`pacman_skill_status` 的顺序为：

```text
[DOUBLE_SCORE, SPEED_UP, MAGNET, SHIELD, FROZE]
```

冻结常量为 `[8, 8, 8, 8, 2]`。其中护盾位置表示可叠加的护盾数量；其余位置表示剩余轮数。

- 智慧圣典：普通金币和双倍荣耀币的得分乘 2，持续 8 轮；
- 疾风之翼：每轮沿同一方向移动两个子步，持续 8 轮；
- 智引磁石：拾取路径位置周围 3×3 范围内的物品，持续 8 轮；
- 护学之盾：抵挡一次同轮抓捕，可叠加；
- 时间宝石：幽灵动作按 STAY 结算，持续 2 轮。

在第 `i` 轮拾取的技能从第 `i+1` 轮开始生效。再次拾取同类计时技能会重置剩余轮数。进入新关卡或被吃掉会清除全部特殊状态；护盾破坏只消耗一层护盾。

## 6. 计分

### 6.1 Rollman

- 知识金币：基础分 +1；
- 双倍荣耀币：基础分 +2；
- 同一关连续 100 轮未被吃且未破盾：+50；
- 吃完本关全部普通/双倍金币：+50；
- 被幽灵吃掉：-60；
- 提前通关时间奖励：

```text
floor((MaxRound(level) - CurrentRound) * 0.43)
```

智慧圣典只翻倍普通金币与双倍荣耀币，不翻倍连续存活、吃完金币、死亡惩罚或时间奖励。

### 6.2 Ghosts

- 吃掉 Rollman：实施抓捕的幽灵 +50；
- 摧毁一层护盾：实施抓捕的幽灵 +10；
- 每累计吃掉 Rollman 5 次：三个幽灵各 +20，计数清零；
- 本关因最大轮数结束：三个幽灵各 +25。

Ghosts 选手最终得分是三个幽灵分数之和。

## 7. 单轮结算顺序

每轮按以下顺序：

1. 保存本轮开始时技能状态并递减计时；
2. 若时间宝石有效，将三个幽灵动作替换为 STAY；
3. Rollman 移动并沿路径拾取物品；
4. 三个幽灵移动；
5. 处理 Rollman 终点物品；
6. 按路径判断抓捕；
7. 处理护盾、无敌、死亡、重生和连续存活奖励；
8. 激活本轮拾取的技能状态；
9. 判断是否经过已开放传送门；
10. 更新传送门开放状态；
11. 判断是否吃完金币；
12. 判断关卡是否达到最大轮数。

抓捕优先于进入传送门。Rollman 本轮被吃掉时不能通过传送门。

## 8. 路径抓捕

抓捕比较 Rollman 与每个幽灵本轮的路径，不包含起点，包含终点。

- 普通一步路径补入中点，使其与疾风之翼的两个子步具有统一的三个采样点；
- 撞墙路径恢复为最后一个有效坐标；
- 对应路径采样点的曼哈顿距离小于等于 `0.5` 时视为相遇；
- 任一幽灵相遇即触发一次抓捕；
- 同轮多个幽灵相遇只结算一次，按冻结后端的幽灵遍历顺序确定得分者。

护盾被摧毁后获得 1 轮无敌；被吃并重生后获得 3 轮无敌。无敌期间的相遇不触发抓捕。

Rollman 被吃后重生到空地中“到三个幽灵的最小曼哈顿距离”最大的格子，并清除全部技能。

## 9. AI 可见状态

策略入口是 `ai_func(game_state)`；冻结 Python SDK 传入 `GameState`
（即 `core.gamedata.GameState`）对象，而不是回放 JSON 字典。对象直接属性为：

- `space_info`
- `level`
- `round`
- `board_size`
- `board`
- `pacman_skill_status`
- `pacman_pos`
- `ghosts_pos`
- `pacman_score`
- `ghosts_score`
- `beannumber`
- `portal_available`
- `portal_coord`

策略可调用对象的 `gamestate_to_statedict()` 方法（即
`game_state.gamestate_to_statedict()`）得到规范字典。该字典将
`pacman_pos` 映射为 `pacman_coord`、将 `ghosts_pos` 映射为
`ghosts_coord`，并将 `pacman_score` 与 `ghosts_score` 合并为
`score = [pacman_score, ghosts_score]`。策略不得假设对象本身直接提供
`pacman_coord`、`ghosts_coord` 或 `score` 属性。

规范字典包含：

- `level`
- `round`
- `board_size`
- `board`
- `pacman_skill_status`
- `pacman_coord`
- `ghosts_coord`
- `score`
- `beannumber`
- `portal_available`
- `portal_coord`

每轮结束，AI 收到双方实际操作：

- `pacman_action`
- `ghosts_action`

Rollman 不获得 Ghosts 的源码或隐藏意图。

## 10. 回放格式

回放是 UTF-8 JSON Lines 文件。每行是一个完整 JSON 对象。

### 10.1 关卡初始化帧

每关开始写入一条全量初始化状态，字段与 AI 可见状态一致。`board` 是二维整数数组，坐标直接使用冻结后端坐标系。

### 10.2 增量帧

每轮执行后写入一条增量帧：

- `round`：本关轮数；
- `level`：关卡号；
- `pacman_step_block`：Rollman 本轮原始路径；
- `pacman_coord`：Rollman 结算后坐标；
- `pacman_skills`：本轮开始时的技能状态；
- `ghosts_step_block`：三个幽灵的本轮原始路径；
- `ghosts_coord`：三个幽灵结算后坐标；
- `score`：`[Rollman 总分, Ghosts 总分]`；
- `events`：本轮事件整数列表；
- `portal_available`：结算后的传送门状态；
- `StopReason`：正常增量帧为 `null`。

事件：

| 数值 | 事件 |
|---:|---|
| 0 | EATEN_BY_GHOST |
| 1 | SHIELD_DESTROYED |
| 2 | FINISH_LEVEL |
| 3 | TIMEOUT |

`0` 与 `1` 不应同轮同时出现；`2` 与 `3` 不应同轮同时出现。

### 10.3 终止帧

比赛终止时再写入一条与增量帧同结构的对象，`StopReason` 为非空字符串。基础设施异常可能留下不可解析尾部；此类回放必须标记为无效，不能按负局或零分进入科研聚合。

## 11. 科研决策与信息增益

标准 Rollman 决策是提交给后端的方向 `0..4`。每个真实到达的 Rollman 决策点，在动作执行前保存可见状态和候选程序私有记忆标识。

确定性动作通过统一 epsilon 测量通道转成测量分布：

```text
pi^epsilon(a|z) = (1-epsilon) I[a=f(z)] + epsilon/5
```

局部 KL 只比较相邻版本在同一完整方向支持上的行为。规则不引入“目标物品”“路线意图”或“战术假设”动作，也不从后续轨迹反推高层决策。

## 12. 可复现 seed

冻结 `main.py` 读取 `config.random_seed`，但未调用 Python `random.seed`。Framework 的 Rollman adapter 必须在构造 `PacmanEnv` 前调用：

```text
random.seed(seed)
numpy.random.seed(seed)
```

每场比赛记录 seed、后端哈希、双方版本、角色、原始回放和规范化回放哈希。

## 13. 官方资料

- https://agent-guide.net9.org/document/pacman/
- https://agent-guide.net9.org/document/pacman_api/
- https://agent-guide.net9.org/sdk_structure/pacman_sdk/
- https://github.com/PacMan-Logic/PacmanLogic
- https://github.com/PacMan-Logic/PacmanSDK-python
- https://github.com/PacMan-Logic/GhostsSDK-python
