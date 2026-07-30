 # Skill: 看懂 SnakeGo 回放（Agent 可直接消费）

 人工写清规则、回放字段与数字含义、关键事件、常见误读；由 Agent 润色，
 并用一次真实回放解析验证（见末尾「验证」一节）。

 这份文档是给「读回放做学习」的 Agent 看的：它告诉 Agent 回放文件里每个
 字段是什么、每个数字代表什么、哪些事件值得提取、哪些是误读陷阱。

 ---

 ## 0. 回放文件长什么样

 一个回放 = 一局完整、确定性的对局记录。由 `snakego/host.py` 产出，结构：

 ```jsonc
 {
   "config":   {"length":16, "width":16, "max_round":512, "seed":7},
   "items":    [{"x":6,"y":8,"time":2,"type":0,"param":4}, ...],  // 道具生成表
   "names":    ["my_v6", "sample_ai"],
   "ops":      [[round, player, snake_id, op], ...],              // 每个决策点
   "scores":   [178, 44],
   "winner":   0,
   "rounds":   452,
   "moves":    1588,
   "end_reason":"normal",
   "error":    null
 }
 ```

 关键性质：**确定性**。给定 `items` + `seed`，用 `snakego/engine.py` 重放
 `ops`，能 100% 复现终局分数（已验证：178-44 完全一致）。所以回放可以做任何
 离线分析，不需要再跑一遍游戏。

 ## 1. 字段含义

 | 字段 | 含义 | 注意 |
|------|------|------|
| `ops[i]` | `[round, player, snake_id, op]` | player∈{0,1}；snake_id 是**被操作的那条蛇**的 id；op 见下 |
| `items[k].time` | 该道具在第几回合生成 | 不是「持续时间」 |
| `items[k].type` | 0=加长(param=增加长度1-5)；2=轨道炮(param=持续回合) | type 1 不存在 |
| `items[k].param` | type0 时=加长量；type2 时=持续回合数 | 同名字段不同语义，看 type |
| `scores` | [P0, P1] 领地分；蛇身格+己方墙格各 +2 | 不是击杀数 |
| `winner` | 0/1/-1；-1 表示出错/平局 | error≠null 时 winner 不可信 |
| `rounds` | 终局回合数 | <512 多半是某方蛇全死或非法 |
| `error` | null=正常；"timeout"=超时仍有效；其余=对局作废 | 见「误读」§4 |

 ## 2. op 动作码（必须记牢）

 | op | 动作 | 判定 |
|----|------|------|
| 1 | 蛇头 +x（向右）| 出界/撞墙/撞敌身=死；撞己身=固化(seal) |
| 2 | 蛇头 +y（向上）| 同上 |
| 3 | 蛇头 -x（向左）| 同上 |
| 4 | 蛇头 -y（向下）| 同上 |
| 5 | 轨道炮 | 需持有轨道炮道具且蛇长≥2，否则非法 |
| 6 | 分裂 | 蛇长≥2 且本方蛇<4，否则非法 |

 方向是**棋盘绝对方向**，不是相对蛇头朝向。这点和很多贪吃蛇游戏不同。

 ## 3. 关键事件（用 `replay2.key_events` 自动提取）

 读回放时，这几类事件信息量最大：

 1. **split（op==6）**：分裂时机。看人类在第几回合、蛇多长时分裂。
    实测：顶尖人类在 round 10-22、蛇长 9-16 时第一次分裂，很快到 4 条蛇。
 2. **seal / 大段墙增长**：蛇头撞己身触发固化，把一片区域变己方墙。
    实测：人类从 round ~25 起每次 seal +6~+20 墙，这是得分爆发点。
 3. **railgun（op==5）**：清墙开路。人类每局用 1-3 次。
 4. **snake_died**：某条蛇被消灭（被围/被撞）。看自己第几回合死、怎么死的。
 5. **score_delta_X_Y**：某步后分数变化 X(P0) Y(P1)。大 delta 常对应 seal。

 提取方法：`snakego.replay2.key_events(replay)` 返回
 `[(step, round, player, event_str, scores)]`。

 ## 4. 常见误读（必看，否则会学错）

 1. **生长窗口假死亡**：round 1-8 蛇自动生长，长度=1 的蛇移动后蛇尾不退、
    蛇仍在。`snake_id` 在重放里若对不上，别误判为「死了」。判定真死亡要看
    该 id 是否从所属玩家的蛇列表里**永久消失**（`replay2` 已修正此误判）。
 2. **ops 里的 snake_id 错位**：早期记录在 `do_operation` **之后**取
    `current_snake_id`，得到的是下一条蛇的 id。已修为「操作前」记录。用旧回放
    分析时，分数复现不受影响（不依赖 sid），但事件分析要用修后的新回放。
 3. **error="timeout"**：某步超时但游戏继续，**结果仍有效**，可用于战绩统计；
    `error` 为 ConnectionReset 等：人类子进程崩溃，对局**作废**，不能算战绩。
 4. **scores 是领地分不是击杀分**：一方蛇全死后，另一方继续占地直到 512 回合，
    分数会持续涨。所以「终局 200 vs 0」不代表杀了 100 条蛇，是占了 100 格。
 5. **方向码 ≠ 蛇头朝向**：op=1 是棋盘 +x，与蛇当前朝向无关。分析「回头」
    要用坐标判断，不能假设 op=3 就是「倒退」。

 ## 5. Agent 怎么用回放学习

 标准流程（`snakego/loop.py` 已实现）：

 1. 读回放 → 提取自己的行为分布 `{op: 次数}` 和人类的分布。
 2. 算信息增益 IG（`snakego/ig.py`）：KL(旧版 || 新版) 看策略变了多少；
    KL(我 || 人类) 看模仿距离。KL 严格按固定 support{1..6}+拉普拉斯平滑算，
    算不出（回放不全/步数<30）就记 `incomplete_replay`，**绝不拿别的指标冒充**。
 3. 把量化发现（如「我分裂次数=0，人类=4」）写成一条 Lesson，append 进
    Experience（`snakego/experience.py`），并对相关权重做 delta。
 4. Experience 自带压缩：新教训与旧教训在同一权重轴冲突时，旧的标记
    `superseded` 归档，不堆叠。

 ## 6. 验证：一次真实回放解析（已跑通）

 回放：`work/myrollout_v6_sample_ai_s1.json`（my_v6 vs sample_ai，178-44 胜）

 ```
 $ python -c "from snakego import replay2 as R; d=R.load_replay('work/myrollout_v6_sample_ai_s1.json'); d.setdefault('config',{}); print(R.summary(d)); print('score复现一致:', list(R.iter_frames(d))[-1][1]['scores']==d['scores'])"
 my_v6 vs sample_ai: 178-44, winner=my_v6, rounds=452, moves=1588, reason=normal
 score复现一致: True
 ```

 提取到的关键事实（来自 `work/analyze_humans.py`）：
 - 顶尖人类首次分裂：round 10-22、蛇长 9-16；最多 4 条蛇。
 - 人类从 round ~25 起做 seal，每次 +6~+20 墙。
 - 人类每局用轨道炮 1-3 次；我（旧版）0 次。

 这些事实已固化进 `experience.py` 的 SEED_LESSONS，作为策略迭代的起点。
