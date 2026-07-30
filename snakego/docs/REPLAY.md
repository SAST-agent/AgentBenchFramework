# 回放查看器使用说明

## 快速开始

```bash
python -m snakego.replay_demo
```

生成两个文件：
- `outputs/snakego_replays/greedy_vs_random.json` — 回放数据
- `outputs/snakego_replays/greedy_vs_random.html` — HTML 查看器

用浏览器打开 `.html` 文件即可观看回放。

## 录制任意对战

```python
from snakego.agents import GreedyAgent, RandomAgent
from snakego.replay import play_with_replay, build_html

result, recorder = play_with_replay(
    GreedyAgent(), RandomAgent(0), seed=42
)
build_html("my_game.json", "my_game.html")
```

## HTML 查看器功能

| 控制 | 功能 |
|------|------|
| 播放/暂停 | 自动从头到尾播放 |
| 步进 | 单步前进/后退 |
| 速度 | 调整播放速度（1x-10x） |
| 拖拽 | 拖动进度条跳转到任意帧 |

## 回放数据格式

每帧包含：
- `wall_map` — 墙壁地图（16×16，值=阵营或 -1）
- `snake_map` — 蛇身地图（16×16，值=蛇 ID 或 -1）
- `item_map` — 道具地图（16×16，值=道具 ID 或 -1）
- `snakes` — 所有蛇的坐标列表和状态
- `scores` — 当前双方得分
- `turn` — 当前回合

## 事件类型

回放中记录的事件：
- `move` — 蛇移动
- `die` — 蛇死亡
- `solidify` — 围困固化（附围困区域坐标）
- `railgun` — 轨道炮发射（附清除的墙壁）
- `split` — 蛇分裂
- `game_over` — 游戏结束

## 画面说明

- **蓝色格** — P0 蛇身
- **红色格** — P1 蛇身
- **深蓝墙** — P0 围困区域
- **深红墙** — P1 围困区域
- **绿色圆** — 加长道具
- **黄色圆** — 轨道炮道具
