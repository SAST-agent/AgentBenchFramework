"""26_snakego / 贪吃蛇围棋 — clean in-process game environment.

Faithful port of the Saiblo judge logic (judge_dev_logic/) with the
stdin/stdout protocol stripped out, so agents can play directly in one
process.  Grid is length×width (default 16×16), 2 players, 512 turns.

Coordinate system (kept identical to the original):
    x in [0, length),  y in [0, width)   —  map[x][y]
    directions:  0=+x  1=+y  2=-x  3=-y
    actions:     1..4 = move (action-1 = direction)
                 5 = railgun/fire   6 = split

Turn structure: each logical *turn* both players act.  Within a player's
turn they operate every alive snake they own, one action each.  After
both players finish, the turn counter advances and items are processed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

ITEM_EXPIRE_TIME = 16

# direction deltas — index = direction
DX = [1, 0, -1, 0]
DY = [0, 1, 0, -1]

ACT_MOVE_BASE = 1      # actions 1..4 are moves
ACT_RAILGUN = 5
ACT_SPLIT = 6


# --------------------------------------------------------------------------- #
# spawn configuration
# --------------------------------------------------------------------------- #


@dataclass
class SpawnConfig:
    start_round: int
    end_round: int
    mean: float
    area: Tuple[int, int, int, int]
    weight: Tuple[int, int, int]


# (start, end, mean, area, weight(length, split, fire))
# split items (type 1) are unused in the original generator (falls into the
# type-0 branch), so weight[1] effectively adds more length items.
SPAWN_TABLE = [
    SpawnConfig(1,   20,  0.25, (6, 6, 9, 9),   (5, 0, 1)),
    SpawnConfig(21,  40,  0.25, (5, 5, 10, 10), (5, 0, 1)),
    SpawnConfig(41,  60,  0.25, (4, 4, 11, 11), (5, 0, 1)),
    SpawnConfig(61,  64,  0.25, (3, 3, 12, 12), (5, 0, 1)),
    SpawnConfig(65,  80,  0.25, (3, 3, 12, 12), (5, 0, 2)),
    SpawnConfig(81,  100, 0.25, (2, 2, 13, 13), (5, 0, 2)),
    SpawnConfig(101, 120, 0.25, (1, 1, 14, 14), (5, 0, 2)),
    SpawnConfig(121, 384, 0.25, (0, 0, 15, 15), (5, 0, 2)),
    SpawnConfig(385, 512, 0.25, (0, 0, 15, 15), (4, 0, 3)),
]


@dataclass
class GameConfig:
    length: int = 16
    width: int = 16
    max_round: int = 512
    seed: int = 0


@dataclass
class Item:
    id: int
    x: int
    y: int
    time: int        # round at which it appears
    type: int        # 0 = length, 2 = railgun
    param: int       # length gain, or max-round for railgun
    gotten_time: int = -1

    def to_dict(self) -> dict:
        return {"id": self.id, "x": self.x, "y": self.y, "time": self.time,
                "type": self.type, "param": self.param,
                "gotten_time": self.gotten_time}


def _generate_items(cfg: GameConfig) -> List[Item]:
    """Port of spawn.generate_items — deterministic given cfg.seed."""
    rng = random.Random(cfg.seed)
    items: List[Item] = []
    nxt_id = 0
    # item_map tracks the last round a cell was occupied, to avoid overlap
    item_map = [[-1024] * cfg.width for _ in range(cfg.length)]

    def gen_single(round_: int, sc: SpawnConfig) -> Item:
        nonlocal nxt_id
        x1, y1, x2, y2 = sc.area
        x = rng.randint(x1, x2)
        y = rng.randint(y1, y2)
        while item_map[x][y] + ITEM_EXPIRE_TIME >= round_:
            x = rng.randint(x1, x2)
            y = rng.randint(y1, y2)
        item_map[x][y] = round_
        t = rng.randint(0, sum(sc.weight) - 1)
        if t < sc.weight[0]:
            it = Item(nxt_id, x, y, round_, 0, rng.randint(1, 5))
        elif t < sc.weight[0] + sc.weight[1]:
            it = Item(nxt_id, x, y, round_, 0, rng.randint(1, 5))
        else:
            it = Item(nxt_id, x, y, round_, 2, cfg.max_round)
        nxt_id += 1
        return it

    current = 1
    for sc in SPAWN_TABLE:
        while current in range(sc.start_round, sc.end_round + 1):
            if rng.uniform(0, 1) <= sc.mean:
                items.append(gen_single(current, sc))
            current += 1
    return items


# --------------------------------------------------------------------------- #
# entities
# --------------------------------------------------------------------------- #


@dataclass
class Snake:
    coor_list: List[Tuple[int, int]]
    item_list: List[Item]
    length_bank: int = 0
    camp: int = 0
    id: int = -1

    @property
    def length(self) -> int:
        return len(self.coor_list)

    def add_item(self, item: Item) -> None:
        if item.type == 0:
            self.length_bank += item.param
        else:
            for idx in range(len(self.item_list)):
                if self.item_list[idx].type == item.type:
                    self.item_list[idx] = item
                    return
            self.item_list.append(item)

    def has_railgun(self) -> bool:
        return any(it.type == 2 for it in self.item_list)


class IllegalAction(Exception):
    """Raised when a submitted action is illegal (would lose the game)."""
    def __init__(self, player: int, msg: str):
        self.player = player
        self.msg = msg
        super().__init__(f"[P{player}] {msg}")


# --------------------------------------------------------------------------- #
# flood-fill enclosure (port of operate.Graph)  — iterative
# --------------------------------------------------------------------------- #


def _convert_dir(u, v) -> int:
    x = u[0] - v[0]
    y = u[1] - v[1]
    if y == 0:
        return x + 1
    if x == 0:
        return y + 2
    return 0


def _calc_enclosure(bound, length, width) -> List[Tuple[int, int]]:
    """Return enclosed cells when *bound* (a snake segment) closes a loop."""
    table = [[0] * width for _ in range(length)]
    for x, y in bound:
        table[x][y] = -1
    inner = {1: True, 2: True}

    def floodfill(sx, sy, c):
        if not (0 <= sx < length and 0 <= sy < width):
            inner[c] = False
            return
        if table[sx][sy] != 0:
            return
        stack = [(sx, sy)]
        table[sx][sy] = c
        while stack:
            x, y = stack.pop()
            for i in range(4):
                tx, ty = x + DX[i], y + DY[i]
                if not (0 <= tx < length and 0 <= ty < width):
                    inner[c] = False
                    continue
                if table[tx][ty] == 0:
                    table[tx][ty] = c
                    stack.append((tx, ty))

    for i in range(len(bound)):
        d = _convert_dir(bound[i], bound[i - 1])
        floodfill(bound[i][0] + DX[(d + 3) % 4], bound[i][1] + DY[(d + 3) % 4], 1)
        floodfill(bound[i][0] + DX[(d + 1) % 4], bound[i][1] + DY[(d + 1) % 4], 2)

    ret = []
    for k in (1, 2):
        if inner[k]:
            for i in range(length):
                for j in range(width):
                    if table[i][j] == k:
                        ret.append((i, j))
    return ret


# --------------------------------------------------------------------------- #
# the game
# --------------------------------------------------------------------------- #


class SnakeGoGame:
    """One complete SnakeGo game, drivable action-by-action."""

    def __init__(self, config: GameConfig | None = None):
        self.config = config or GameConfig()
        self._reset()

    # -- state setup --------------------------------------------------------

    def _reset(self) -> None:
        cfg = self.config
        L, W = cfg.length, cfg.width
        self.snakes: List[Snake] = [
            Snake(coor_list=[(0, W - 1)], item_list=[], camp=0, id=0),
            Snake(coor_list=[(L - 1, 0)], item_list=[], camp=1, id=1),
        ]
        self._next_snake_id = 2
        self.items: List[Item] = _generate_items(cfg)
        self.wall_map = [[-1] * W for _ in range(L)]
        self.snake_map = [[-1] * W for _ in range(L)]
        self.snake_map[0][W - 1] = 0
        self.snake_map[L - 1][0] = 1
        self.item_map = [[-1] * W for _ in range(L)]
        self.turn = 1
        self.current_player = 0
        self.auto_growth_round = 8
        self.first_get_time = -1
        self.first_get_player = -1
        self._over = False
        self._winner = -1
        self._scores = [0, 0]
        self._error = None           # IllegalAction that ended the game
        self._ply_snakes: List[Tuple[Snake, bool]] = []
        self._ply_next = -1
        self._init_ply()
        # full action log for replay
        self.action_log: List[dict] = []
        self.round_logs: List[dict] = []

    def reset(self, seed: int | None = None) -> "SnakeGoGame":
        if seed is not None:
            self.config = GameConfig(self.config.length, self.config.width,
                                     self.config.max_round, seed)
        self._reset()
        return self

    def _init_ply(self) -> None:
        """Set up the per-turn snake list for the current player."""
        self._ply_snakes = [(s, False) for s in self.snakes]
        self._ply_next = -1
        self._advance_to_next_snake()

    def _advance_to_next_snake(self) -> None:
        start = self._ply_next + 1
        for idx in range(start, len(self._ply_snakes)):
            s, dead = self._ply_snakes[idx]
            if s.camp == self.current_player and not dead:
                self._ply_next = idx
                return
        self._ply_next = -1

    # -- public query API ---------------------------------------------------

    def current_snake_id(self) -> Optional[int]:
        if self._ply_next == -1:
            return None
        return self._ply_snakes[self._ply_next][0].id

    def is_over(self) -> bool:
        return self._over

    def scores(self) -> Tuple[int, int]:
        return tuple(self._scores)  # type: ignore[return-value]

    def winner(self) -> int:
        return self._winner

    def snake_count(self, camp: int) -> int:
        return sum(1 for s in self.snakes if s.camp == camp)

    def obs(self) -> dict:
        """Full observable state for an agent."""
        return {
            "length": self.config.length,
            "width": self.config.width,
            "turn": self.turn,
            "max_round": self.config.max_round,
            "current_player": self.current_player,
            "current_snake_id": self.current_snake_id(),
            "snakes": [
                {"id": s.id, "camp": s.camp,
                 "coor_list": list(s.coor_list),
                 "length_bank": s.length_bank,
                 "railgun": s.has_railgun()}
                for s in self.snakes
            ],
            "wall_map": [row[:] for row in self.wall_map],
            "snake_map": [row[:] for row in self.snake_map],
            "item_map": [row[:] for row in self.item_map],
            "items": [it.to_dict() for it in self.items
                      if it.gotten_time == -1],
        }

    # -- legal actions ------------------------------------------------------

    def valid_actions(self) -> List[int]:
        """Actions that are legal (won't trigger an illegal-action loss).

        Suicidal moves (into walls / other snakes) are excluded too, since
        they instantly kill the snake and no sane agent wants them.
        Solidify moves (closing a loop on yourself) are included.
        """
        sid = self.current_snake_id()
        if sid is None:
            return []
        snake = self._get_snake(sid)
        acts: List[int] = []
        coor = snake.coor_list
        grows = self._grows(snake)
        for d in range(4):
            nx, ny = coor[0][0] + DX[d], coor[0][1] + DY[d]
            if not (0 <= nx < self.config.length and 0 <= ny < self.config.width):
                continue
            if self.wall_map[nx][ny] != -1:
                continue
            occ = self.snake_map[nx][ny]
            if occ != -1 and occ != snake.id:
                continue  # another snake — would die
            # turn-back check
            if (len(coor) > 2 or (len(coor) == 2 and grows)) and \
                    (nx, ny) == coor[1]:
                continue
            acts.append(ACT_MOVE_BASE + d)
        # railgun
        if snake.has_railgun() and len(coor) > 1:
            acts.append(ACT_RAILGUN)
        # split
        if self.snake_count(snake.camp) < 4 and len(coor) > 1:
            acts.append(ACT_SPLIT)
        if not acts:
            acts = [ACT_MOVE_BASE]  # fallback — should not normally happen
        return acts

    def _grows(self, snake: Snake) -> bool:
        return (self.turn <= self.auto_growth_round and
                snake.camp == snake.id) or snake.length_bank > 0

    def _get_snake(self, sid: int) -> Snake:
        for s in self.snakes:
            if s.id == sid:
                return s
        raise KeyError(f"snake {sid} not found")

    # -- apply one action ---------------------------------------------------

    def act(self, action: int) -> dict:
        """Apply *action* for the current snake; return an info dict."""
        if self._over:
            return {"over": True}
        sid = self.current_snake_id()
        if sid is None:
            self._end_player_turn()
            return {"turn_end": True}

        snake = self._get_snake(sid)
        info: dict = {"player": self.current_player, "snake": sid,
                      "action": action}
        try:
            if action <= 4:
                self._move(snake, action - 1, info)
            elif action == ACT_RAILGUN:
                self._railgun(snake, info)
            elif action == ACT_SPLIT:
                self._split(snake, info)
            else:
                raise IllegalAction(snake.camp, f"bad action {action}")
        except IllegalAction as exc:
            self._end_with_error(exc)
            info["illegal"] = exc.msg
            return info

        self.action_log.append(info)
        self._advance_to_next_snake()
        if self._ply_next == -1:
            self._end_player_turn()
        return info

    def _end_player_turn(self) -> None:
        # switch player
        self.current_player = 1 - self.current_player
        if self.current_player == 0:
            # full round done — process items & check end
            self._process_round()
        # set up next ply (may immediately be over)
        if not self._over:
            self._init_ply()
            # if current player has no snakes, keep advancing
            while not self._over and self.current_snake_id() is None:
                self.current_player = 1 - self.current_player
                if self.current_player == 0:
                    self._process_round()
                if not self._over:
                    self._init_ply()

    # -- round processing (port of round_preprocess) -----------------------

    def _process_round(self) -> None:
        ri: dict = {"turn": self.turn, "expired": [], "gotten": [], "extra_len": []}
        # expire old map items
        for it in list(self.items):
            if it.gotten_time == -1 and it.time <= self.turn - ITEM_EXPIRE_TIME:
                if self.item_map[it.x][it.y] == it.id:
                    self.item_map[it.x][it.y] = -1
                ri["expired"].append(it.id)
        # items appearing this turn
        for it in list(self.items):
            if it.gotten_time == -1 and it.time == self.turn:
                sid = self.snake_map[it.x][it.y]
                if sid >= 0:
                    it.gotten_time = self.turn
                    self._note_first_get(self._get_snake(sid).camp)
                    self._get_snake(sid).add_item(it)
                    ri["gotten"].append((it.id, sid))
                else:
                    self.item_map[it.x][it.y] = it.id
        # expire held railgun items
        for s in self.snakes:
            for it in list(s.item_list):
                if it.type == 2 and it.gotten_time != -1 and \
                        self.turn - it.gotten_time > it.param:
                    s.item_list.remove(it)
                    ri["expired"].append(it.id)
        self.round_logs.append(ri)
        self.turn += 1
        self._check_end()

    def _note_first_get(self, camp: int) -> None:
        if self.first_get_time == -1 or \
                (self.first_get_time == self.turn and self.first_get_player == 0):
            self.first_get_time = self.turn
            self.first_get_player = camp

    def _check_end(self) -> None:
        n = [self.snake_count(0), self.snake_count(1)]
        if n == [0, 0] or self.turn > self.config.max_round:
            self._settle()

    # -- scoring (port of settle_round) ------------------------------------

    def _settle(self) -> None:
        score = [0, 0]
        for x in range(self.config.length):
            for y in range(self.config.width):
                sid = self.snake_map[x][y]
                if sid != -1:
                    score[self._get_snake(sid).camp] += 2
                elif self.wall_map[x][y] == 0:
                    score[0] += 2
                elif self.wall_map[x][y] == 1:
                    score[1] += 2
        if self.first_get_player != -1:
            score[self.first_get_player] += 1
        self._scores = score
        self._winner = 0 if score[0] >= score[1] else 1
        self._over = True

    def _end_with_error(self, exc: IllegalAction) -> None:
        self._settle()
        # player who erred loses
        self._scores[exc.player] = -100
        self._winner = 1 - exc.player
        self._error = exc
        self._over = True

    # -- move (port of Controller.move) ------------------------------------

    def _delete_snake(self, sid: int) -> None:
        # Scan the whole grid rather than snake.coor_list, so we clear
        # stale entries regardless of when coor_list was updated.
        for x in range(self.config.length):
            for y in range(self.config.width):
                if self.snake_map[x][y] == sid:
                    self.snake_map[x][y] = -1
        self.snakes = [s for s in self.snakes if s.id != sid]
        # mark dead in ply list
        self._ply_snakes = [(s, (s.id == sid or dead))
                            for s, dead in self._ply_snakes]

    def _add_snake(self, snake: Snake, index: int) -> None:
        self.snakes.insert(index, snake)
        for x, y in snake.coor_list:
            self.snake_map[x][y] = snake.id

    def _collect_item(self, snake: Snake, item_id: int) -> None:
        self._note_first_get(snake.camp)
        item = None
        for it in self.items:
            if it.id == item_id:
                item = it
                break
        if item is None:
            return
        item.gotten_time = self.turn
        snake.add_item(item)
        self.item_map[item.x][item.y] = -1

    def _move(self, snake: Snake, direction: int, info: dict) -> None:
        sid = snake.id
        coor = snake.coor_list
        auto_grow = self.turn <= self.auto_growth_round and snake.camp == snake.id
        x, y = coor[0][0] + DX[direction], coor[0][1] + DY[direction]
        if len(coor) == 1:
            new_coor = [(x, y)]
        else:
            new_coor = [(x, y)] + coor[:-1]
        items_get = -1

        # out of bounds / wall → die
        if not (0 <= x < self.config.length and 0 <= y < self.config.width) or \
                self.wall_map[x][y] != -1:
            self._delete_snake(sid)
            info.update(result="die", dead=[sid])
            return

        # turn-back
        if (len(coor) > 2 or (len(coor) == 2 and (auto_grow or snake.length_bank))) \
                and (x, y) == coor[1]:
            raise IllegalAction(snake.camp, "DON'T TURN BACK!")

        # grow
        if auto_grow:
            new_coor = new_coor + [coor[-1]]
        elif snake.length_bank:
            snake.length_bank -= 1
            new_coor = new_coor + [coor[-1]]

        snake.coor_list = new_coor

        # pick up item
        if self.item_map[x][y] != -1:
            items_get = self.item_map[x][y]
            self._collect_item(snake, items_get)

        # self-intersection → solidify
        for i in range(1, len(new_coor)):
            if x == new_coor[i][0] and y == new_coor[i][1]:
                dead = [sid]
                solid_coor = new_coor[:i]
                extra = _calc_enclosure(solid_coor, self.config.length,
                                        self.config.width)
                for c in new_coor[i:]:
                    if c in extra:
                        solid_coor.append(c)
                        extra.remove(c)
                for c in list(extra):
                    if self.snake_map[c[0]][c[1]] != -1:
                        victim = self.snake_map[c[0]][c[1]]
                        dead.append(victim)
                        self._delete_snake(victim)
                for c in solid_coor:
                    self.wall_map[c[0]][c[1]] = snake.camp
                for c in extra:
                    self.wall_map[c[0]][c[1]] = snake.camp
                self._delete_snake(sid)
                all_solid = sorted(extra + solid_coor)
                info.update(result="solidify", dead=dead,
                            solidified=all_solid, got_item=items_get)
                return

        # collision with another snake → die
        if self.snake_map[x][y] != -1 and self.snake_map[x][y] != sid:
            self._delete_snake(sid)
            info.update(result="die", dead=[sid], got_item=items_get)
            return

        # normal move — update map
        self._delete_snake(sid)
        snake.coor_list = new_coor
        self._add_snake(snake, _find_insert(self.snakes, sid))
        info.update(result="move", dead=[], got_item=items_get)

    # -- railgun (port of Controller.fire) ----------------------------------

    def _railgun(self, snake: Snake, info: dict) -> None:
        if not snake.has_railgun():
            raise IllegalAction(snake.camp, "No railgun item!")
        if len(snake.coor_list) <= 1:
            raise IllegalAction(snake.camp, "Fire length too short!")
        # consume first railgun item
        snake.item_list = [it for it in snake.item_list if it.type != 2] \
            if False else snake.item_list  # keep ordering; pop first type-2
        for i, it in enumerate(snake.item_list):
            if it.type == 2:
                snake.item_list.pop(i)
                break
        x1, y1 = snake.coor_list[0]
        x2, y2 = snake.coor_list[1]
        dx, dy = x1 - x2, y1 - y2
        walls = []
        while 0 <= x1 + dx < self.config.length and \
                0 <= y1 + dy < self.config.width:
            x1, y1 = x1 + dx, y1 + dy
            walls.append((x1, y1))
        for wx, wy in walls:
            self.wall_map[wx][wy] = -1
        info.update(result="railgun", cleared=walls)

    # -- split (port of Controller.split) ----------------------------------

    def _split(self, snake: Snake, info: dict) -> None:
        if self.snake_count(snake.camp) >= 4:
            raise IllegalAction(snake.camp, "Split too much!")
        coor = snake.coor_list
        if len(coor) <= 1:
            raise IllegalAction(snake.camp, "Split length too short!")
        mid = (len(coor) + 1) // 2
        head = coor[:mid]
        tail = coor[mid:][::-1]
        h_item, t_item = [], []
        for it in snake.item_list:
            if it.type == 0:
                t_item.append(it)
            elif it.type == 2:
                h_item.append(it)
        snake.coor_list = head
        snake.item_list = h_item
        new = Snake(coor_list=tail, item_list=t_item,
                    camp=snake.camp, id=self._next_snake_id)
        new.length_bank = snake.length_bank
        snake.length_bank = 0
        self._next_snake_id += 1
        self._delete_snake(snake.id)
        # re-add head at original position
        self._add_snake(snake, _find_insert(self.snakes, snake.id))
        self._add_snake(new, _find_insert(self.snakes, snake.id) + 1)
        info.update(result="split", new_snake=new.id)


def _find_insert(snakes: List[Snake], sid: int) -> int:
    """Find the index where snake *sid* lived, for re-insertion."""
    for i, s in enumerate(snakes):
        if s.id > sid:
            return i
    return len(snakes)


__all__ = [
    "SnakeGoGame", "GameConfig", "Item", "Snake", "IllegalAction",
    "ACT_MOVE_BASE", "ACT_RAILGUN", "ACT_SPLIT",
]
