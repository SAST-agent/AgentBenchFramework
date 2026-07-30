"""Faithful Python port of adk.hpp Context (the SnakeGo game engine).

Mirrors the C++ reference bit-for-bit so subprocess human AIs (compiled from the
same adk.hpp) stay in perfect sync with the host.

Coordinate convention matches adk.hpp: grids are grid[x][y], x in [0,length),
y in [0,width). Op types: 1=right(+x) 2=up(+y) 3=left(-x) 4=down(-y) 5=railgun 6=split.
"""
from collections import deque

GROWING_ROUNDS = 8
ITEM_EXPIRE_LIMIT = 16
SNAKE_LIMIT = 4

MOVE_DX = [0, 1, 0, -1, 0]
MOVE_DY = [0, 0, 1, 0, -1]
SEAL_DX = [1, 0, -1, 0]
SEAL_DY = [0, 1, 0, -1]


class Item:
    __slots__ = ("x", "y", "id", "time", "type", "param", "eaten", "expired")

    def __init__(self, x, y, id, time, type, param):
        self.x = x
        self.y = y
        self.id = id
        self.time = time
        self.type = type
        self.param = param
        self.eaten = False
        self.expired = False


class Snake:
    __slots__ = ("coord_list", "id", "length_bank", "camp", "railgun_item_id")

    def __init__(self, coords, id, length_bank, camp, railgun_item_id):
        self.coord_list = coords
        self.id = id
        self.length_bank = length_bank
        self.camp = camp
        self.railgun_item_id = railgun_item_id

    @property
    def length(self):
        return len(self.coord_list)


class Engine:
    def __init__(self, length, width, max_round, items):
        self.length = length
        self.width = width
        self.max_round = max_round
        self.current_round = 0
        self.current_player = 1
        self.wall_map = [[-1] * width for _ in range(length)]
        self.snake_map = [[-1] * width for _ in range(length)]
        self.item_map = [[-1] * width for _ in range(length)]
        self.item_list = list(items)
        self.snake_list_0 = [Snake([(0, width - 1)], 0, 0, 0, -1)]
        self.snake_list_1 = [Snake([(length - 1, 0)], 1, 0, 1, -1)]
        self.snake_map[0][width - 1] = 0
        self.snake_map[length - 1][0] = 1
        self.tmp_list_0 = []
        self.tmp_list_1 = []
        self.current_snake_id = 0
        self.next_snake_id = 2
        self.new_snakes = []
        self.remove_snakes = []
        self.round_preprocess()

    def my_snakes(self):
        return self.snake_list_0 if self.current_player == 0 else self.snake_list_1

    def opponents_snakes(self):
        return self.snake_list_1 if self.current_player == 0 else self.snake_list_0

    def tmp_my_snakes(self):
        return self.tmp_list_0 if self.current_player == 0 else self.tmp_list_1

    def find_item(self, item_id):
        return self.item_list[item_id]

    def find_snake(self, snake_id):
        for s in self.snake_list_0:
            if s.id == snake_id:
                return s
        for s in self.snake_list_1:
            if s.id == snake_id:
                return s
        raise KeyError(snake_id)

    def current_snake(self):
        for s in self.my_snakes():
            if s.id == self.current_snake_id:
                return s
        return self.my_snakes()[0]

    def alive_player(self):
        """True if the current_player actually has snakes to operate."""
        return len(self.my_snakes()) > 0

    def do_operation(self, op_type):
        if not self.my_snakes():
            # current player has no snakes: just advance the turn (no-op move),
            # mirroring the official judge which skips a move read entirely.
            return (not self.find_next_snake()) or self.round_preprocess()
        if op_type == 5:
            if not self.fire_railgun():
                return False
        elif op_type == 6:
            if not self.split_snake():
                return False
        elif 1 <= op_type <= 4:
            if not self.move_snake(op_type):
                return False
        else:
            return False
        return (not self.find_next_snake()) or self.round_preprocess()

    def move_snake(self, op_type):
        snake = self.current_snake()
        cl = snake.coord_list
        hx, hy = cl[0]
        nx = hx + MOVE_DX[op_type]
        ny = hy + MOVE_DY[op_type]

        # Validate U-turn BEFORE shrinking the tail.  The official judge
        # (operate.py move()) checks the reversal condition on the ORIGINAL
        # body, then raises PlayerError(IA) which rejects the operation
        # entirely -- the snake is NOT modified.  We must do the same:
        # if this is an illegal reversal, return True (turn consumed, no
        # body change) without having popped the tail.
        auto_grow = (self.current_round <= GROWING_ROUNDS
                     and snake.id == snake.camp)
        if (len(cl) > 2 or (len(cl) == 2 and (auto_grow or snake.length_bank))) \
                and len(cl) >= 2 and (nx, ny) == cl[1]:
            return True

        if self.current_round > GROWING_ROUNDS or snake.id != snake.camp:
            if snake.length_bank > 0:
                snake.length_bank -= 1
            else:
                tx, ty = cl[-1]
                self.snake_map[tx][ty] = -1
                cl.pop()
                if not cl:
                    # body emptied by shrinkage; treat as dead
                    self.remove_snake(snake.id)
                    return True
        dead = False
        sealed = False
        if nx < 0 or ny < 0 or nx >= self.length or ny >= self.width or self.wall_map[nx][ny] != -1:
            dead = True
        elif self.snake_map[nx][ny] == snake.id:
            if len(cl) >= 2 and (nx, ny) == cl[1]:
                return True
            cl.insert(0, (nx, ny))
            sealed = True
        elif self.snake_map[nx][ny] != -1:
            dead = True
        if dead:
            self.remove_snake(snake.id)
        elif sealed:
            self.seal_region()
        else:
            cl.insert(0, (nx, ny))
            self.snake_map[nx][ny] = snake.id
            iid = self.item_map[nx][ny]
            if iid != -1:
                item = self.find_item(iid)
                item.eaten = True
                if item.type == 0:
                    snake.length_bank += item.param
                elif item.type == 2:
                    snake.railgun_item_id = item.id
                else:
                    return False
                self.item_map[nx][ny] = -1
        return True

    def remove_snake(self, snake_id):
        for lst in (self.snake_list_0, self.snake_list_1):
            for i, s in enumerate(lst):
                if s.id == snake_id:
                    for (cx, cy) in s.coord_list:
                        self.snake_map[cx][cy] = -1
                    self.remove_snakes.append(snake_id)
                    del lst[i]
                    return

    def flood_fill(self, grid, x, y, v, dir_ok):
        q = deque()
        q.append((x, y))
        L, W = self.length, self.width
        while q:
            cx, cy = q.popleft()
            if cx < 0 or cx >= L or cy < 0 or cy >= W:
                dir_ok[v] = False
                continue
            if grid[cx][cy] != 0:
                continue
            grid[cx][cy] = v
            q.append((cx + 1, cy))
            q.append((cx - 1, cy))
            q.append((cx, cy + 1))
            q.append((cx, cy - 1))

    def seal_region(self):
        snake = self.current_snake()
        cl = snake.coord_list
        L, W = self.length, self.width
        grid = [[0] * W for _ in range(L)]
        x0, y0 = cl[0]
        is_head = True
        seg_len = 0
        for (cx, cy) in cl:
            if cx == x0 and cy == y0 and not is_head:
                break
            grid[cx][cy] = 3
            is_head = False
            seg_len += 1
        dir_ok = [False, True, True, True]
        for i in range(seg_len):
            ix, iy = cl[i]
            jx, jy = cl[(i + 1) % seg_len]
            if ix == jx:
                dir1 = 2 if iy > jy else 0
            else:
                dir1 = 1 if ix > jx else 3
            dir2 = (dir1 + 2) % 4
            self.flood_fill(grid, ix + SEAL_DX[dir1], iy + SEAL_DY[dir1], 1, dir_ok)
            self.flood_fill(grid, ix + SEAL_DX[dir2], iy + SEAL_DY[dir2], 2, dir_ok)
        for i in range(L):
            for j in range(W):
                if dir_ok[grid[i][j]]:
                    self.wall_map[i][j] = self.current_player
                    sid = self.snake_map[i][j]
                    if sid != -1:
                        self.remove_snake(sid)

    def fire_railgun(self):
        snake = self.current_snake()
        if snake.railgun_item_id < 0:
            return False
        cl = snake.coord_list
        if len(cl) < 2:
            return False
        cx, cy = cl[0]
        dx = cl[0][0] - cl[1][0]
        dy = cl[0][1] - cl[1][1]
        while 0 <= cx < self.length and 0 <= cy < self.width:
            self.wall_map[cx][cy] = -1
            cx += dx
            cy += dy
        snake.railgun_item_id = -1
        return True

    def split_snake(self):
        if len(self.my_snakes()) == SNAKE_LIMIT:
            return False
        snake = self.current_snake()
        cl = snake.coord_list
        if len(cl) < 2:
            return False
        mid = (len(cl) + 1) // 2
        cl_head = cl[:mid]
        cl_tail = list(reversed(cl[mid:]))
        new_id = self.next_snake_id
        self.next_snake_id += 1
        new_snake = Snake(cl_tail, new_id, snake.length_bank, snake.camp, -1)
        snake.coord_list = cl_head
        snake.length_bank = 0
        ms = self.my_snakes()
        idx = 0
        for i, s in enumerate(ms):
            if s.id == self.current_snake_id:
                idx = i
                break
        ms.insert(idx + 1, new_snake)
        self.new_snakes.append(new_id)
        for (cx, cy) in cl_tail:
            self.snake_map[cx][cy] = new_id
        return True

    def find_next_snake(self):
        tmp = self.tmp_my_snakes()
        flag = False
        for s in tmp:
            if s == self.current_snake_id:
                flag = True
                continue
            if flag:
                invalid = False
                for ns in self.remove_snakes:
                    if s == ns:
                        invalid = True
                        break
                if invalid:
                    continue
                self.current_snake_id = s
                return False
        if not self.opponents_snakes():
            if self.my_snakes():
                self.current_snake_id = self.my_snakes()[0].id
        else:
            self.current_snake_id = self.opponents_snakes()[0].id
        return True

    def round_preprocess(self):
        self.remove_snakes = []
        if not self.snake_list_0 and not self.snake_list_1:
            return False
        self.new_snakes = []
        self.current_player = 1 - self.current_player
        if self.current_player == 0 or (self.current_player == 1 and not self.my_snakes()):
            self.current_round += 1
            if self.current_round > self.max_round:
                return False
            for i in range(self.length):
                for j in range(self.width):
                    iid = self.item_map[i][j]
                    if iid == -1:
                        continue
                    item = self.find_item(iid)
                    if self.current_round >= item.time + ITEM_EXPIRE_LIMIT:
                        self.item_map[i][j] = -1
                        item.expired = True
            for item in self.item_list:
                if item.time == self.current_round:
                    sid = self.snake_map[item.x][item.y]
                    if sid == -1:
                        self.item_map[item.x][item.y] = item.id
                    else:
                        item.eaten = True
                        if item.type == 0:
                            self.find_snake(sid).length_bank += item.param
                        elif item.type == 2:
                            self.find_snake(sid).railgun_item_id = item.id
                        else:
                            return False
        self.tmp_list_0 = [s.id for s in self.snake_list_0]
        self.tmp_list_1 = [s.id for s in self.snake_list_1]
        if not self.my_snakes():
            self.current_player = 1 - self.current_player
        return True

    def score(self):
        s = [0, 0]
        for x in range(self.length):
            for y in range(self.width):
                sid = self.snake_map[x][y]
                if sid != -1:
                    snake = self.find_snake(sid)
                    s[snake.camp] += 2
                elif self.wall_map[x][y] == 0:
                    s[0] += 2
                elif self.wall_map[x][y] == 1:
                    s[1] += 2
        return s

    def snapshot(self):
        return {
            "round": self.current_round,
            "player": self.current_player,
            "snake_id": self.current_snake_id,
            "wall": [row[:] for row in self.wall_map],
            "snake": [row[:] for row in self.snake_map],
            "item": [row[:] for row in self.item_map],
            "snakes": {
                "p0": [[[c[0], c[1]] for c in s.coord_list] + [[s.length_bank, s.railgun_item_id]] for s in self.snake_list_0],
                "p1": [[[c[0], c[1]] for c in s.coord_list] + [[s.length_bank, s.railgun_item_id]] for s in self.snake_list_1],
            },
            "score": self.score(),
        }
