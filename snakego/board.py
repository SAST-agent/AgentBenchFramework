"""Board-analysis helpers shared by all my strategy versions.

Keeps strategy code readable: flood-fill space estimation, move legality
classification, seal-area estimation, nearest-item search. These mirror the
adk.hpp engine semantics so strategy decisions are consistent with the engine.
"""
from collections import deque

# direction id -> (dx, dy), index 0 unused
DIRS = {1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0, -1)}


def head_after(snake, d):
    dx, dy = DIRS[d]
    return snake.coord_list[0][0] + dx, snake.coord_list[0][1] + dy


def is_reversal(snake, d):
    """True if d points back onto the neck (illegal 180)."""
    if snake.length < 2:
        return False
    hx, hy = snake.coord_list[0]
    nx, ny = head_after(snake, d)
    return (nx, ny) == snake.coord_list[1]


def classify_move(eng, snake, d):
    """Classify a candidate move for the given snake.

    Returns one of:
      'dead'    -> moving kills the snake (wall / boundary / enemy / illegal)
      'seal'    -> head hits own body, forms a loop, triggers region seal
      'free'    -> normal move into empty space (may pick up an item)
    """
    if is_reversal(snake, d):
        return "dead"
    nx, ny = head_after(snake, d)
    if nx < 0 or ny < 0 or nx >= eng.length or ny >= eng.width:
        return "dead"
    if eng.wall_map[nx][ny] != -1:
        return "dead"
    occ = eng.snake_map[nx][ny]
    if occ != -1:
        if occ == snake.id:
            return "seal"  # own body -> closes a loop
        # Enemy body. The enemy tail will move next tick, so a tail cell may be
        # safe, but distinguishing that is brittle; treat enemy body as dead.
        return "dead"
    return "free"


def legal_moves(eng, snake):
    """Return list of (d, kind) for non-dead moves."""
    out = []
    for d in (1, 2, 3, 4):
        kind = classify_move(eng, snake, d)
        if kind != "dead":
            out.append((d, kind))
    return out


def reachable_space(eng, start_x, start_y, my_snake_ids, limit=400):
    """BFS count of free cells reachable from (start_x,start_y).

    own_body_walkable=True (legacy) treats our own snake bodies as walkable,
    which over-estimates room and is how v1-v3 trapped themselves. The default
    (False) treats ALL snake bodies as obstacles, giving a safe, realistic lower
    bound on escape room; this is what v4+ keep-alive uses.
    """
    return _reachable(eng, start_x, start_y, my_snake_ids, limit, own_body_walkable=False)


def reachable_space_lenient(eng, start_x, start_y, my_snake_ids, limit=400):
    """Old over-optimistic estimate; kept for comparison only."""
    return _reachable(eng, start_x, start_y, my_snake_ids, limit, own_body_walkable=True)


def _reachable(eng, start_x, start_y, my_snake_ids, limit, own_body_walkable):
    if not (0 <= start_x < eng.length and 0 <= start_y < eng.width):
        return 0
    if eng.wall_map[start_x][start_y] != -1:
        return 0
    seen = [[False] * eng.width for _ in range(eng.length)]
    seen[start_x][start_y] = True
    q = deque([(start_x, start_y)])
    count = 0
    while q and count < limit:
        cx, cy = q.popleft()
        count += 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < eng.length and 0 <= ny < eng.width) or seen[nx][ny]:
                continue
            seen[nx][ny] = True
            if eng.wall_map[nx][ny] != -1:
                continue
            sid = eng.snake_map[nx][ny]
            if sid != -1:
                if own_body_walkable and sid in my_snake_ids:
                    pass
                else:
                    continue
            q.append((nx, ny))
    return count


def estimate_seal_area(eng, snake):
    """Estimate how many cells would be sealed if the snake closed a loop now.

    Approximates by counting interior empty cells enclosed by the snake body via
    a flood fill from outside; the complement within the bounding box is the
    candidate sealed region. Good enough for ranking candidate seal moves.
    """
    L, W = eng.length, eng.width
    body = set(snake.coord_list)
    outside = [[False] * W for _ in range(L)]
    q = deque()
    # seed from the border
    for x in range(L):
        for y in (0, W - 1):
            if (x, y) not in body and not outside[x][y]:
                outside[x][y] = True
                q.append((x, y))
    for y in range(W):
        for x in (0, L - 1):
            if (x, y) not in body and not outside[x][y]:
                outside[x][y] = True
                q.append((x, y))
    while q:
        cx, cy = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < L and 0 <= ny < W) or outside[nx][ny]:
                continue
            if (nx, ny) in body:
                continue
            outside[nx][ny] = True
            q.append((nx, ny))
    sealed = 0
    for x in range(L):
        for y in range(W):
            if not outside[x][y] and (x, y) not in body:
                sealed += 1
    return sealed


def manhattan(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def nearest_live_item(eng, x, y, current_round):
    """Closest reachable item by Manhattan distance (growth items only)."""
    best = None
    best_d = 10 ** 9
    for it in eng.item_list:
        if it.eaten or it.expired:
            continue
        if eng.item_map[it.x][it.y] == -1:
            continue
        if it.type != 0:
            continue
        d = manhattan(x, y, it.x, it.y)
        # keep even if it expires before we arrive -- the direction toward the
        # spawn area is still useful because new items spawn in the same zone
        if d < best_d:
            best_d = d
            best = it
    return best, best_d


def estimate_loop_potential(eng, snake, nx, ny):
    if snake.length < 4:
        return 0
    body_list = snake.coord_list
    min_gap = 999
    for i, (bx, by) in enumerate(body_list):
        if i < 2:
            continue
        d = abs(nx - bx) + abs(ny - by)
        if d < min_gap:
            min_gap = d
    if min_gap > 3:
        return 0
    hyp_body = [(nx, ny)] + body_list[:-1]
    hyp_body_set = set(hyp_body)
    L, W = eng.length, eng.width
    outside = [[False] * W for _ in range(L)]
    q = deque()
    for x in range(L):
        for y in (0, W - 1):
            if (x, y) not in hyp_body_set and not outside[x][y] and eng.wall_map[x][y] == -1 and eng.snake_map[x][y] == -1:
                outside[x][y] = True
                q.append((x, y))
    for y in range(W):
        for x in (0, L - 1):
            if (x, y) not in hyp_body_set and not outside[x][y] and eng.wall_map[x][y] == -1 and eng.snake_map[x][y] == -1:
                outside[x][y] = True
                q.append((x, y))
    while q:
        cx, cy = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxi, nyi = cx + dx, cy + dy
            if not (0 <= nxi < L and 0 <= nyi < W) or outside[nxi][nyi]:
                continue
            if (nxi, nyi) in hyp_body_set:
                continue
            if eng.wall_map[nxi][nyi] != -1:
                continue
            if eng.snake_map[nxi][nyi] != -1:
                continue
            outside[nxi][nyi] = True
            q.append((nxi, nyi))
    enclosed = 0
    for x in range(L):
        for y in range(W):
            if not outside[x][y] and (x, y) not in hyp_body_set and eng.wall_map[x][y] == -1 and eng.snake_map[x][y] == -1:
                enclosed += 1
    if enclosed < 2:
        return 0
    return enclosed * (4 - min_gap) / 3.0


def territory_contest_area(eng, nx, ny):
    L, W = eng.length, eng.width
    if not (0 <= nx < L and 0 <= ny < W):
        return 0
    count = 0
    for dx in range(-4, 5):
        for dy in range(-4, 5):
            x, y = nx + dx, ny + dy
            if 0 <= x < L and 0 <= y < W:
                if eng.wall_map[x][y] == -1 and eng.snake_map[x][y] == -1:
                    d = abs(dx) + abs(dy)
                    if d <= 4:
                        count += (5 - d)
    return count
