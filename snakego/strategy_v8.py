# Strategy v8: loop-closing phase machine (GROW -> SPLIT -> HUNT_LOOP -> SEAL).
#
# In SnakeGo the ONLY way to score permanent territory is: head moves onto own
# body (not neck) -> engine.seal_region() flood-fills the interior into walls
# (+2/cell). Crucially, sealing DISSOLVES the sealing snake's body into walls --
# so a lone snake that seals dies with no body left. rank07 (FREDZEL) survives by
# SPLITTING first (up to 4 snakes), so one snake can seal/sacrifice while the
# others continue claiming territory. v0-v7 never split-then-sealed, which is
# why they sealed at most once and scored nothing.
#
# PHASES (structural, not weight tweaks)
#   SPLIT      once a snake is long enough and we have <4 snakes, split so there
#              is always a backup. This is the precondition for repeatable sealing.
#   GROW       build length by extending in straight lines toward open space +
#              growth items. Straight-line growth lets a later loop be large.
#   HUNT_LOOP  once long enough, BFS toward the TAIL REGION of the body (back
#              half). Closing onto a far-back segment makes a large ring.
#   SEAL       a move touches own body AND the ring encloses >= SEAL_MIN cells:
#              take it. The sealing snake dissolves into walls; the others
#              continue, so we keep scoring.
from collections import deque

from snakego import board as B
from snakego.strategy_core import Weights, score_action
from snakego.decision_space import compute_mask

SPLIT_LEN = 10     # split once a snake reaches this length (if <4 snakes)
SEAL_MIN = 6       # only seal rings that enclose >= this many cells
HUNT_LEN = 16      # minimum length to start hunting a large loop
DIRS = {1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0, -1)}


def seal_area_if_move(eng, snake, op):
    if not (1 <= op <= 4):
        return 0
    nx, ny = B.head_after(snake, op)
    if not (0 <= nx < eng.length and 0 <= ny < eng.width):
        return 0
    if eng.snake_map[nx][ny] != snake.id:
        return 0
    cl = snake.coord_list
    if len(cl) >= 2 and (nx, ny) == cl[1]:
        return 0
    growing = eng.current_round <= 8 or snake.id != snake.camp or snake.length_bank > 0
    body = [(nx, ny)] + list(cl)
    if not growing:
        body = body[:-1]
    ring_end = None
    for i in range(1, len(body)):
        if body[i] == (nx, ny):
            ring_end = i
            break
    if ring_end is None or ring_end < 3:
        return 0
    ring_set = set(body[: ring_end + 1])
    L, W = eng.length, eng.width
    outside = [[False] * W for _ in range(L)]
    q = deque()
    for x in range(L):
        for y in (0, W - 1):
            if (x, y) not in ring_set and not outside[x][y] and eng.wall_map[x][y] == -1:
                outside[x][y] = True
                q.append((x, y))
    for y in range(W):
        for x in (0, L - 1):
            if (x, y) not in ring_set and not outside[x][y] and eng.wall_map[x][y] == -1:
                outside[x][y] = True
                q.append((x, y))
    while q:
        cx, cy = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ax, ay = cx + dx, cy + dy
            if not (0 <= ax < L and 0 <= ay < W) or outside[ax][ay]:
                continue
            if (ax, ay) in ring_set or eng.wall_map[ax][ay] != -1:
                continue
            outside[ax][ay] = True
            q.append((ax, ay))
    enclosed = 0
    for x in range(L):
        for y in range(W):
            if not outside[x][y] and (x, y) not in ring_set and eng.wall_map[x][y] == -1:
                enclosed += 1
    return enclosed


def best_seal_move(eng, snake):
    mask = compute_mask(eng)
    best = None
    for op in (1, 2, 3, 4):
        if not mask.legal[op]:
            continue
        if B.classify_move(eng, snake, op) != "seal":
            continue
        area = seal_area_if_move(eng, snake, op)
        if area >= SEAL_MIN and (best is None or area > best[1]):
            best = (op, area)
    return best


def _should_split(eng, snake):
    # Split if: long enough, <4 snakes, enough room, and we have a spare slot.
    # Splitting is the precondition for repeatable sealing (the sealing snake
    # dissolves, so we need backups).
    n_mine = len(eng.my_snakes())
    if n_mine >= 4 or snake.length < SPLIT_LEN:
        return False
    mask = compute_mask(eng)
    if not mask.legal[6]:
        return False
    my_ids = {s.id for s in eng.my_snakes()}
    hx, hy = snake.coord_list[0]
    room = B.reachable_space(eng, hx, hy, my_ids)
    return room >= snake.length + 8


def _grow_move(eng, snake):
    # Build a long arm: prefer straight-ahead, then open space + items, with a
    # strong survival margin AND enemy-head avoidance (the v7 weighted scorer's
    # enemy_proximity term) so we dont walk into the opponent.
    mask = compute_mask(eng)
    my_ids = {s.id for s in eng.my_snakes()}
    cl = snake.coord_list
    head = cl[0]
    cur_dir = None
    if len(cl) >= 2:
        ddx, ddy = head[0] - cl[1][0], head[1] - cl[1][1]
        for d, (dx, dy) in DIRS.items():
            if (dx, dy) == (ddx, ddy):
                cur_dir = d
                break
    enemy_heads = [s.coord_list[0] for s in eng.opponents_snakes() if s.coord_list]
    cands = []
    for op in (1, 2, 3, 4):
        if not mask.legal[op]:
            continue
        kind = B.classify_move(eng, snake, op)
        if kind == "dead":
            continue
        nx, ny = B.head_after(snake, op)
        if not (0 <= nx < eng.length and 0 <= ny < eng.width):
            continue
        room = B.reachable_space(eng, nx, ny, my_ids)
        if room < snake.length + 4:
            continue  # keep a real margin; dont edge into death
        # penalize stepping near an enemy head (collision risk)
        near_enemy = sum(1 for eh in enemy_heads if B.manhattan(nx, ny, eh[0], eh[1]) <= 2)
        if near_enemy:
            continue
        item, item_d = B.nearest_live_item(eng, head[0], head[1], eng.current_round)
        nd = B.manhattan(nx, ny, item.x, item.y) if item else 0
        straight_bonus = 3.0 if op == cur_dir else 0.0
        item_bonus = (item_d - nd) * 2.0 if item else 0.0
        score = room + straight_bonus + item_bonus
        cands.append((score, op))
    if not cands:
        # tight spot: fall back to the battle-tested v7 scorer which has
        # full survival + enemy-proximity logic built in.
        w = Weights()
        best = None
        best_op = 1
        for op in (1, 2, 3, 4, 5, 6):
            if not mask.legal[op]:
                continue
            dt = score_action(eng, op, w)
            if best is None or dt.total > best:
                best = dt.total
                best_op = op
        return best_op
    cands.sort(reverse=True)
    return cands[0][1]


def hunt_loop_move(eng, snake):
    cl = snake.coord_list
    if snake.length < HUNT_LEN:
        return None
    head = cl[0]
    neck = cl[1] if len(cl) >= 2 else None
    half = len(cl) // 2
    target_body = set()
    for i in range(half, len(cl)):
        target_body.add(cl[i])
    if not target_body:
        return None
    L, W = eng.length, eng.width
    seen = [[False] * W for _ in range(L)]
    seen[head[0]][head[1]] = True
    q = deque()
    for d in (1, 2, 3, 4):
        nx, ny = head[0] + DIRS[d][0], head[1] + DIRS[d][1]
        if not (0 <= nx < L and 0 <= ny < W):
            continue
        if (nx, ny) == neck:
            continue
        if eng.wall_map[nx][ny] != -1:
            continue
        occ = eng.snake_map[nx][ny]
        if occ != -1 and occ != snake.id:
            continue
        if not seen[nx][ny]:
            seen[nx][ny] = True
            q.append((nx, ny, d))
    radius = 0
    while q and radius < 80:
        cx, cy, first_d = q.popleft()
        radius += 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ax, ay = cx + dx, cy + dy
            if (ax, ay) in target_body:
                return first_d
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ax, ay = cx + dx, cy + dy
            if not (0 <= ax < L and 0 <= ay < W) or seen[ax][ay]:
                continue
            if eng.wall_map[ax][ay] != -1:
                continue
            occ = eng.snake_map[ax][ay]
            if occ != -1 and occ != snake.id:
                continue
            seen[ax][ay] = True
            q.append((ax, ay, first_d))
    return None


def _survive_move(eng, snake):
    mask = compute_mask(eng)
    my_ids = {s.id for s in eng.my_snakes()}
    best = None
    for op in (1, 2, 3, 4):
        if not mask.legal[op]:
            continue
        nx, ny = B.head_after(snake, op)
        if not (0 <= nx < eng.length and 0 <= ny < eng.width):
            continue
        room = B.reachable_space(eng, nx, ny, my_ids)
        if best is None or room > best[1]:
            best = (op, room)
    if best:
        return best[0]
    from snakego.board import is_reversal as _ir2
    return next((c for c in (1,2,3,4) if not _ir2(snake, c)), 1)


def decide(eng, my_id, trace=False):
    snake = eng.current_snake()
    if snake is None or not snake.coord_list:
        return 1
    mask = compute_mask(eng)

    # PHASE: SEAL -- the scoring move. Absolute priority. The sealing snake
    # dissolves into walls, but we split beforehand so others continue.
    seal = best_seal_move(eng, snake)
    if seal is not None:
        if trace:
            return seal[0], {"phase": "SEAL", "area": seal[1]}
        return seal[0]

    # PHASE: SPLIT -- ensure we have backup snakes before we start sealing.
    # Without this, the first seal kills our only snake.
    if _should_split(eng, snake):
        if trace:
            return 6, {"phase": "SPLIT"}
        return 6

    # PHASE: HUNT_LOOP -- head back toward the tail-region body to form a ring.
    if snake.length >= HUNT_LEN:
        hunt = hunt_loop_move(eng, snake)
        if hunt is not None and mask.legal[hunt]:
            my_ids = {s.id for s in eng.my_snakes()}
            nx, ny = B.head_after(snake, hunt)
            room = B.reachable_space(eng, nx, ny, my_ids)
            if room >= snake.length:
                if trace:
                    return hunt, {"phase": "HUNT_LOOP", "room": room}
                return hunt

    # PHASE: GROW -- extend the arm in straight lines.
    op = _grow_move(eng, snake)
    if trace:
        return op, {"phase": "GROW"}
    return op


def make_decide():
    def _d(eng, my_id):
        return decide(eng, my_id)
    _d.__name__ = "strategy_v8"
    return _d


