# Strategy v9: survive-and-multiply (structural HL iteration over v8).
#
# DIAGNOSIS OF v8 (from runs/v8_sandbox.json):
#   v8 is bimodal. On open boards it survives >400 rounds, splits, hunts and
#   seals repeatedly (seed 23: 9 seals, 156 walls). On contested boards it dies
#   in <25 rounds (seed 1: ratio 0.097; seed 7: ratio 0.020). Root cause: v8's
#   GROW phase hard-rejects EVERY move within Manhattan-2 of an enemy head, and
#   also rejects any move with room < length+4. When the board is tight BOTH
#   filters empty the candidate set, so it falls back to the v7 *territory*
#   weighted scorer -- which optimises for sealing/items, not survival, and the
#   snake walks into a wall and dies before it ever splits or seals.
#
# v9 STRUCTURAL CHANGES (decision architecture, not weight tweaks):
#   1. EARLY-SPLIT INSURANCE. Split as soon as a snake is long enough and there
#      is room, targeting >=2 snakes quickly. A backup snake is the only real
#      insurance against an early death.
#   2. SPACE-MAXIMISING GROW. GROW scores each legal move by the flood-fill free
#      space it leaves, with straight-line and item tiebreaks. Enemy-head
#      proximity is a SOFT penalty, never a hard reject.
#   3. NEVER FALL BACK TO A TERRITORY SCORER. If no growth move clears the
#      survival margin, v9 degrades to PURE space-maximisation -- the most
#      survivable legal move -- instead of the v7 territory scorer that caused
#      the early deaths. Territory only matters if you are still alive.
#   SEAL / HUNT_LOOP are inherited unchanged from v8.
from collections import deque

from snakego import board as B
from snakego.decision_space import compute_mask
from snakego.strategy_v8 import best_seal_move, hunt_loop_move

SPLIT_LEN = 8
HUNT_LEN = 16
GROW_MARGIN = 6
DIRS = {1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0, -1)}


def _should_split(eng, snake):
    # Insurance-first: split at the first safe opportunity so a backup exists.
    n_mine = len(eng.my_snakes())
    if n_mine >= 3 or snake.length < SPLIT_LEN:
        return False
    mask = compute_mask(eng)
    if not mask.legal[6]:
        return False
    my_ids = {s.id for s in eng.my_snakes()}
    hx, hy = snake.coord_list[0]
    room = B.reachable_space(eng, hx, hy, my_ids)
    return room >= snake.length + 6


def _score_grow(eng, snake, op, cur_dir, item, item_d, enemy_heads, my_ids):
    nx, ny = B.head_after(snake, op)
    if not (0 <= nx < eng.length and 0 <= ny < eng.width):
        return None
    room = B.reachable_space(eng, nx, ny, my_ids)
    if room < 2:
        return None
    straight_bonus = 2.0 if op == cur_dir else 0.0
    nd = B.manhattan(nx, ny, item.x, item.y) if item else 0
    item_bonus = (item_d - nd) * 1.5 if item else 0.0
    min_enemy = min((B.manhattan(nx, ny, eh[0], eh[1]) for eh in enemy_heads),
                    default=99)
    enemy_pen = 6.0 if min_enemy <= 1 else (2.0 if min_enemy == 2 else 0.0)
    return room + straight_bonus + item_bonus - enemy_pen, op


def _grow_move(eng, snake):
    # Two-tier growth: greedy where margin holds, else pure space-max survival.
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
    enemy_heads = [s.coord_list[0] for s in eng.opponents_snakes()
                   if s.coord_list]
    item, item_d = B.nearest_live_item(eng, head[0], head[1],
                                       eng.current_round)
    tier_a, tier_b = [], []
    for op in (1, 2, 3, 4):
        if not mask.legal[op]:
            continue
        scored = _score_grow(eng, snake, op, cur_dir, item, item_d,
                             enemy_heads, my_ids)
        if scored is None:
            continue
        score, _ = scored
        nx2, ny2 = B.head_after(snake, op)
        room = B.reachable_space(eng, nx2, ny2, my_ids)
        if room >= snake.length + GROW_MARGIN:
            tier_a.append((score, op))
        tier_b.append((score, op))
    if tier_a:
        tier_a.sort(reverse=True)
        return tier_a[0][1]
    if tier_b:
        tier_b.sort(reverse=True)
        return tier_b[0][1]
    return _survive_move(eng, snake)


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
    seal = best_seal_move(eng, snake)
    if seal is not None:
        if trace:
            return seal[0], {"phase": "SEAL", "area": seal[1]}
        return seal[0]
    if _should_split(eng, snake):
        if trace:
            return 6, {"phase": "SPLIT"}
        return 6
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
    op = _grow_move(eng, snake)
    if trace:
        return op, {"phase": "GROW"}
    return op


def make_decide():
    def _d(eng, my_id):
        return decide(eng, my_id)
    _d.__name__ = "strategy_v9"
    return _d
