# Strategy v11: curve-and-seal (structural iteration over v9).
#
# DIAGNOSIS (runs/v9_sandbox.json + runs/v10_sandbox.json):
#   v9 survives well (seed1/7 worst cases fixed) but seals too little (avg 3.5
#   vs opponent's ~7). The opponent's scoring edge comes from REPEATED sealing,
#   and the opponent achieves that via U-TURN FORCING: after 4+ straight body
#   segments it forces a 90-degree turn, which geometrically creates loops that
#   eventually close into seals. v9 goes in straight space-maximising lines and
#   never curves, so it rarely forms a ring.
#   v10 tried to fix this by aggressively HUNTING loops (lowered threshold to
#   10), but that drove snakes into traps and regressed to 0.284.
#
# v11 STRUCTURAL CHANGE (a named geometric rule, not a hunt, not a weight):
#   CURVE-FORCING in GROW. After the body has been straight for >= CURVE_STRAIGHT
#   segments, and the snake is long enough to matter (>= CURVE_LEN), v11 forces a
#   perpendicular turn -- choosing the 90-degree direction that keeps the most
#   room. This makes the snake trace an L / rectangle path that naturally loops
#   back onto its own body, producing seals WITHOUT the risky hunt that killed
#   v10. Survival stays first: the curve only fires when room >= length + margin.
#
#   Decision order: SEAL > SPLIT(if <2 snakes) > CURVE(in GROW) > GROW(survival)
from collections import deque

from snakego import board as B
from snakego.decision_space import compute_mask
from snakego.strategy_v8 import best_seal_move
from snakego.strategy_v9 import _should_split, _grow_move, _survive_move

CURVE_LEN = 9        # only curve once long enough that a loop can enclose >= SEAL_MIN
CURVE_STRAIGHT = 4   # force a turn after this many colinear segments
CURVE_MARGIN = 4     # minimum room surplus required to divert from growth
DIRS = {1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0, -1)}


def _curve_move(eng, snake, mask, my_ids):
    # CURVE-FORCING: if the body has been straight long enough, turn 90 degrees
    # to build a loop-forming path. Returns op or None.
    if snake.length < CURVE_LEN:
        return None
    cl = snake.coord_list
    if len(cl) < CURVE_STRAIGHT + 1:
        return None
    h0x, h0y = cl[0]
    h1x, h1y = cl[1]
    hkx, hky = cl[CURVE_STRAIGHT]
    dx1, dy1 = h0x - h1x, h0y - h1y
    dxk, dyk = h0x - hkx, h0y - hky
    is_straight = ((dx1 != 0 and dxk != 0 and dx1 * dxk > 0) or
                   (dy1 != 0 and dyk != 0 and dy1 * dyk > 0))
    if not is_straight:
        return None
    head_room = B.reachable_space(eng, h0x, h0y, my_ids)
    if head_room < snake.length + CURVE_MARGIN:
        return None  # too tight to divert; stay on survival growth
    # perpendicular turns relative to current heading
    if dx1 != 0:
        turns = [2, 4]   # moving in x -> turn in y
    else:
        turns = [1, 3]   # moving in y -> turn in x
    best = None
    for op in turns:
        if not mask.legal[op]:
            continue
        kind = B.classify_move(eng, snake, op)
        if kind == "dead":
            continue
        nx, ny = B.head_after(snake, op)
        room = B.reachable_space(eng, nx, ny, my_ids)
        if room < snake.length + 2:
            continue
        if best is None or room > best[1]:
            best = (op, room)
    return best[0] if best else None


def decide(eng, my_id, trace=False):
    snake = eng.current_snake()
    if snake is None or not snake.coord_list:
        return 1
    mask = compute_mask(eng)
    my_ids = {s.id for s in eng.my_snakes()}

    # PHASE: SEAL -- scoring move, absolute priority.
    seal = best_seal_move(eng, snake)
    if seal is not None:
        if trace:
            return seal[0], {"phase": "SEAL", "area": seal[1]}
        return seal[0]

    # PHASE: SPLIT -- insurance first.
    if _should_split(eng, snake):
        if trace:
            return 6, {"phase": "SPLIT"}
        return 6

    # PHASE: CURVE -- geometric loop-forming turn (NEW over v9).
    cv = _curve_move(eng, snake, mask, my_ids)
    if cv is not None:
        if trace:
            return cv, {"phase": "CURVE"}
        return cv

    # PHASE: GROW -- survival-first growth (inherited from v9).
    op = _grow_move(eng, snake)
    if trace:
        return op, {"phase": "GROW"}
    return op


def make_decide():
    def _d(eng, my_id):
        return decide(eng, my_id)
    _d.__name__ = "strategy_v11"
    return _d
