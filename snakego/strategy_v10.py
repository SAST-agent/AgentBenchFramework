# Strategy v10: survive-to-territory (structural combination of v8 + v9).
#
# DIAGNOSIS OF v9 vs v8 (runs/v9_sandbox.json):
#   v9 fixed v8's early-death bug (seed1 0.097->0.429, seed7 0.020->0.351)
#   but LOST sealing aggressiveness (seed2 0.484->0.125, seed23 0.821->0.684).
#   Net avg flat (0.418 -> 0.408). The survival fix was right, but v9 over-
#   corrected into pure space-maximising growth and never switched back to
#   territory mode. It survives but does not seal enough to win.
#
# v10 STRUCTURAL CHANGE (decision architecture, not weight tweaks):
#   Adds a TERRITORY gate: ONCE a backup snake exists (>=2 snakes), the longest
#   snake switches from survival-growth to AGGRESSIVE loop-closing -- it lowers
#   the hunt threshold and actively curves back toward its own body to form
#   rings. The backup snake carries insurance so the territory snake can
#   sacrifice itself on a seal and still leave territory behind.
#
#   Decision order: SEAL > SPLIT(if <2 snakes) > HUNT/TERRITORY(if >=2 snakes)
#                   > GROW(survival-first)
#   This is "survive until you have insurance, then play for territory."
from collections import deque

from snakego import board as B
from snakego.decision_space import compute_mask
from snakego.strategy_v8 import best_seal_move, hunt_loop_move
from snakego.strategy_v9 import _grow_move, _should_split, _survive_move

HUNT_LEN_TERRITORY = 10   # aggressive hunt once we have a backup (v8 used 16)
HUNT_LEN_SOLO = 16        # conservative hunt when we have no backup yet


def _territory_hunt(eng, snake, mask):
    # Aggressive loop-closing: hunt toward the back-half body to form a ring,
    # but only accept moves that keep enough room to not die instantly.
    threshold = HUNT_LEN_TERRITORY if len(eng.my_snakes()) >= 2 else HUNT_LEN_SOLO
    if snake.length < threshold:
        return None
    hunt = hunt_loop_move(eng, snake)
    if hunt is None or not mask.legal[hunt]:
        return None
    my_ids = {s.id for s in eng.my_snakes()}
    nx, ny = B.head_after(snake, hunt)
    room = B.reachable_space(eng, nx, ny, my_ids)
    # need room to retreat if the ring does not close; keep a real margin
    needed = snake.length if len(eng.my_snakes()) >= 2 else snake.length + 4
    if room >= needed:
        return hunt
    return None


def decide(eng, my_id, trace=False):
    snake = eng.current_snake()
    if snake is None or not snake.coord_list:
        return 1
    mask = compute_mask(eng)
    n_mine = len(eng.my_snakes())

    # PHASE: SEAL -- scoring move, absolute priority.
    seal = best_seal_move(eng, snake)
    if seal is not None:
        if trace:
            return seal[0], {"phase": "SEAL", "area": seal[1], "snakes": n_mine}
        return seal[0]

    # PHASE: SPLIT -- insurance first. Keep splitting until we have a backup.
    if n_mine < 2 and _should_split(eng, snake):
        if trace:
            return 6, {"phase": "SPLIT", "snakes": n_mine}
        return 6

    # PHASE: TERRITORY -- once insured (>=2 snakes), hunt aggressively. This is
    # the structural addition over v9: survival-bought insurance is SPENT on
    # territory instead of hoarded.
    if n_mine >= 2:
        hunt = _territory_hunt(eng, snake, mask)
        if hunt is not None:
            if trace:
                return hunt, {"phase": "TERRITORY", "snakes": n_mine}
            return hunt

    # PHASE: GROW -- survival-first growth (inherited from v9).
    op = _grow_move(eng, snake)
    if trace:
        return op, {"phase": "GROW", "snakes": n_mine}
    return op


def make_decide():
    def _d(eng, my_id):
        return decide(eng, my_id)
    _d.__name__ = "strategy_v10"
    return _d
