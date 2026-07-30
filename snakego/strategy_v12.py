# Strategy v12: v9 + danger-recycle sealing (single structural change).
#
# The v9 -> v12 ablations showed that changing GROW (item weight, hard room
# gate) or the global seal threshold breaks v9's board-specific strengths:
# seed 23 needs frequent productive sealing (v9 seals 10x there and wins),
# while seeds 1/7 reward long survival. Sweeping changes regress either side.
#
# So v12 makes ONE structural change on top of v9, targeted at the single
# behaviour that is unambiguously wasteful: a snake that is about to be trapped
# DIES and scores nothing, when it could SEAL whatever loop it can close and
# convert its body+interior into permanent walls.
#
# THE CHANGE (danger-recycle seal):
#   When a snake's escape room falls below DANGER_ROOM, lower the seal threshold
#   to 1 -- seal ANY closeable loop immediately. This is pure salvage: a doomed
#   snake turning into walls is strictly better than a doomed snake dying empty.
#   In the endgame (round>=400) do the same (an unsealed loop scores nothing at
#   game end). Everything else is inherited verbatim from v9.
import copy as _copy

from snakego import strategy_v9 as _v9
from snakego import board as B
from snakego.decision_space import compute_mask
from snakego.strategy_v8 import seal_area_if_move

DANGER_ROOM = 14
ENDGAME_START = 400
SALVAGE_SEAL = 1        # when doomed/endgame: seal any loop with >=1 interior


def _salvage_seal_move(eng, snake):
    """Seal the largest closeable loop; used only when doomed or in endgame."""
    mask = compute_mask(eng)
    best = None
    for op in (1, 2, 3, 4):
        if not mask.legal[op]:
            continue
        if B.classify_move(eng, snake, op) != "seal":
            continue
        area = seal_area_if_move(eng, snake, op)
        if area >= SALVAGE_SEAL and (best is None or area > best[1]):
            best = (op, area)
    return best


def _doomed(eng, snake):
    my_ids = {s.id for s in eng.my_snakes()}
    hx, hy = snake.coord_list[0]
    return B.reachable_space(eng, hx, hy, my_ids) < DANGER_ROOM


def decide(eng, my_id, trace=False):
    snake = eng.current_snake()
    if snake is None or not snake.coord_list:
        return 1
    # SINGLE STRUCTURAL ADDITION over v9: salvage-seal when doomed or in endgame.
    if _doomed(eng, snake) or eng.current_round >= ENDGAME_START:
        salv = _salvage_seal_move(eng, snake)
        if salv is not None:
            if trace:
                return salv[0], {"phase": "SALVAGE_SEAL", "area": salv[1],
                                 "doomed": _doomed(eng, snake)}
            return salv[0]
    # Everything else is v9 unchanged.
    return _v9.decide(eng, my_id, trace=trace)


def make_decide():
    def _d(eng, my_id):
        return decide(eng, my_id)
    _d.__name__ = "strategy_v12"
    return _d
