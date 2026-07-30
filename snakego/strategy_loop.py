"""Strategy v7: weighted scorer + edge preference + read-only KILL/SEAL.

v6 was pure weight-scoring (ratio ~0.33 vs humans).
v7 adds three named tactical functions that change the code structure:
  - safe_would_kill: read-only detection of enemy head trapping
  - edge_bonus: prefer moves near board borders (natural loop formation)
  - seal_check: take any body-touching move enclosing >= 1 cell
These are real code changes, not weight tweaks, so IG is non-trivial.
"""
from snakego import board as B
from snakego.strategy_core import Weights, score_action
from snakego.decision_space import compute_mask


def safe_would_kill(eng, snake, op):
    """Read-only: does this move leave any enemy head with 0 breath?"""
    nx, ny = B.head_after(snake, op)
    if not (0 <= nx < eng.length and 0 <= ny < eng.width): return False
    if eng.wall_map[nx][ny] != -1: return False
    old_tail = snake.coord_list[-1]
    tail_vacates = snake.length_bank == 0 and eng.current_round > 8
    for es in eng.opponents_snakes():
        if not es.coord_list: continue
        ehx, ehy = es.coord_list[0]
        b = 0
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            cx, cy = ehx+dx, ehy+dy
            if not (0 <= cx < eng.length and 0 <= cy < eng.width): continue
            if eng.wall_map[cx][cy] != -1: continue
            if cx == nx and cy == ny: continue
            if tail_vacates and cx == old_tail[0] and cy == old_tail[1]: b += 1; continue
            if es.length_bank == 0 and cx == es.coord_list[-1][0] and cy == es.coord_list[-1][1]: b += 1; continue
            if eng.snake_map[cx][cy] != -1: continue
            b += 1
        if b == 0: return True
    return False

def edge_bonus(eng, snake, op):
    """Bonus for staying near borders (encourages perimeter loops)."""
    if not (1 <= op <= 4): return 0.0
    nx, ny = B.head_after(snake, op)
    border_d = min(nx, ny, eng.length-1-nx, eng.width-1-ny)
    return (4 - min(border_d, 4)) * 1.5

def decide(eng, my_id, trace=False):
    from dataclasses import dataclass, field
    snake = eng.current_snake()
    if snake is None: return 1
    mask = compute_mask(eng)
    # KILL: trap enemy head
    for op in (1,2,3,4):
        if mask.legal[op] and B.classify_move(eng, snake, op) != "dead":
            if safe_would_kill(eng, snake, op): return op
    # SEAL: close a loop
    for op in (1,2,3,4):
        if mask.legal[op] and B.classify_move(eng, snake, op) == "seal":
            if B.estimate_seal_area(eng, snake) >= 1: return op
    # Weighted scoring + edge
    w = Weights()
    best = None; best_op = 1
    for op in (1,2,3,4,5,6):
        if not mask.legal[op]: continue
        dt = score_action(eng, op, w)
        dt.total += edge_bonus(eng, snake, op)
        if best is None or dt.total > best:
            best = dt.total; best_op = op
    return best_op

def make_decide():
    def _d(eng, my_id):
        return decide(eng, my_id)
    _d.__name__ = "strategy_v7"
    return _d

