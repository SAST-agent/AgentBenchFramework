"""My iterative if-else strategy family for 26_snakego.

Each version is a pure function decide(eng, my_id) -> op_type in {1..6}.
Versions build on the previous one; the changelog documents what each
iteration learned from human rollout analysis.

Changelog
---------
v1 baseline: survival-first. Legal-move filter, greedily eat growth items,
             otherwise move into the most open space. No intentional sealing.
"""
from snakego import board as B
from snakego.engine import GROWING_ROUNDS


def _my_snake_ids(eng, my_id):
    return {s.id for s in (eng.snake_list_0 if my_id == 0 else eng.snake_list_1)}


def _first_snake(eng, my_id):
    snakes = eng.snake_list_0 if my_id == 0 else eng.snake_list_1
    return snakes[0] if snakes else None


def decide_v1(eng, my_id):
    """Baseline: never die if avoidable, eat growth, maximize open space."""
    snake = eng.current_snake()
    if snake is None:
        return 1
    my_ids = _my_snake_ids(eng, my_id)
    options = B.legal_moves(eng, snake)
    if not options:
        return 1  # forced; will die, pick something

    # 1) Eat a nearby growth item if a free move takes us closer.
    hx, hy = snake.coord_list[0]
    item, item_d = B.nearest_live_item(eng, hx, hy, eng.current_round)
    if item is not None:
        best = None
        best_score = item_d
        for d, kind in options:
            if kind != "free":
                continue
            nx, ny = B.head_after(snake, d)
            nd = B.manhattan(nx, ny, item.x, item.y)
            if nd < best_score:
                best_score = nd
                best = d
        if best is not None:
            return best

    # 2) Otherwise move into the most open space (avoid trapping ourselves).
    best_d = None
    best_space = -1
    for d, kind in options:
        nx, ny = B.head_after(snake, d)
        space = B.reachable_space(eng, nx, ny, my_ids)
        if space > best_space:
            best_space = space
            best_d = d
    if best_d is not None:
        return best_d

    # 3) Fallback: first legal move.
    return options[0][0]


# registry so the runner can pick a version by name
def decide_v2(eng, my_id):
    """Keep-alive gate + outward expansion + opportunistic sealing.

    Fixes v1's two failures: (1) we now reject moves whose only escape room is
    smaller than the snake body (so we stop trapping ourselves to death),
    (2) we bias toward open/contested regions to actually cover ground, and
    we snap-shut a loop whenever a seal would claim a big area.
    """
    snake = eng.current_snake()
    if snake is None:
        return 1
    my_ids = _my_snake_ids(eng, my_id)
    options = B.legal_moves(eng, snake)
    if not options:
        return 1

    # 1) Opportunistic sealing: close a loop now if the captured area is big.
    best_seal = None
    best_seal_area = 0
    for d, kind in options:
        if kind == "seal":
            area = B.estimate_seal_area(eng, snake)
            if area > best_seal_area:
                best_seal_area = area
                best_seal = d
    if best_seal is not None and best_seal_area >= 6:
        return best_seal

    # 2) Collect "free" moves that pass a keep-alive gate: after stepping, the
    # reachable open space must be at least body_length + buffer, else we will
    # box ourselves in and die a few ticks later.
    threshold = max(snake.length + 3, 8)
    safe = []
    free_moves = [(d, nx, ny) for d, nx, ny in
                  ((d, *B.head_after(snake, d)) for d, k in options if k == "free")]
    for d, nx, ny in free_moves:
        space = B.reachable_space(eng, nx, ny, my_ids)
        if space >= threshold:
            safe.append((d, nx, ny, space))

    # If nothing passes the gate, fall back to the free move with the most room.
    if not safe:
        if free_moves:
            best = max(free_moves, key=lambda t: B.reachable_space(eng, t[1], t[2], my_ids))
            return best[0]
        return options[0][0]

    # 3) Greedily chase a reachable growth item (length is the prerequisite for
    #    building loops to seal).
    hx, hy = snake.coord_list[0]
    item, item_d = B.nearest_live_item(eng, hx, hy, eng.current_round)
    if item is not None:
        best = None
        best_score = item_d
        for d, nx, ny, _ in safe:
            nd = B.manhattan(nx, ny, item.x, item.y)
            if nd < best_score:
                best_score = nd
                best = d
        if best is not None:
            return best

    # 4) Otherwise expand outward: head toward the direction with the most open
    #    space ahead, breaking ties by distance from our own territory centroid
    #    (go claim neutral ground, not retread our trail).
    cx = sum(c[0] for c in snake.coord_list) / max(1, snake.length)
    cy = sum(c[1] for c in snake.coord_list) / max(1, snake.length)
    best_d = None
    best_key = (-1, -1.0)
    for d, nx, ny, space in safe:
        away = B.manhattan(nx, ny, cx, cy)  # farther from our body centroid
        key = (space, away)
        if key > best_key:
            best_key = key
            best_d = d
    return best_d


# registry so the runner can pick a version by name
VERSIONS = {"v1": decide_v1, "v2": decide_v2}


def get_decide(name):
    return VERSIONS[name]


def _survives(eng, snake, d, my_ids):
    """True if stepping in direction d leaves us at least snake.length room."""
    nx, ny = B.head_after(snake, d)
    if not (0 <= nx < eng.length and 0 <= ny < eng.width):
        return False
    if eng.wall_map[nx][ny] != -1:
        return False
    occ = eng.snake_map[nx][ny]
    if occ != -1 and occ != snake.id:
        return False
    return B.reachable_space(eng, nx, ny, my_ids) >= max(snake.length + 2, 6)


def _safe_free_moves(eng, snake, my_ids):
    out = []
    for d, kind in B.legal_moves(eng, snake):
        if kind == "free" and _survives(eng, snake, d, my_ids):
            nx, ny = B.head_after(snake, d)
            out.append((d, nx, ny))
    return out


def decide_v3(eng, my_id):
    """v2 + split for action economy + aggressive growth + better expansion.

    Lesson from v2 rollouts: humans win by having many snakes (split) that each
    claim territory and seal loops, while I stayed one snake and could only act
    once per turn. v3 splits as soon as it is long enough, prioritises growth
    items to fund loops, and keeps the keep-alive gate from v2.
    """
    snake = eng.current_snake()
    if snake is None:
        return 1
    my_ids = _my_snake_ids(eng, my_id)
    my_snakes = eng.snake_list_0 if my_id == 0 else eng.snake_list_1
    options = B.legal_moves(eng, snake)
    if not options:
        return 1

    # 1) Opportunistic seal: close a loop now if it captures a real area.
    best_seal = None
    best_area = 0
    for d, kind in options:
        if kind == "seal":
            area = B.estimate_seal_area(eng, snake)
            if area > best_area:
                best_area = area
                best_seal = d
    if best_seal is not None and best_area >= 5:
        return best_seal

    # 2) Split when long enough and we have spare snake slots -> more snakes
    #    means more moves per turn and more sealing opportunities.
    if (snake.length >= 8 and len(my_snakes) < 3 and
            snake.id == my_snakes[0].id):
        # only split if there is open space around to avoid trapping
        head = snake.coord_list[0]
        if B.reachable_space(eng, head[0], head[1], my_ids) >= snake.length + 10:
            return 6

    # 3) Growth items: getting longer funds both splits and bigger seals.
    safe = _safe_free_moves(eng, snake, my_ids)
    if not safe:
        free = [(d, *B.head_after(snake, d)) for d, k in options if k == "free"]
        if free:
            return max(free, key=lambda t: B.reachable_space(eng, t[1], t[2], my_ids))[0]
        return options[0][0]

    hx, hy = snake.coord_list[0]
    item, item_d = B.nearest_live_item(eng, hx, hy, eng.current_round)
    if item is not None:
        best = None
        best_score = item_d
        for d, nx, ny in safe:
            nd = B.manhattan(nx, ny, item.x, item.y)
            if nd < best_score:
                best_score = nd
                best = d
        if best is not None:
            return best

    # 4) Expand outward, preferring directions that maximise reachable space and
    #    push away from our body centroid (claim neutral ground).
    cx = sum(c[0] for c in snake.coord_list) / max(1, snake.length)
    cy = sum(c[1] for c in snake.coord_list) / max(1, snake.length)
    best_d = safe[0][0]
    best_key = (-1, -1.0)
    for d, nx, ny in safe:
        space = B.reachable_space(eng, nx, ny, my_ids)
        away = B.manhattan(nx, ny, cx, cy)
        key = (space, away)
        if key > best_key:
            best_key = key
            best_d = d
    return best_d


VERSIONS["v3"] = decide_v3
VERSIONS["v3"] = decide_v3


def _count_free(eng):
    n = 0
    for x in range(eng.length):
        for y in range(eng.width):
            if eng.wall_map[x][y] == -1 and eng.snake_map[x][y] == -1:
                n += 1
    return n


def _survives_safe(eng, snake, d, my_ids):
    """Conservative keep-alive: ALL snake bodies are obstacles (no leniency)."""
    nx, ny = B.head_after(snake, d)
    if not (0 <= nx < eng.length and 0 <= ny < eng.width):
        return False
    if eng.wall_map[nx][ny] != -1:
        return False
    occ = eng.snake_map[nx][ny]
    if occ != -1 and occ != snake.id:
        return False
    return B.reachable_space(eng, nx, ny, my_ids) >= max(snake.length + 2, 8)


def decide_v4(eng, my_id):
    """Conservative survival + territory-maximising expansion + smart sealing.

    Fixes v3's early-death problem: keep-alive now treats every snake body as a
    real obstacle (v1-v3 over-counted room through our own bodies), so we stop
    walking into boxes we cannot escape.
    """
    snake = eng.current_snake()
    if snake is None:
        return 1
    my_ids = _my_snake_ids(eng, my_id)
    my_snakes = eng.snake_list_0 if my_id == 0 else eng.snake_list_1
    options = B.legal_moves(eng, snake)
    if not options:
        return 1

    # 1) Seal now if it captures a worthwhile area (lower threshold mid/late).
    best_seal = None
    best_area = 0
    for d, kind in options:
        if kind == "seal":
            area = B.estimate_seal_area(eng, snake)
            if area > best_area:
                best_area = area
                best_seal = d
    seal_floor = 8 if eng.current_round < 40 else 4
    if best_seal is not None and best_area >= seal_floor:
        return best_seal

    # 2) Split for action economy once long enough and room exists around.
    if (snake.length >= 10 and len(my_snakes) < 3 and snake.id == my_snakes[0].id):
        hx, hy = snake.coord_list[0]
        if B.reachable_space(eng, hx, hy, my_ids) >= snake.length + 14:
            return 6

    # 3) Conservative keep-alive filtering of free moves.
    safe = []
    for d, kind in options:
        if kind == "free" and _survives_safe(eng, snake, d, my_ids):
            safe.append((d, *B.head_after(snake, d)))
    if not safe:
        free = [(d, *B.head_after(snake, d)) for d, k in options if k == "free"]
        if free:
            return max(free, key=lambda t: B.reachable_space(eng, t[1], t[2], my_ids))[0]
        return options[0][0]

    # 4) Chase a reachable growth item first (length funds seals/splits).
    hx, hy = snake.coord_list[0]
    item, item_d = B.nearest_live_item(eng, hx, hy, eng.current_round)
    if item is not None:
        best = None
        best_score = item_d
        for d, nx, ny in safe:
            nd = B.manhattan(nx, ny, item.x, item.y)
            if nd < best_score:
                best_score = nd
                best = d
        if best is not None:
            return best

    # 5) Territory move: maximise reachable neutral space, then push away from
    #    our body centroid toward open ground.
    cx = sum(c[0] for c in snake.coord_list) / max(1, snake.length)
    cy = sum(c[1] for c in snake.coord_list) / max(1, snake.length)
    best_d = safe[0][0]
    best_key = (-1, -1.0)
    for d, nx, ny in safe:
        space = B.reachable_space(eng, nx, ny, my_ids)
        away = B.manhattan(nx, ny, cx, cy)
        key = (space, away)
        if key > best_key:
            best_key = key
            best_d = d
    return best_d


VERSIONS["v4"] = decide_v4
VERSIONS["v4"] = decide_v4


def _nearest_own_body(eng, snake):
    """Return (body_coord, distance) of the closest non-neck own body cell, to
    encourage closing a loop. None if the snake is too short."""
    cl = snake.coord_list
    if len(cl) < 6:
        return None
    hx, hy = cl[0]
    best = None
    best_d = 10 ** 9
    for i, (bx, by) in enumerate(cl):
        if i <= 1:
            continue
        dd = abs(bx - hx) + abs(by - hy)
        if dd < best_d:
            best_d = dd
            best = (bx, by)
    return (best, best_d) if best else None


def decide_v5(eng, my_id):
    """Aggressive sealing + full snake count + body-closing bias.

    Diagnosis from v4 rollouts: we lose the mid-game territory race because we
    split to only 2-3 snakes while humans hold 4, and we almost never seal
    (humans get burst +30 per seal). v5 pushes to 4 snakes fast, lowers the seal
    threshold, and biases long snakes toward closing onto their own body so loops
    form and get sealed.
    """
    snake = eng.current_snake()
    if snake is None:
        return 1
    my_ids = _my_snake_ids(eng, my_id)
    my_snakes = eng.snake_list_0 if my_id == 0 else eng.snake_list_1
    options = B.legal_moves(eng, snake)
    if not options:
        return 1

    # 1) Seal whenever the captured area is worth it (aggressive threshold).
    best_seal = None
    best_area = 0
    for d, kind in options:
        if kind == "seal":
            area = B.estimate_seal_area(eng, snake)
            if area > best_area:
                best_area = area
                best_seal = d
    if best_seal is not None and best_area >= 4:
        return best_seal

    # 2) Split early and often to match humans' 4-snake action economy.
    if snake.length >= 8 and len(my_snakes) < 4 and snake.id == my_snakes[0].id:
        hx, hy = snake.coord_list[0]
        if B.reachable_space(eng, hx, hy, my_ids) >= snake.length + 12:
            return 6

    # 3) Conservative keep-alive filtering.
    safe = []
    for d, kind in options:
        if kind == "free" and _survives_safe(eng, snake, d, my_ids):
            safe.append((d, *B.head_after(snake, d)))
    if not safe:
        free = [(d, *B.head_after(snake, d)) for d, k in options if k == "free"]
        if free:
            return max(free, key=lambda t: B.reachable_space(eng, t[1], t[2], my_ids))[0]
        return options[0][0]

    # 4) Growth item (length funds seals and splits).
    hx, hy = snake.coord_list[0]
    item, item_d = B.nearest_live_item(eng, hx, hy, eng.current_round)
    if item is not None:
        best = None
        best_score = item_d
        for d, nx, ny in safe:
            nd = B.manhattan(nx, ny, item.x, item.y)
            if nd < best_score:
                best_score = nd
                best = d
        if best is not None:
            return best

    # 5) Body-closing bias: long snake steps toward its own body to form a loop.
    own = _nearest_own_body(eng, snake)
    if own is not None and snake.length >= 12:
        (bx, by), body_d = own
        if 2 <= body_d <= 6:
            best = None
            best_d = body_d
            for d, nx, ny in safe:
                nd = B.manhattan(nx, ny, bx, by)
                if nd < best_d:
                    best_d = nd
                    best = d
            if best is not None:
                return best

    # 6) Territory move: maximise reachable space, push away from centroid.
    cx = sum(c[0] for c in snake.coord_list) / max(1, snake.length)
    cy = sum(c[1] for c in snake.coord_list) / max(1, snake.length)
    best_d = safe[0][0]
    best_key = (-1, -1.0)
    for d, nx, ny in safe:
        space = B.reachable_space(eng, nx, ny, my_ids)
        away = B.manhattan(nx, ny, cx, cy)
        key = (space, away)
        if key > best_key:
            best_key = key
            best_d = d
    return best_d


VERSIONS["v5"] = decide_v5
VERSIONS["v5"] = decide_v5


def decide_v6(eng, my_id):
    """v4 survival balance + v5 body-closing seal bias (no early over-split).

    Rollout verdict: v4 survived best vs rank15, v5 sealed more but died
    early from over-splitting. v6 keeps v4's conservative split (>=10, <3 snakes)
    and keep-alive, but adds v5's body-closing loop formation so we actually
    get burst seals in the mid-game without going fragile.
    """
    snake = eng.current_snake()
    if snake is None:
        return 1
    my_ids = _my_snake_ids(eng, my_id)
    my_snakes = eng.snake_list_0 if my_id == 0 else eng.snake_list_1
    options = B.legal_moves(eng, snake)
    if not options:
        return 1

    # 1) Seal whenever worth it.
    best_seal = None
    best_area = 0
    for d, kind in options:
        if kind == "seal":
            area = B.estimate_seal_area(eng, snake)
            if area > best_area:
                best_area = area
                best_seal = d
    if best_seal is not None and best_area >= 4:
        return best_seal

    # 2) Conservative split (v4 calibration: long enough, room around, <3 snakes).
    if snake.length >= 10 and len(my_snakes) < 3 and snake.id == my_snakes[0].id:
        hx, hy = snake.coord_list[0]
        if B.reachable_space(eng, hx, hy, my_ids) >= snake.length + 14:
            return 6

    # 3) Conservative keep-alive filter.
    safe = []
    for d, kind in options:
        if kind == "free" and _survives_safe(eng, snake, d, my_ids):
            safe.append((d, *B.head_after(snake, d)))
    if not safe:
        free = [(d, *B.head_after(snake, d)) for d, k in options if k == "free"]
        if free:
            return max(free, key=lambda t: B.reachable_space(eng, t[1], t[2], my_ids))[0]
        return options[0][0]

    # 4) Growth item.
    hx, hy = snake.coord_list[0]
    item, item_d = B.nearest_live_item(eng, hx, hy, eng.current_round)
    if item is not None:
        best = None
        best_score = item_d
        for d, nx, ny in safe:
            nd = B.manhattan(nx, ny, item.x, item.y)
            if nd < best_score:
                best_score = nd
                best = d
        if best is not None:
            return best

    # 5) Body-closing seal bias (from v5) for long snakes.
    own = _nearest_own_body(eng, snake)
    if own is not None and snake.length >= 12:
        (bx, by), body_d = own
        if 2 <= body_d <= 6:
            best = None
            best_d = body_d
            for d, nx, ny in safe:
                nd = B.manhattan(nx, ny, bx, by)
                if nd < best_d:
                    best_d = nd
                    best = d
            if best is not None:
                return best

    # 6) Territory move.
    cx = sum(c[0] for c in snake.coord_list) / max(1, snake.length)
    cy = sum(c[1] for c in snake.coord_list) / max(1, snake.length)
    best_d = safe[0][0]
    best_key = (-1, -1.0)
    for d, nx, ny in safe:
        space = B.reachable_space(eng, nx, ny, my_ids)
        away = B.manhattan(nx, ny, cx, cy)
        key = (space, away)
        if key > best_key:
            best_key = key
            best_d = d
    return best_d


VERSIONS["v6"] = decide_v6
