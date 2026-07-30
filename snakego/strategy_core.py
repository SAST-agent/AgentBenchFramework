"""The consolidated, scoring-based strategy.

This replaces the v1..v6 if-else stack with ONE interpretable policy: at each
decision we score every legal macro-action by a weighted sum of named feature
terms, then pick the argmax. The feature terms are interpretable (each has a
clear meaning and a unit), and the weights are the "experience" -- they are
what iteration tunes, and what gets compressed/consolidated.

This is "interpretable code, not just if-else": the policy is a linear model
over hand-engineered features. It can express rich behaviour, but every
decision is auditable ("move 2 won because: space_term=+18, danger_term=-40,
seal_term=0"). New lessons from replay become new feature terms or weight
edits, then the whole thing re-consolidates -- no version pile-up.
"""
from dataclasses import dataclass, field
from snakego import board as B
from snakego.decision_space import compute_mask, observe


@dataclass
class Weights:
    """The tunable experience. One flat struct, easy to diff/version/rollback."""
    # survival
    space_per_body_len: float = 0.4      # reachable_space / max(len,1), capped at 4.0
    trap_penalty: float = -8.0           # per unit deficit when room < len+buffer
    lethal_trap: float = -1000.0         # room < len: near-certain death
    enemy_proximity: float = -1.0        # penalty per enemy-head within distance 3
    # territory
    claim_free_cell: float = 1.5         # moving into a neutral empty cell
    seal_area: float = 2.5              # captured area if this move seals
    approach_sealable: float = 0.4       # move toward closing a loop
    # resources
    growth_item_delta: float = 6.0       # manhattan delta toward a growth item
    railgun_value: float = 6.0          # firing railgun to clear a useful wall
    center_seeking: float = 0.1          # bonus for being near board center (tiebreaker)
    # tempo / action economy
    split_value: float = 8.0            # base; length urgency adds more
    split_urgency: float = 1.2          # per unit of length beyond 6
    expand_from_centroid: float = 0.02   # push away from own body centroid (reduced)
    # anti-patterns learned from replays
    overcommit_corner: float = -2.0      # avoid moving into a 1-exit pocket
    max_growth_length: float = 14.0      # above this, growth items lose value
    # territory sealing (iter8+: humans seal to win, we never did)
    loop_potential: float = 0.6          # reward moves that approach a sealable loop
    territory_contest: float = 0.15      # reward moves into unclaimed contested space


@dataclass
class DecisionTrace:
    """Audit record for one decision, so every move is explainable."""
    op: int
    total: float
    terms: dict = field(default_factory=dict)


def _reachable_safe(eng, snake, d, my_ids):
    nx, ny = B.head_after(snake, d)
    if not (0 <= nx < eng.length and 0 <= ny < eng.width):
        return 0
    if eng.wall_map[nx][ny] != -1:
        return 0
    occ = eng.snake_map[nx][ny]
    if occ != -1 and occ != snake.id:
        return 0
    return B.reachable_space(eng, nx, ny, my_ids)


def _exits(eng, x, y):
    """Number of open neighbours of (x,y) -- 1 means a pocket (dead end)."""
    n = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < eng.length and 0 <= ny < eng.width:
            if eng.wall_map[nx][ny] == -1 and eng.snake_map[nx][ny] == -1:
                n += 1
    return n


def score_action(eng, op, w: Weights) -> DecisionTrace:
    """Score one legal macro-action. Returns a trace."""
    snake = eng.current_snake()
    terms = {}
    if snake is None:
        return DecisionTrace(op, -1e9, terms)
    my_ids = {s.id for s in eng.my_snakes()}

    n_mine = len(eng.my_snakes())

    if op == 5:  # railgun -- value only when holding one
        if snake.railgun_item_id < 0:
            return DecisionTrace(op, -1e9, {"no_railgun": -1e9})
        terms["railgun_value"] = w.railgun_value
        return DecisionTrace(op, sum(terms.values()), terms)
    if op == 6:  # split
        # Length-urgency split: humans split at length 9-16. The longer the
        # snake, the more urgent (a long snake is a survival liability). The
        # old hard gate (room >= len+12) combined with the uncapped space term
        # meant split NEVER won, so the snake grew to length 28 and died.
        if n_mine >= 4 or snake.length < 4:
            return DecisionTrace(op, -1e9, {"split_blocked": -1e9})
        if snake.length < 8:
            # too short: splitting at len 6-7 creates two length-3 snakes that
            # can't reach items or seal territory. Humans split at len 9-16.
            return DecisionTrace(op, -1e9, {"split_too_short": -1e9})
        hx, hy = snake.coord_list[0]
        room = B.reachable_space(eng, hx, hy, my_ids)
        if room < snake.length:
            terms["split_value"] = -1e9  # can't afford to split right now
        else:
            urgency = max(0, snake.length - 6) * w.split_urgency
            terms["split_value"] = w.split_value + urgency
        return DecisionTrace(op, sum(terms.values()), terms)

    # moves 1-4
    if not (1 <= op <= 4):
        return DecisionTrace(op, -1e9, terms)
    kind = B.classify_move(eng, snake, op)
    if kind == "dead":
        return DecisionTrace(op, -1e9, {"dead": -1e9})
    nx, ny = B.head_after(snake, op)
    L = max(snake.length, 1)

    # space / survival
    room = B.reachable_space(eng, nx, ny, my_ids)
    # CAPPED so it doesn't dominate every other term early game (was ~26).
    terms["space_per_body_len"] = w.space_per_body_len * min(room / L, 4.0)
    # SURVIVAL GATE: room < len means the snake cannot fit -> near-certain death.
    # room < len+4 means getting tight -> escalating penalty.
    if room < L:
        terms["lethal_trap"] = w.lethal_trap
        terms["trap_penalty"] = w.trap_penalty * (L - room)
    elif room < L + 4:
        terms["trap_penalty"] = w.trap_penalty * (L + 4 - room)

    # territory
    if kind == "free":
        occ = eng.snake_map[nx][ny]
        if occ == -1 and eng.wall_map[nx][ny] == -1:
            terms["claim_free_cell"] = w.claim_free_cell
    if kind == "seal":
        area = B.estimate_seal_area(eng, snake)
        terms["seal_area"] = w.seal_area * area


    # loop potential: guide the snake toward positions where it can form
    # a sealing loop. Humans win by closing loops to capture big regions.
    # Normalized to 0-4 range so it doesn't dominate survival terms.
    loop_val = B.estimate_loop_potential(eng, snake, nx, ny)
    if loop_val > 0:
        terms["loop_potential"] = w.loop_potential * min(loop_val / 10.0, 4.0)

    # territory contest: reward moves into unclaimed contested space,
    # denying the opponent the chance to claim or seal it.
    # Normalized to 0-4 range.
    contest = B.territory_contest_area(eng, nx, ny)
    terms["territory_contest"] = w.territory_contest * min(contest / 40.0, 4.0)


# growth item
    item, item_d = B.nearest_live_item(eng, snake.coord_list[0][0],
                                       snake.coord_list[0][1], eng.current_round)
    if item is not None:
        nd = B.manhattan(nx, ny, item.x, item.y)
        growth_factor = 1.0 if L < w.max_growth_length else max(
            0.1, 1.0 - (L - w.max_growth_length) * 0.1)
        terms["growth_item_delta"] = w.growth_item_delta * growth_factor * max(0, item_d - nd)

    # enemy proximity: penalize moves that bring us close to enemy snake heads.
    # From replay analysis: my snakes converge to enemy territory and get boxed
    # in by enemy bodies (not walls). Keeping distance from enemy heads reduces
    # the chance they can surround us next turn.
    enemy_heads = [s.coord_list[0] for s in eng.opponents_snakes() if s.coord_list]
    n_close_enemies = sum(1 for eh in enemy_heads if B.manhattan(nx, ny, eh[0], eh[1]) <= 3)
    terms["enemy_proximity"] = w.enemy_proximity * n_close_enemies

    # center seeking: items spawn in the board center (6-9, 6-9 early game).
    # Staying near center gives access to growth items. This also counteracts
    # expand_from_centroid pushing snakes toward the enemy corner.
    center_x, center_y = eng.length / 2, eng.width / 2
    dist_to_center = B.manhattan(nx, ny, center_x, center_y)
    terms["center_seeking"] = w.center_seeking * max(0, 8 - dist_to_center)

    # anti-pattern: pocket
    if _exits(eng, nx, ny) <= 1:
        terms["overcommit_corner"] = w.overcommit_corner

    # expand from centroid
    cx = sum(c[0] for c in snake.coord_list) / L
    cy = sum(c[1] for c in snake.coord_list) / L
    terms["expand_from_centroid"] = w.expand_from_centroid * B.manhattan(nx, ny, cx, cy)

    return DecisionTrace(op, sum(terms.values()), terms)


def heuristic_override(eng, w: Weights):
    """Hard if-else rules that take priority over weighted scoring.

    These are NOT weight tweaks -- they are explicit behavioral rules derived
    from replay analysis:
      1. BIG SEAL: if any legal move seals > 8 cells, take it immediately.
         Humans win by closing big loops (+28 in one move); we never did.
      2. EARLY SPLIT: if round < 40, I have < 3 snakes, and current snake
         length >= 9, split now. Humans double-split at round 22-23 for 4
         snakes; I was too slow at round 8/27/49.
      3. SURVIVAL GATE: if the best scoring move has room < length, fall
         through to scoring but force-skip lethal moves.
    Returns op 1-6, or None to fall through to weighted scoring.
    """
    from snakego.decision_space import compute_mask
    from snakego import board as B

    mask = compute_mask(eng)
    snake = eng.current_snake()
    if snake is None:
        return None

    my_ids = {s.id for s in eng.my_snakes()}
    n_mine = len(eng.my_snakes())

    # Rule 1: BIG SEAL -- if this move closes a loop enclosing >= 8 cells, take it.
    for op in (1, 2, 3, 4):
        if not mask.legal[op]:
            continue
        kind = B.classify_move(eng, snake, op)
        if kind == "seal":
            area = B.estimate_seal_area(eng, snake)
            if area >= 6:
                return op

    # Rule 2: EARLY SPLIT -- match human double-split tempo (r22-23).
    # Humans reach 4 snakes fast; being stuck at 2 loses the action economy.
    if (eng.current_round < 50 and n_mine <= 2
            and snake.length >= 8 and mask.legal[6]):
        hx, hy = snake.coord_list[0]
        room = B.reachable_space(eng, hx, hy, my_ids)
        if room >= snake.length + 6:
            return 6

    # Rule 3a: SURVIVAL HARD GATE -- if getting tight (room < length + 6),
    # restrict to moves that maintain or increase available room. This is
    # non-negotiable: a dead snake scores zero. The weighted scoring tries
    # to do this but its trap_penalty term is sometimes overridden by high
    # item/seal/split values. This gate has absolute priority.
    hx, hy = snake.coord_list[0]
    my_room = B.reachable_space(eng, hx, hy, my_ids)
    if my_room < snake.length + 6:
        safe_ops = []
        for op in (1, 2, 3, 4):
            if not mask.legal[op]:
                continue
            nx, ny = B.head_after(snake, op)
            if not (0 <= nx < eng.length and 0 <= ny < eng.width):
                continue
            kind = B.classify_move(eng, snake, op)
            if kind == "dead":
                continue
            room_after = B.reachable_space(eng, nx, ny, my_ids)
            if room_after >= my_room:  # only moves that dont shrink room
                safe_ops.append(op)
        if len(safe_ops) >= 1:
            return safe_ops[0]

    # Rule 3b: U-TURN FORCING -- detect when the snake has moved straight for
    # 4+ body segments. When long enough (>=10) and safe, force a 90-degree
    # turn. This creates the curving behaviour that leads to loop formation
    # and sealing. Without it, the snake walks straight lines forever and
    # never encloses territory. This is the single most impactful rule.
    if snake.length >= 10 and len(snake.coord_list) >= 5:
        cl = snake.coord_list
        h0x, h0y = cl[0]
        h1x, h1y = cl[1]
        h4x, h4y = cl[4] if len(cl) > 4 else cl[-1]
        # check straight-line: head, neck, and segment 4 are colinear
        dx1, dy1 = h0x - h1x, h0y - h1y
        dx2, dy2 = h0x - h4x, h0y - h4y
        is_straight = (dx1 != 0 and dx2 != 0 and dx1 * dx2 > 0) or \
                      (dy1 != 0 and dy2 != 0 and dy1 * dy2 > 0)
        if is_straight:
            hx, hy = cl[0]
            my_room = B.reachable_space(eng, hx, hy, my_ids)
            if my_room >= snake.length + 6:
                # find the perpendicular turns (90 degrees from current dir)
               if dx1 != 0:  # moving horizontally -> turn up/down
                   turns = [3, 4]
               else:  # moving vertically -> turn left/right
                   turns = [1, 2]
               safe_turns = []
               if dx1 != 0:  # moving horizontally -> perpendicular = up/down
                   turns = [2, 4]
               else:  # moving vertically -> perpendicular = left/right
                   turns = [1, 3]
               safe_turns = []
               for op in turns:
                if dx1 != 0:  # moving horizontally -> perpendicular = up/down
                    turns = [2, 4]
                else:  # moving vertically -> perpendicular = left/right
                    turns = [1, 3]
                safe_turns = []
                for op in turns:
                    if not mask.legal[op]:
                        continue
                    kind = B.classify_move(eng, snake, op)
                    if kind == "dead":
                        continue
                    nx, ny = B.head_after(snake, op)
                    room_after = B.reachable_space(eng, nx, ny, my_ids)
                    if room_after >= snake.length + 2:
                        safe_turns.append((op, room_after))
                if safe_turns:
                    # pick the turn with more room
                    safe_turns.sort(key=lambda x: -x[1])
                    return safe_turns[0][0]

    return None  # fall through to weighted scoring


def decide(eng, my_id, w: Weights, trace=False):
    """The consolidated policy. Returns op, or (op, traces) if trace=True.

    Layer 1: heuristic_override -- hard if-else rules (big seal, early split,
             survival gate). Falls through to Layer 2 if no rule fires.
    Layer 2: weighted scoring over all legal actions.
    """
    # Layer 1: interpretable if-else rules
    forced = heuristic_override(eng, w)
    if forced is not None:
        dt = DecisionTrace(forced, 0, {"heuristic_override": True})
        return (forced, dt) if trace else forced

    # Layer 2: weighted scoring
    mask = compute_mask(eng)
    best = None
    best_trace = None
    for op in (1, 2, 3, 4, 5, 6):
        if not mask.legal[op]:
            continue
        dt = score_action(eng, op, w)
        if best is None or dt.total > best:
            best = dt.total
            best_trace = dt
            best_op = op
    if best is None:
        # No legal scoring move: never default to a U-turn (official judge
        # ends the game with score -100 for reversal). Pick the first
        # non-reversal direction.
        from snakego.board import is_reversal as _ir
        snake2 = eng.current_snake()
        op = 1
        if snake2 and snake2.length >= 2:
            for cand in (1, 2, 3, 4):
                if not _ir(snake2, cand):
                    op = cand
                    break
        best_trace = DecisionTrace(op, -1e9, {"fallback_no_uturn": True})
    else:
        op = best_op
    return (op, best_trace) if trace else op


def make_decide(w: Weights):
    """Bind weights into a decide(engine, my_id) callable for the host."""
    def _d(eng, my_id):
        return decide(eng, my_id, w)
    _d.__name__ = "consolidated"
    return _d
