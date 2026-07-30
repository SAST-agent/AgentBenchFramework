"""Decision-space contract for 26_snakego.

Defines the observation, the legal macro-action set, the action mask, the
termination conditions, and the action-support set used for information-gain
computation. This is the formal interface every strategy version implements, so
IG and evaluation always speak the same language across versions.
"""
from dataclasses import dataclass
from typing import List, Optional

# --------------------------------------------------------------------------
# Macro-action set
# --------------------------------------------------------------------------
# The primitive op types from adk.hpp. Every decision collapses to ONE of these
# six macros per (player, snake-to-operate). Direction macros are absolute
# board directions (not relative), so the strategy is board-frame-stable.
MACRO_ACTIONS = {
    1: "MOVE_RIGHT",   # head +x
    2: "MOVE_UP",      # head +y
    3: "MOVE_LEFT",    # head -x
    4: "MOVE_DOWN",    # head -y
    5: "RAILGUN",      # fire railgun if held (clears a wall line)
    6: "SPLIT",        # split current snake in two
}
ACTION_IDS = sorted(MACRO_ACTIONS.keys())  # [1,2,3,4,5,6]

# The full ordered support set IG is computed over. Keeping it fixed across
# versions means IG(behaviour) is directly comparable version-to-version.
SUPPORT = ACTION_IDS


@dataclass
class Observation:
    """The complete, board-frame observation a strategy sees at decision time.

    This is exactly what engine.py exposes; we freeze it so the observation
    space is version-stable and IG/eval cannot drift as strategies evolve.
    """
    round: int
    player: int               # whose turn (0 or 1)
    snake_id: int             # snake to operate this turn
    length: int               # length of the snake to operate
    head: tuple               # (x, y)
    coord_list: list          # full body, head first
    length_bank: int
    has_railgun: bool
    wall_map: list            # [x][y] -> camp or -1
    snake_map: list           # [x][y] -> snake id or -1
    item_map: list            # [x][y] -> item id or -1
    my_snake_ids: set         # ids of my own snakes
    enemy_snake_ids: set
    board_len: int
    board_wid: int


def observe(eng) -> Observation:
    """Project the live Engine state into the frozen observation contract."""
    snake = eng.current_snake()
    my_ids = {s.id for s in eng.my_snakes()}
    en_ids = {s.id for s in eng.opponents_snakes()}
    return Observation(
        round=eng.current_round,
        player=eng.current_player,
        snake_id=eng.current_snake_id,
        length=snake.length if snake else 0,
        head=tuple(snake.coord_list[0]) if snake and snake.coord_list else None,
        coord_list=[tuple(c) for c in snake.coord_list] if snake else [],
        length_bank=snake.length_bank if snake else 0,
        has_railgun=(snake.railgun_item_id != -1) if snake else False,
        wall_map=[row[:] for row in eng.wall_map],
        snake_map=[row[:] for row in eng.snake_map],
        item_map=[row[:] for row in eng.item_map],
        my_snake_ids=set(my_ids),
        enemy_snake_ids=set(en_ids),
        board_len=eng.length,
        board_wid=eng.width,
    )


# --------------------------------------------------------------------------
# Action mask (legality)
# --------------------------------------------------------------------------
@dataclass
class ActionMask:
    """Which of the 6 macros are legal right now, and why each is masked.

    A macro is 'legal' if the engine will accept it (not crash / not no-op).
    Note: a legal-but-suicidal move is still 'legal' here; survivability is a
    strategy concern, enforced separately via scoring, not via the mask.
    """
    legal: dict   # {1:True,...,6:False}
    reasons: dict  # {op: short reason string or "" }


def compute_mask(eng) -> ActionMask:
    """Compute the legality mask for the current snake from engine state."""
    from snakego.board import classify_move, head_after
    snake = eng.current_snake()
    legal = {a: False for a in ACTION_IDS}
    reasons = {a: "" for a in ACTION_IDS}
    if snake is None or not snake.coord_list:
        for a in ACTION_IDS:
            reasons[a] = "no_snake"
        return ActionMask(legal, reasons)
    # moves 1-4
    for d in (1, 2, 3, 4):
        kind = classify_move(eng, snake, d)
        if kind == "dead":
            legal[d] = False
            reasons[d] = _dead_reason(eng, snake, d)
        else:
            legal[d] = True
    # railgun: need item + length>=2
    if snake.railgun_item_id != -1 and snake.length >= 2:
        legal[5] = True
    else:
        reasons[5] = "no_railgun" if snake.railgun_item_id == -1 else "too_short"
    # split: <4 snakes, length>=2
    n_mine = len(eng.my_snakes())
    if n_mine < 4 and snake.length >= 2:
        legal[6] = True
    else:
        reasons[6] = "max_snakes" if n_mine >= 4 else "too_short"
    return ActionMask(legal, reasons)


def _dead_reason(eng, snake, d):
    from snakego.board import head_after, is_reversal
    if is_reversal(snake, d):
        return "reversal"
    nx, ny = head_after(snake, d)
    if nx < 0 or ny < 0 or nx >= eng.length or ny >= eng.width:
        return "oob"
    if eng.wall_map[nx][ny] != -1:
        return "wall"
    occ = eng.snake_map[nx][ny]
    if occ != -1 and occ != snake.id:
        return "enemy_body"
    return "?"


# --------------------------------------------------------------------------
# Termination
# --------------------------------------------------------------------------
def is_terminal(eng) -> bool:
    """Game ends when both camps have no snakes, or max_round exceeded.

    Note: one camp losing ALL its snakes does NOT end the game -- the survivor
    keeps claiming territory until max_round. This matches the official judge's
    settle_round semantics (running = not (snake_num==[0,0] or turn>max_round)).
    """
    return (not eng.snake_list_0 and not eng.snake_list_1) or eng.current_round > eng.max_round


def outcome(eng):
    """Return (winner, scores) at terminal state, else (None, current_scores)."""
    s = eng.score()
    return (0 if s[0] >= s[1] else 1, s)


# --------------------------------------------------------------------------
# Behaviour distribution (for IG)
# --------------------------------------------------------------------------
def action_distribution(action_counts):
    """Turn a {op: count} tally into a distribution over the fixed SUPPORT set.

    Pseudocount of 0.5 per action keeps every mass strictly positive so KL/IG
    are always finite (Laplace smoothing). This is the unified distribution
    convention: every version and every episode uses this exact transform.
    """
    counts = {a: action_counts.get(a, 0) + 0.5 for a in SUPPORT}
    total = sum(counts.values())
    return {a: counts[a] / total for a in SUPPORT}
