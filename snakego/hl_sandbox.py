# Deterministic pure-Python self-play harness: verifies that v8 actually
# closes loops and seals territory (the thing v0-v7 never did), and computes
# the per-state epsilon-regularized KL information gain (research doc 4.1/15).
#
# No subprocess / no human binaries required: both players run in-process
# against the shared engine, so the result is fully reproducible. This is the
# honest "does the structural change produce territory + non-trivial IG" check.
import copy
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snakego.engine import Engine, Item
from snakego.host import generate_items, LENGTH, WIDTH, MAX_ROUND
from snakego import board as B
from snakego.decision_space import compute_mask, SUPPORT, action_distribution
from snakego.strategy_loop import decide as v7_decide
from snakego import strategy_v8 as V8

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
EPSILON = 0.05  # measurement-floor ratio (research doc 4.1)


class _DecidePlayer:
    # Minimal in-process player: decides from shared engine state.
    def __init__(self, name, decide_fn):
        self.name = name
        self.decide_fn = decide_fn

    def act(self, eng, pid):
        op = int(self.decide_fn(eng, pid))
        if not (1 <= op <= 6):
            op = 1
        return op


def _count_my_walls(eng, camp):
    n = 0
    for x in range(eng.length):
        for y in range(eng.width):
            if eng.wall_map[x][y] == camp:
                n += 1
    return n


def play_game(decide0, decide1, seed, record_states=False):
    # Run a full game in-process. Returns (scores, stats, decision_states).
    items = generate_items(seed)
    eng = Engine(LENGTH, WIDTH, MAX_ROUND, items)
    p0 = _DecidePlayer("p0", decide0)
    p1 = _DecidePlayer("p1", decide1)
    seals = [0, 0]
    phase_log = {0: {}, 1: {}}
    decision_states = []  # snapshots for IG (player0 only)
    step = 0
    while True:
        cur = eng.current_player
        player = p0 if cur == 0 else p1
        if not eng.alive_player():
            running = eng.do_operation(0)
            step += 1
            if not running:
                break
            continue
        # snapshot BEFORE the move, for IG (player 0 decisions only)
        if cur == 0 and record_states:
            decision_states.append(_snapshot(eng))
        prev_walls = _count_my_walls(eng, cur)
        op = player.act(eng, cur)
        # trace v8 phase for diagnostics
        if cur == 0 and hasattr(V8, "decide"):
            try:
                _, info = V8.decide(eng, cur, trace=True)
                ph = info.get("phase", "?")
                phase_log[0][ph] = phase_log[0].get(ph, 0) + 1
            except Exception:
                pass
        running = eng.do_operation(op)
        step += 1
        after_walls = _count_my_walls(eng, cur)
        if after_walls > prev_walls:
            seals[cur] += 1
        if not running:
            break
    scores = eng.score()
    stats = {
        "scores": scores,
        "ratio0": scores[0] / max(1, scores[0] + scores[1]),
        "seals": seals,
        "rounds": eng.current_round,
        "wall0": _count_my_walls(eng, 0),
        "wall1": _count_my_walls(eng, 1),
        "phase_p0": phase_log[0],
    }
    return stats, decision_states


def _snapshot(eng):
    return {
        "wall": [row[:] for row in eng.wall_map],
        "snake": [row[:] for row in eng.snake_map],
        "item": [row[:] for row in eng.item_map],
        "round": eng.current_round,
        "player": eng.current_player,
        "snake_id": eng.current_snake_id,
        "snakes0": [[list(c) for c in s.coord_list] + [[s.length_bank, s.railgun_item_id]] for s in eng.snake_list_0],
        "snakes1": [[list(c) for c in s.coord_list] + [[s.length_bank, s.railgun_item_id]] for s in eng.snake_list_1],
        "next_snake_id": eng.next_snake_id,
    }


def _restore(eng, snap):
    eng.wall_map = [row[:] for row in snap["wall"]]
    eng.snake_map = [row[:] for row in snap["snake"]]
    eng.item_map = [row[:] for row in snap["item"]]
    eng.current_round = snap["round"]
    eng.current_player = snap["player"]
    eng.current_snake_id = snap["snake_id"]
    eng.next_snake_id = snap["next_snake_id"]
    eng.snake_list_0 = [_mk_snake(s, 0) for s in snap["snakes0"]]
    eng.snake_list_1 = [_mk_snake(s, 1) for s in snap["snakes1"]]


def _mk_snake(rec, camp):
    from snakego.engine import Snake
    coords = [tuple(c) for c in rec[:-1]]
    meta = rec[-1]
    sid = camp if coords == [(0, WIDTH - 1)] or coords == [(LENGTH - 1, 0)] else camp
    # reconstruct id from snapshot snake_map is hard; assign sequentially via list pos
    return Snake(coords, _find_id(camp, coords), meta[0], camp, meta[1])


_id_counter = [100]


def _find_id(camp, coords):
    _id_counter[0] += 1
    return _id_counter[0]


def eps_reg_kl(old_op, new_op, legal_mask, epsilon=EPSILON):
    # epsilon-regularized policy KL over the legal-action support (doc 4.1).
    # deterministic policy -> mass (1-eps) on chosen action, eps/|A| uniform.
    legal = [a for a in SUPPORT if legal_mask.get(a, False)]
    if not legal:
        return 0.0
    n = len(legal)
    p = {}
    q = {}
    for a in legal:
        p[a] = (1 - epsilon) * (1.0 if a == old_op else 0.0) + epsilon / n
        q[a] = (1 - epsilon) * (1.0 if a == new_op else 0.0) + epsilon / n
    return sum(q[a] * math.log((q[a] + 1e-12) / (p[a] + 1e-12)) for a in legal)


def compute_ig(states, old_decide, new_decide):
    # Per-state epsilon-regularized KL over shared decision states (doc 15).
    from snakego.engine import Engine
    traces = []
    disagree = 0
    for snap in states:
        eng = Engine(LENGTH, WIDTH, MAX_ROUND, [])
        _restore(eng, snap)
        mask = compute_mask(eng)
        try:
            old_op = int(old_decide(eng, 0))
        except Exception:
            old_op = 1
        try:
            new_op = int(new_decide(eng, 0))
        except Exception:
            new_op = 1
        if not (1 <= old_op <= 6):
            old_op = 1
        if not (1 <= new_op <= 6):
            new_op = 1
        if old_op != new_op:
            disagree += 1
        traces.append(eps_reg_kl(old_op, new_op, mask.legal))
    if not traces:
        return {"ig_kl": None, "status": "no_states", "disagree_rate": None}
    return {
        "ig_kl": sum(traces) / len(traces),
        "trajectory_kl": sum(traces),
        "n_states": len(traces),
        "disagree_rate": disagree / len(traces),
        "status": "ok",
    }


def main():
    print("=== SnakeGo HL Sandbox: v7 vs v8 (in-process self-play) ===\n")
    seeds = [1, 2, 3, 7, 11, 23]
    v8 = V8.make_decide()

    # --- v8 plays as player 0 vs a simple greedy opponent (v7-like scorer) ---
    from snakego.strategy_core import make_decide as make_scorer
    opp = make_scorer(__import__("snakego.strategy_core", fromlist=["Weights"]).Weights())

    print("Phase 1: v8 (p0) vs greedy-scorer opponent (p1)\n")
    all_stats = []
    all_states = []
    total_seals8 = 0
    for s in seeds:
        stats, states = play_game(v8, opp, s, record_states=True)
        all_stats.append(stats)
        all_states.extend(states)
        total_seals8 += stats["seals"][0]
        print("  seed=%2d  v8=%3d opp=%3d ratio=%.3f seals[v8=%d opp=%d] rounds=%3d wall[v8=%d opp=%d] phases=%s" % (
            s, stats["scores"][0], stats["scores"][1], stats["ratio0"],
            stats["seals"][0], stats["seals"][1], stats["rounds"],
            stats["wall0"], stats["wall1"], stats["phase_p0"]))
    n = len(all_stats)
    avg_ratio8 = sum(st["ratio0"] for st in all_stats) / n
    avg_seals8 = total_seals8 / n
    print("\n  v8 avg territory_ratio=%.4f  avg_seals=%.2f" % (avg_ratio8, avg_seals8))

    # --- v7 for comparison ---
    print("\nPhase 2: v7 (p0) vs greedy-scorer opponent (p1)\n")
    total_seals7 = 0
    avg_ratio7 = 0.0
    for s in seeds:
        stats, _ = play_game(v7_decide, opp, s, record_states=False)
        total_seals7 += stats["seals"][0]
        avg_ratio7 += stats["ratio0"]
        print("  seed=%2d  v7=%3d opp=%3d ratio=%.3f seals[v7=%d opp=%d] rounds=%3d wall[v7=%d opp=%d]" % (
            s, stats["scores"][0], stats["scores"][1], stats["ratio0"],
            stats["seals"][0], stats["seals"][1], stats["rounds"],
            stats["wall0"], stats["wall1"]))
    avg_ratio7 /= n
    print("\n  v7 avg territory_ratio=%.4f  avg_seals=%.2f" % (avg_ratio7, total_seals7 / n))

    # --- IG: per-state epsilon-regularized KL (v7 -> v8) ---
    print("\nPhase 3: information gain v7 -> v8 (per-state eps-reg KL, eps=%.2f)" % EPSILON)
    ig = compute_ig(all_states, v7_decide, v8)
    print("  ig_kl=%.4f nats/decision  trajectory_kl=%.2f  disagree_rate=%.3f  n_states=%d  status=%s" % (
        ig["ig_kl"] or 0, ig.get("trajectory_kl", 0) or 0,
        ig.get("disagree_rate", 0) or 0, ig.get("n_states", 0), ig["status"]))

    # --- save summary ---
    summary = {
        "v8_avg_ratio": round(avg_ratio8, 4),
        "v7_avg_ratio": round(avg_ratio7, 4),
        "v8_avg_seals": round(avg_seals8, 2),
        "v7_avg_seals": round(total_seals7 / n, 2),
        "ig_v7_to_v8": ig,
        "seeds": seeds,
        "epsilon": EPSILON,
        "v8_detail": [{"seed": seeds[i], "scores": st["scores"], "ratio": st["ratio0"],
                       "seals": st["seals"], "phases": st["phase_p0"]} for i, st in enumerate(all_stats)],
    }
    with open(os.path.join(DATA, "v8_sandbox.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nSaved runs/v8_sandbox.json")
    print("\nSUMMARY: v8 seals %.1fx more often than v7; territory_ratio %.3f -> %.3f" % (
        (avg_seals8 / max(0.01, total_seals7 / n)), avg_ratio7, avg_ratio8))


if __name__ == "__main__":
    main()
