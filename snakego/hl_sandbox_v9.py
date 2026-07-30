# Corrected HL sandbox: deepcopy-based IG measurement (fixes the snake-ID
# corruption in hl_sandbox.py's hand-rolled _restore) + v8-vs-v9 evaluation.
#
# Measurement contract (research doc sections 4.1 / 15):
#   - local_policy_kl per decision state, epsilon-regularized so deterministic
#     HL policies yield finite, comparable KL (not 0 / +inf).
#   - decision states = player-0 move points SHARED across both versions, taken
#     from a v8 rollout, then re-decided by both v8 and v9 on the SAME state.
#   - deepcopy guarantees bit-exact state (snake IDs, walls, items) so the two
#     policies are scored on identical ground truth.
import copy
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snakego.engine import Engine
from snakego.host import generate_items, LENGTH, WIDTH, MAX_ROUND
from snakego import board as B
from snakego.decision_space import compute_mask, SUPPORT
from snakego.strategy_loop import decide as v8_decide_loop  # noqa: F401 (v7 compat)
from snakego.ig import per_state_ig as _compute_ig, EPSILON
from snakego import strategy_v8 as V8
from snakego import strategy_v9 as V9

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
# EPSILON now imported from snakego.ig (canonical source of truth)
SEEDS = [1, 2, 3, 7, 11, 23]


class _DecidePlayer:
    def __init__(self, name, decide_fn):
        self.name = name
        self.decide_fn = decide_fn

    def act(self, eng, pid):
        op = int(self.decide_fn(eng, pid))
        if not (1 <= op <= 6):
            op = 1
        return op


def _count_walls(eng, camp):
    return sum(1 for row in eng.wall_map for v in row if v == camp)


def play_game(decide0, decide1, seed, record_states=False):
    eng = Engine(LENGTH, WIDTH, MAX_ROUND, generate_items(seed))
    p0 = _DecidePlayer("p0", decide0)
    p1 = _DecidePlayer("p1", decide1)
    seals = [0, 0]
    phase_log = {0: {}}
    decision_states = []
    while True:
        cur = eng.current_player
        if not eng.alive_player():
            if not eng.do_operation(0):
                break
            continue
        player = p0 if cur == 0 else p1
        if cur == 0 and record_states:
            decision_states.append(copy.deepcopy(eng))
        prev = _count_walls(eng, cur)
        op = player.act(eng, cur)
        if cur == 0:
            try:
                _, info = V9.decide(eng, cur, trace=True)
                ph = info.get("phase", "?")
                phase_log[0][ph] = phase_log[0].get(ph, 0) + 1
            except Exception:
                pass
        running = eng.do_operation(op)
        if _count_walls(eng, cur) > prev:
            seals[cur] += 1
        if not running:
            break
    scores = eng.score()
    stats = {
        "scores": scores,
        "ratio0": scores[0] / max(1, scores[0] + scores[1]),
        "seals": seals,
        "rounds": eng.current_round,
        "wall0": _count_walls(eng, 0),
        "wall1": _count_walls(eng, 1),
        "phase_p0": phase_log[0],
        "snakes0": len(eng.snake_list_0),
    }
    return stats, decision_states


def eps_reg_kl(old_op, new_op, legal_mask, epsilon=EPSILON):
    legal = [a for a in SUPPORT if legal_mask.get(a, False)]
    if not legal:
        return 0.0
    n = len(legal)
    kl = 0.0
    for a in legal:
        p = (1 - epsilon) * (1.0 if a == old_op else 0.0) + epsilon / n
        q = (1 - epsilon) * (1.0 if a == new_op else 0.0) + epsilon / n
        kl += q * math.log((q + 1e-12) / (p + 1e-12))
    return kl


def compute_ig(states, old_decide, new_decide):
    # Per-state eps-reg KL over SHARED decision states (research doc 15).
    # Each state is a deepcopy, so both policies decide on identical ground.
    traces = []
    disagree = 0
    for snap in states:
        # old policy decides on its own copy; new policy on another copy.
        e_old = copy.deepcopy(snap)
        e_new = copy.deepcopy(snap)
        try:
            old_op = int(old_decide(e_old, 0))
        except Exception:
            old_op = 1
        try:
            new_op = int(new_decide(e_new, 0))
        except Exception:
            new_op = 1
        if not (1 <= old_op <= 6):
            old_op = 1
        if not (1 <= new_op <= 6):
            new_op = 1
        mask = compute_mask(snap)
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


def _eval(decide0, label, opp_decide, seeds):
    print("  %s (p0) vs opponent (p1):" % label)
    all_stats, all_states = [], []
    tot_seals = 0
    for s in seeds:
        st, states = play_game(decide0, opp_decide, s, record_states=True)
        all_stats.append(st)
        all_states.extend(states)
        tot_seals += st["seals"][0]
        print("    seed=%-2d score=[%3d,%3d] ratio=%.3f seals=[%d,%d] r=%3d "
              "snakes0=%d phases=%s" % (
                  s, st["scores"][0], st["scores"][1], st["ratio0"],
                  st["seals"][0], st["seals"][1], st["rounds"], st["snakes0"],
                  st["phase_p0"]))
    n = len(seeds)
    avg = sum(st["ratio0"] for st in all_stats) / n
    print("    -> avg_ratio=%.4f avg_seals=%.2f" % (avg, tot_seals / n))
    return all_stats, all_states, avg


def main():
    print("=== SnakeGo HL Sandbox v9 (deepcopy IG) ===\n")
    v8 = V8.make_decide()
    v9 = V9.make_decide()
    # Frozen evaluation opponent = the greedy territory scorer used by ALL prior
    # iterations (v0-v8), so the ratio is directly comparable across curves.json.
    from snakego.strategy_core import make_decide as make_scorer, Weights
    opp = make_scorer(Weights())

    print("Phase A: v9 vs frozen greedy-scorer opponent (eval-set comparable)")
    st9, states9, avg9 = _eval(v9, "v9", opp, SEEDS)

    # Collect decision states from a v8 rollout (the version we iterate FROM) so
    # that both v8 and v9 re-decide on identical ground for the KL (doc 15).
    print("\nPhase B: collecting v8 decision states for IG (quiet)")
    _, states8, _ = _eval(v8, "v8", opp, SEEDS)

    print("\nPhase C: information gain v8 -> v9 (per-state eps-reg KL, eps=%.2f)"
          % EPSILON)
    ig = _compute_ig(states8, v8, v9)
    print("  ig_kl=%.4f nats/decision  disagree_rate=%.3f  n_states=%d  status=%s"
          % (ig["ig_kl"] or 0, ig.get("disagree_rate", 0) or 0,
             ig.get("n_states", 0), ig["status"]))

    summary = {
        "epsilon": EPSILON,
        "seeds": SEEDS,
        "v9_avg_ratio": round(avg9, 4),
        "ig_v8_to_v9": ig,
        "v9_detail": [{"seed": SEEDS[i], "scores": st["scores"], "ratio": st["ratio0"],
                       "seals": st["seals"], "rounds": st["rounds"],
                       "snakes0": st["snakes0"], "phases": st["phase_p0"]}
                      for i, st in enumerate(st9)],
    }
    with open(os.path.join(DATA, "v9_sandbox.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nSaved runs/v9_sandbox.json")
    print("\nSUMMARY: v9 ratio %.3f vs frozen opponent; IG(v8->v9) %.3f nats/decision"
          % (avg9, ig["ig_kl"] or 0))


if __name__ == "__main__":
    main()
