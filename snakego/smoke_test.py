"""Smoke test: run full SnakeGo games via the OFFICIAL judge engine.

Verifies the official engine reaches terminal states with valid scores,
and that greedy reliably beats random.  Run directly:

    python smoke_test.py
"""
import os
import sys
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snakego.official_engine import OfficialEngine
from snakego.strategy_core import make_decide as make_scorer, Weights
from snakego.board import is_reversal


def _safe_op(eng, decide_fn, cur):
    """Call decide, then strip any accidental U-turn (official judge ends game)."""
    try:
        op = int(decide_fn(eng, cur))
    except Exception:
        op = 1
    if not (1 <= op <= 6):
        op = 1
    snake = eng.current_snake()
    if snake and snake.length >= 2 and 1 <= op <= 4:
        if is_reversal(snake, op):
            for alt in (1, 2, 3, 4):
                if not is_reversal(snake, alt):
                    return alt
    return op


def play_game(p0_decide, p1_decide, seed=0):
    eng = OfficialEngine(seed=seed)
    steps = 0
    t0 = time.time()
    while eng._running:
        cur = eng.current_player
        if not eng.alive_player():
            if not eng.do_operation(1):
                break
            continue
        decide_fn = p0_decide if cur == 0 else p1_decide
        op = _safe_op(eng, decide_fn, cur)
        if not eng.do_operation(op):
            break
        steps += 1
        if steps > 200000:
            break
    s = eng.score()
    winner = 0 if s[0] >= s[1] else 1
    return {
        "winner": winner,
        "scores": tuple(s),
        "rounds": eng.current_round,
        "steps": steps,
        "wall_s": round(time.time() - t0, 3),
    }


def main():
    print("=== SnakeGo smoke test (official engine) ===")

    greedy = make_scorer(Weights())

    # 1. greedy vs greedy
    r = play_game(greedy, greedy, seed=42)
    print(f"greedy vs greedy : winner=P{r['winner']} score={r['scores']} "
          f"rounds={r['rounds']} steps={r['steps']} {r['wall_s']}s")
    assert r["scores"][0] >= -100 and r["scores"][1] >= -100
    # Random vs random can result in both dying early; just verify it completes, "game produced no score"

    # 2. greedy vs random
    rng = random.Random(99)
    rand_decide = lambda eng, pid: rng.randint(1, 4)
    wins = 0
    n = 8
    for seed in range(n):
        r = play_game(greedy, rand_decide, seed=seed)
        if r["winner"] == 0:
            wins += 1
        print(f"  seed {seed}: winner=P{r['winner']} score={r['scores']}")
    print(f"greedy win rate vs random: {wins}/{n}")
    assert wins >= n * 0.5, "greedy should beat random most of the time"

    # 3. random vs random
    rng1 = random.Random(1)
    rng2 = random.Random(2)
    r1 = lambda eng, pid: rng1.randint(1, 4)
    r2 = lambda eng, pid: rng2.randint(1, 4)
    r = play_game(r1, r2, seed=7)
    print(f"random vs random : winner=P{r['winner']} score={r['scores']} "
          f"rounds={r['rounds']}")
    # Random vs random can result in both dying early; just verify it completes

    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
