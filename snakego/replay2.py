"""Deterministic replay reconstruction for the faithful engine port.

A replay file (as emitted by host.run_match) is fully deterministic given the
item list + seed, because engine.py is a bit-exact port of adk.hpp. This module
re-plays the op stream against a fresh Engine, yielding a per-step board frame
at every decision point. That is what the "Agent watches replay" skill consumes
and what the visualizer renders.

Replay file schema (JSON):
  config     : {length, width, max_round, seed}
  items      : [{x, y, time, type, param}, ...]   # spawn schedule (ordered)
  names      : [name0, name1]
  ops        : [[round, player, snake_id, op_type], ...]
  scores     : [s0, s1]
  winner     : 0 | 1 | -1
  end_reason : "normal" | "error" | ...
"""
from snakego.engine import Engine, Item


def load_replay(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rebuild_engine(replay):
    """Recreate the exact Engine state from a replay's items."""
    cfg = replay.get("config", {})
    length = cfg.get("length", 16)
    width = cfg.get("width", 16)
    max_round = cfg.get("max_round", 512)
    items = [Item(it["x"], it["y"], k, it["time"], it["type"], it["param"])
             for k, it in enumerate(replay["items"])]
    return Engine(length, width, max_round, items)


def iter_frames(replay, include_initial=True):
    """Yield (step, frame) for every recorded op.

    frame = {
        step, round, player, snake_id, op,          # the decision
        head, snake_len, n_my_snakes, n_enemy,      # context
        scores,                                     # territory score right now
        events,                                     # human-readable key events
    }
    """
    eng = rebuild_engine(replay)
    if include_initial:
        yield -1, _frame(eng, -1, replay, [])
    prev_scores = [0, 0]
    for step, (r, p, sid, op) in enumerate(replay["ops"]):
        events = []
        if not eng.alive_player():
            eng.do_operation(0)
            events.append("dead_player_skip")
        else:
            snake = eng.current_snake()
            snake_id_before = snake.id if snake else -1
            my_list_before = eng.my_snakes() if eng.current_player == p else []
            had_before = any(s.id == snake_id_before for s in my_list_before)
            old_len = snake.length if snake else 0
            had_railgun = (snake.railgun_item_id != -1) if snake else False
            eng.do_operation(op)
            # detect key events
            if op == 5:
                events.append("railgun_fired")
            elif op == 6:
                events.append("split")
            else:
                # genuine death: the operated snake is gone from its owner list
                cur = [s for s in (eng.snake_list_0 if p == 0 else eng.snake_list_1)
                       if s.id == sid]
                if had_before and not cur:
                    events.append("snake_died")
        scores = eng.score()
        if scores != prev_scores and step > 0:
            events.append("score_delta_%d_%d" % (
                scores[0] - prev_scores[0], scores[1] - prev_scores[1]))
        prev_scores = scores
        yield step, _frame(eng, step, replay, events, r, p, sid, op)


def _frame(eng, step, replay, events, r=0, p=-1, sid=-1, op=0):
    snake = None
    if sid >= 0:
        for s in (eng.snake_list_0 + eng.snake_list_1):
            if s.id == sid:
                snake = s
                break
    return {
        "step": step,
        "round": r,
        "player": p,
        "snake_id": sid,
        "op": op,
        "head": list(snake.coord_list[0]) if snake and snake.coord_list else None,
        "snake_len": snake.length if snake else 0,
        "n_my_snakes": len(eng.snake_list_0) if p == 0 else (
            len(eng.snake_list_1) if p == 1 else -1),
        "n_enemy": len(eng.snake_list_1) if p == 0 else (
            len(eng.snake_list_0) if p == 1 else -1),
        "scores": eng.score(),
        "n_my_walls": sum(row.count(p) for row in eng.wall_map) if p in (0, 1) else -1,
        "events": events,
    }


def summary(replay):
    """One-line human-readable summary of a replay."""
    names = replay.get("names", ["P0", "P1"])
    s = replay["scores"]
    w = replay["winner"]
    wn = names[w] if 0 <= w < 2 else "draw/error"
    return (f"{names[0]} vs {names[1]}: {s[0]}-{s[1]}, "
            f"winner={wn}, rounds={replay.get('rounds')}, "
            f"moves={replay.get('moves')}, reason={replay.get('end_reason')}")


def key_events(replay, max_events=40):
    """Extract the most informative events (seals, splits, deaths, big deltas)."""
    out = []
    for step, fr in iter_frames(replay, include_initial=False):
        for ev in fr["events"]:
            if any(k in ev for k in ("split", "railgun", "snake_died", "score_delta")):
                out.append((step, fr["round"], fr["player"], ev, fr["scores"]))
                if len(out) >= max_events:
                    return out
    return out
