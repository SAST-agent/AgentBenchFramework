"""True HL iteration: each version adds one interpretable rule. Compare all."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snakego.host import run_match, socket_player, python_player, Engine, generate_items, LENGTH, WIDTH, MAX_ROUND
from snakego.strategy_core import Weights, score_action, DecisionTrace
from snakego.experience import seed_experience
from snakego.versions import dict_to_weights
from snakego.decision_space import compute_mask
from snakego import board as B
from snakego import ig as IG
from collections import Counter

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
GAMES = [("rank13",1),("rank13",2),("rank13",3),("rank15",1),("rank15",2),("rank15",3)]

exp0 = seed_experience()
BASE_W = dict_to_weights(exp0.weights_snapshot)


def make_v0_decide():
    """v0: pure weighted scoring, no heuristic override."""
    def _d(eng, my_id):
        mask = compute_mask(eng)
        best, best_op = None, 1
        for op in (1,2,3,4,5,6):
            if not mask.legal[op]:
                continue
            dt = score_action(eng, op, BASE_W)
            if best is None or dt.total > best:
                best, best_op = dt.total, op
        return best_op
    return _d


def make_v1_decide():
    """v1: + survival hard gate (room < length+6 -> restrict to non-shrinking moves)."""
    base = make_v0_decide()
    def _d(eng, my_id):
        snake = eng.current_snake()
        if snake and len(snake.coord_list) >= 2:
            my_ids = {s.id for s in eng.my_snakes()}
            hx, hy = snake.coord_list[0]
            my_room = B.reachable_space(eng, hx, hy, my_ids)
            mask = compute_mask(eng)
            if my_room < snake.length + 6:
                safe = []
                for op in (1,2,3,4):
                    if not mask.legal[op]:
                        continue
                    nx, ny = B.head_after(snake, op)
                    if not (0 <= nx < eng.length and 0 <= ny < eng.width):
                        continue
                    if B.classify_move(eng, snake, op) == "dead":
                        continue
                    r = B.reachable_space(eng, nx, ny, my_ids)
                    if r >= my_room:
                        safe.append(op)
                if safe:
                    return safe[0]
        return base(eng, my_id)
    return _d


def make_v2_decide():
    """v2: + big seal detection (>=6 cells -> force seal)."""
    base = make_v1_decide()
    def _d(eng, my_id):
        snake = eng.current_snake()
        if snake:
            mask = compute_mask(eng)
            for op in (1,2,3,4):
                if not mask.legal[op]:
                    continue
                if B.classify_move(eng, snake, op) == "seal":
                    area = B.estimate_seal_area(eng, snake)
                    if area >= 6:
                        return op
        return base(eng, my_id)
    return _d


def make_v3_decide():
    """v3: + early split (r<50, n<3, len>=8 -> force split)."""
    base = make_v2_decide()
    def _d(eng, my_id):
        snake = eng.current_snake()
        if snake:
            mask = compute_mask(eng)
            my_ids = {s.id for s in eng.my_snakes()}
            n = len(eng.my_snakes())
            if (eng.current_round < 50 and n <= 2 and snake.length >= 8
                    and mask.legal[6]):
                hx, hy = snake.coord_list[0]
                room = B.reachable_space(eng, hx, hy, my_ids)
                if room >= snake.length + 6:
                    return 6
        return base(eng, my_id)
    return _d


def make_v4_decide():
    """v4: + U-turn forcing (straight 4+ segments, long, safe -> turn 90)."""
    base = make_v3_decide()
    def _d(eng, my_id):
        snake = eng.current_snake()
        if snake and snake.length >= 10 and len(snake.coord_list) >= 5:
            cl = snake.coord_list
            h0x, h0y = cl[0]
            h1x, h1y = cl[1]
            h4x, h4y = cl[4]
            dx1, dy1 = h0x - h1x, h0y - h1y
            dx2, dy2 = h0x - h4x, h0y - h4y
            is_straight = ((dx1 != 0 and dx2 != 0 and dx1 * dx2 > 0) or
                          (dy1 != 0 and dy2 != 0 and dy1 * dy2 > 0))
            if is_straight:
                my_ids = {s.id for s in eng.my_snakes()}
                hx, hy = cl[0]
                my_room = B.reachable_space(eng, hx, hy, my_ids)
                if my_room >= snake.length + 6:
                    mask = compute_mask(eng)
                    if dx1 != 0:
                        turns = [3, 4]
                    else:
                        turns = [1, 2]
                    safe_t = []
                    for op in turns:
                        if not mask.legal[op]:
                            continue
                        if B.classify_move(eng, snake, op) == "dead":
                            continue
                        nx, ny = B.head_after(snake, op)
                        r = B.reachable_space(eng, nx, ny, my_ids)
                        if r >= snake.length + 2:
                            safe_t.append((op, r))
                    if safe_t:
                        safe_t.sort(key=lambda x: -x[1])
                        return safe_t[0][0]
        return base(eng, my_id)
    return _d


VERSIONS = [
    ("v0_baseline",   "weighted scoring only",                    make_v0_decide),
    ("v1_survival",   "+ survival gate",                           make_v1_decide),
    ("v2_bigseal",    "+ big seal detection",                     make_v2_decide),
    ("v3_earlysplit", "+ early split",                            make_v3_decide),
    ("v4_uturn",      "+ U-turn forcing",                         make_v4_decide),
]


def eval_decide(decide_fn, label):
    ratios = []
    details = []
    all_ops = []
    for opp_name, seed in GAMES:
        me = python_player("me", decide_fn)
        opp = socket_player(opp_name, "bin_humans")
        r = run_match(me, opp, seed=seed, time_limit_per_move=10.0, record=True)
        my_s, opp_s = r.scores[0], r.scores[1]
        ratio = my_s / max(1, my_s + opp_s)
        ratios.append(ratio)
        details.append({"opp": opp_name, "seed": seed, "my": my_s, "opp": opp_s,
                        "ratio": round(ratio, 4), "r": r.rounds, "err": r.error is not None})
        my_ops = Counter(op for rnd,p,sid,op in r.ops if p == 0)
        all_ops.append(my_ops)
    avg = sum(ratios) / len(ratios)
    avg_my = sum(d["my"] for d in details) / len(details)
    avg_opp = sum(d["opp"] for d in details) / len(details)
    print("%s: ratio=%.4f avg_my=%.0f avg_opp=%.0f" % (label, avg, avg_my, avg_opp))
    for d in details:
        t = "ERR" if d["err"] else ""
        print("  %s s%d my=%4d opp=%4d ratio=%.3f r=%4d %s" % (d["opp"], d["seed"], d["my"], d["opp"], d["ratio"], d["r"], t))
    return {"ratio": round(avg, 4), "avg_my": round(avg_my,1), "avg_opp": round(avg_opp,1),
            "details": details, "ops": all_ops}


def main():
    print("=== SnakeGo HL Rule-Addition Iteration ===")
    print("Each version adds ONE interpretable rule. Eval vs rank13/15 x3 seeds each.\n")
    t0 = time.time()
    results = []
    prev_ops = None
    for i, (label, desc, make_fn) in enumerate(VERSIONS):
        print("--- iter %d: %s (%s) ---" % (i, label, desc))
        res = eval_decide(make_fn(), label)
        res["iter"] = i
        res["label"] = label
        res["desc"] = desc
        # IG: compare action distribution vs previous version
        if prev_ops:
            # aggregate ops across games for current and previous
            cur_total = Counter()
            prev_total = Counter()
            for o in res["ops"]:
                cur_total.update(o)
            for o in prev_ops:
                prev_total.update(o)
            ig = IG.ig_version_transition(dict(prev_total), dict(cur_total))
            res["ig_kl"] = ig.get("ig_kl")
            res["ig_status"] = ig.get("status", "ok")
        else:
            res["ig_kl"] = None
            res["ig_status"] = "no_baseline"
        prev_ops = res["ops"]
        results.append(res)
        print()

    # Save
    curves = {"score_iteration": {"iter": [], "ratio": [], "score": []},
              "ig_iteration": {"iter": [], "ig_kl": []},
              "points": []}
    for r in results:
        curves["score_iteration"]["iter"].append(r["iter"])
        curves["score_iteration"]["ratio"].append(r["ratio"])
        curves["score_iteration"]["score"].append(r["avg_my"])
        curves["ig_iteration"]["iter"].append(r["iter"])
        kl = r["ig_kl"]
        curves["ig_iteration"]["ig_kl"].append(kl)
        curves["points"].append({
            "iter": r["iter"], "label": r["label"], "desc": r["desc"],
            "territory_ratio": r["ratio"], "avg_my": r["avg_my"], "avg_opp": r["avg_opp"],
            "ig_kl": kl, "ig_status": r["ig_status"],
            "details": [{k:v for k,v in d.items() if k != "ops"} for d in r["details"]],
        })
    with open(os.path.join(DATA, "curves.json"), "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2, ensure_ascii=False)
    print("Saved curves.json")
    print("Total time: %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
