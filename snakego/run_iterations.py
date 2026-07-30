"""Run the full iteration loop: play -> learn -> snapshot -> eval -> IG.

Uses the honest multi-opponent loop: learning rotates through rank06/09/11/13/15,
evaluation uses a FIXED pool (rank13 s1-s2, rank15 s1-s2) so territory_ratio
is comparable across iterations. Weight changes that degrade the aggregate
score are reverted; after 3 rejects, random perturbation escapes local optima.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snakego.loop import Loop

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def main(n_iters=6):
    loop = Loop(data_dir=DATA)
    print("=== SnakeGo HL Iteration (honest multi-opponent) ===")
    print("learn_pool=%s" % loop.learn_pool)
    print("eval_games=%s" % loop.eval_games)
    print("writable=%s\n" % loop._writable)

    for i in range(n_iters):
        entry = loop.step()
        ev = entry["eval"]
        ig = entry["ig"]
        kl = ig.get("ig_kl")
        kl_s = "%.4f" % kl if kl is not None else "n/a(%s)" % ig.get("status", "?")
        tag = "ACCEPT" if entry["accepted"] else "REVERT"
        print("iter %d [%s] vs=%-8s ratio=%.4f avg=%-4d opp=%-4d wins=%d/%d  ig_kl=%s" % (
            entry["iter"], tag, entry["learn_vs"],
            ev["territory_ratio"], ev["avg_score"], ev["avg_opp_score"],
            ev["wins"], ev["games"], kl_s))
        print("  lesson: %s" % entry["lesson"])
        ws = loop.experience.weights_snapshot
        print("  w: split=%.1f seal=%.1f trap=%.1f growth_max=%.1f" % (
            ws.get("split_value", 0), ws.get("seal_area", 0),
            ws.get("trap_penalty", 0), ws.get("max_growth_length", 0)))
        for g in entry["eval_detail"]:
            print("    %-8s s%d  my=%-4d opp=%-4d ratio=%.3f r=%-4d %s" % (
                g["opponent"], g["seed"], g["my"], g["opp"],
                g["ratio"], g["rounds"], "ERR" if g["error"] else ""))
        print()

    ok = loop.persist_all()
    print("persisted=%s" % ok)
    s = loop.series()
    print("\n=== Series ===")
    print("territory_ratio:", list(zip(s["iters"], s["ratios"])))
    print("ig_kl:          ", list(zip(s["iters"], s["ig_kl"])))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    main(n_iters=n)
