
"""One-shot fix: sync ALL curves.json sections from CI summary.json data.

Defects fixed:
  1. points[6] (v8) and points[7] (v9) had stale territory_ratio / ig_kl /
     details from the buggy engine (U-turn = game-over). Now matched to
     CI summaries produced on the fixed engine.
  2. hl_main.iterations[0] (v8) had stale ig_kl=1.868 (old Laplace method)
     and stale details. Now null IG (first HL version, no predecessor) and
     correct per-seed data.
  3. hl_main.baseline was 0.4177 (stale); now 0.3828.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
CI_DIR = os.path.join(_HERE, "agentbench_data", "runs", "26_snakego")
CURVES = os.path.join(_HERE, "runs", "curves.json")


def _latest_summary(agent):
    """Return the most recent summary.json for an agent."""
    agent_dir = os.path.join(CI_DIR, agent)
    if not os.path.isdir(agent_dir):
        raise FileNotFoundError(agent_dir)
    runs = sorted(os.listdir(agent_dir))
    if not runs:
        raise FileNotFoundError("no runs under " + agent_dir)
    latest = os.path.join(agent_dir, runs[-1], "summary.json")
    return json.load(open(latest, encoding="utf-8"))


def _ci_details_to_curves(seed_details):
    """Convert CI seed_details format to curves.json details format."""
    out = []
    for sd in seed_details:
        out.append({
            "seed": sd["seed"],
            "scores": [sd["my_score"], sd["opp_score"]],
            "ratio": sd["ratio"],
            "seals": sd["seals"],
            "rounds": sd["rounds"],
            "snakes0": sd["snakes0"],
        })
    return out


def main():
    print("=== fix_data.py: sync curves.json from CI summaries ===")
    print()

    sum8 = _latest_summary("hl_v8_loopclose")
    sum9 = _latest_summary("hl_v9_survive")

    print("  v8: territory_ratio={}  avg_my={}  avg_opp={}".format(
        sum8["territory_ratio"], sum8["avg_my_score"], sum8["avg_opp_score"]))
    print("  v9: territory_ratio={}  avg_my={}  avg_opp={}  ig_kl={}".format(
        sum9["territory_ratio"], sum9["avg_my_score"], sum9["avg_opp_score"],
        sum9["ig_kl"]))

    curves = json.load(open(CURVES, encoding="utf-8"))

    v8_details = _ci_details_to_curves(sum8["seed_details"])
    v9_details = _ci_details_to_curves(sum9["seed_details"])

    # Fix 1: top-level score_iteration
    curves["score_iteration"]["ratio"] = [sum8["territory_ratio"], sum9["territory_ratio"]]
    curves["score_iteration"]["score"] = [sum8["avg_my_score"], sum9["avg_my_score"]]

    # Fix 2: top-level ig_iteration
    curves["ig_iteration"]["ig_kl"] = [None, sum9["ig_kl"]]

    # Fix 3: points[] entries for v8 (iter 6) and v9 (iter 7)
    for p in curves["points"]:
        if p.get("label") == "v8_loopclose":
            p["territory_ratio"] = sum8["territory_ratio"]
            p["avg_my"] = sum8["avg_my_score"]
            p["avg_opp"] = sum8["avg_opp_score"]
            p["ig_kl"] = None
            p["ig_status"] = "no_baseline"
            p["details"] = v8_details
        elif p.get("label") == "v9_survive":
            p["territory_ratio"] = sum9["territory_ratio"]
            p["avg_my"] = sum9["avg_my_score"]
            p["avg_opp"] = sum9["avg_opp_score"]
            p["ig_kl"] = sum9["ig_kl"]
            p["ig_status"] = "ok"
            p["n_states"] = sum9["ig_n_states"]
            p["disagree_rate"] = sum9["ig_disagree_rate"]
            p["details"] = v9_details

    # Fix 4: hl_main section
    hm = curves["hl_main"]
    hm["baseline"] = sum8["territory_ratio"]
    for it in hm["iterations"]:
        if it["hl_iter"] == 1:
            it["territory_ratio"] = sum8["territory_ratio"]
            it["avg_my"] = sum8["avg_my_score"]
            it["avg_opp"] = sum8["avg_opp_score"]
            it["ig_kl"] = None
            it["ig_status"] = "no_baseline"
            it["details"] = v8_details
        elif it["hl_iter"] == 2:
            it["territory_ratio"] = sum9["territory_ratio"]
            it["avg_my"] = sum9["avg_my_score"]
            it["avg_opp"] = sum9["avg_opp_score"]
            it["ig_kl"] = sum9["ig_kl"]
            it["ig_status"] = "ok"
            it["n_states"] = sum9["ig_n_states"]
            it["disagree_rate"] = sum9["ig_disagree_rate"]
            it["details"] = v9_details

    json.dump(curves, open(CURVES, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print()
    print("  curves.json fully synced from CI data.")

    # Quick consistency self-check
    print()
    print("  Consistency check:")
    si = curves["score_iteration"]["ratio"]
    hm_r = [it["territory_ratio"] for it in hm["iterations"]]
    pt_r = [p["territory_ratio"] for p in curves["points"] if p.get("category") == "hl_structural"]
    ok1 = si == hm_r
    ok2 = si == pt_r
    print("    score_iteration == hl_main: {}  ({} vs {})".format(ok1, si, hm_r))
    print("    score_iteration == points:  {}  ({} vs {})".format(ok2, si, pt_r))
    ig_top = curves["ig_iteration"]["ig_kl"]
    ig_hm = [it.get("ig_kl") for it in hm["iterations"]]
    ok3 = ig_top == ig_hm
    print("    ig_iteration   == hl_main:  {}  ({} vs {})".format(ok3, ig_top, ig_hm))

    if ok1 and ok2 and ok3:
        print()
        print("  ALL CONSISTENT")
    else:
        print()
        print("  MISMATCH DETECTED")


if __name__ == "__main__":
    main()
