"""Produce version-aligned score-iteration and IG-iteration curves.

Consumes a VersionStore (from loop.py) or a list of replay files ordered by
iteration, and emits two aligned series:
  score(iter)  -- my territory score at each iteration's eval
  ig_kl(iter)  -- KL(prev_behaviour || cur_behaviour); None for iter 0
                  and for any iteration whose replay is incomplete (desync /
                  too-few-moves). Missing is preserved as null, never faked.

Output: a JSON document with both series + an ASCII sparkline to stdout, and a
plain-text companion describing each point (won/lost/incomplete). The caller
can pipe stdout to a file; nothing here writes directly (sandbox-safe).
"""
import json
import glob
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snakego import ig as IG


def _sparkline(values):
    """Tiny ASCII sparkline for a numeric series (None -> gap)."""
    blocks = " ▁▂▃▄▅▆▇█"
    nums = [v for v in values if v is not None]
    if not nums:
        return ""
    lo, hi = min(nums), max(nums)
    span = hi - lo or 1.0
    out = []
    for v in values:
        if v is None:
            out.append("·")
        else:
            idx = int((v - lo) / span * (len(blocks) - 1))
            out.append(blocks[idx])
    return "".join(out)


def from_replay_dir(replay_dir, pattern="iter*_learn_s*.json"):
    """Build series from replay files named iter{N}_learn_s{S}.json."""
    files = glob.glob(os.path.join(replay_dir, pattern))
    # parse iteration index from filename
    def itnum(f):
        n = os.path.basename(f)
        return int(n.split("_")[0].replace("iter", ""))
    files.sort(key=itnum)
    series = []
    prev_counts = None
    for f in files:
        it = itnum(f)
        rep = json.load(open(f, encoding="utf-8"))
        counts = {}
        for r, p, sid, op in rep["ops"]:
            if p == 0:
                counts[op] = counts.get(op, 0) + 1
        ig = None
        ig_status = "no_baseline" if prev_counts is None else "ok"
        if prev_counts is not None:
            r = IG.ig_version_transition(prev_counts, counts)
            ig = r["ig_kl"]
            ig_status = r["status"]
        series.append({
            "iter": it,
            "my_score": rep.get("scores", [0, 0])[0],
            "hu_score": rep.get("scores", [0, 0])[1],
            "won": 1 if rep.get("winner") == 0 else 0,
            "rounds": rep.get("rounds"),
            "ig_kl": ig,
            "ig_status": ig_status,
            "replay": os.path.basename(f),
            "error": rep.get("error"),
        })
        prev_counts = counts
    return series


def emit(series, title="iteration"):
    """Print aligned curves + a companion description to stdout."""
    iters = [s["iter"] for s in series]
    scores = [s["my_score"] for s in series]
    igs = [s["ig_kl"] for s in series]
    out = {
        "title": title,
        "score_iteration": {"iter": iters, "score": scores},
        "ig_iteration": {"iter": iters, "ig_kl": igs},
        "points": series,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n# %s curves (ASCII)" % title)
    print("score  : %s  (min=%s max=%s)" % (
        _sparkline(scores), min(scores), max(scores)))
    print("ig_kl  : %s" % _sparkline(igs))
    print("\nper-point:")
    for s in series:
        flag = "WON " if s["won"] else ("INC " if s["ig_kl"] is None and s["ig_status"] != "ok" else "LOST")
        ig_str = "%.4f" % s["ig_kl"] if s["ig_kl"] is not None else "n/a(%s)" % s["ig_status"]
        print("  iter %d  %s  my=%-4d hu=%-4d  ig_kl=%s  err=%s" % (
            s["iter"], flag, s["my_score"], s["hu_score"], ig_str, s["error"]))


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "runs"
    # Prefer the persisted curves.json (eval series from the loop) if present;
    # fall back to scanning learn replays for raw score data.
    curves_path = os.path.join(d, "curves.json")
    if os.path.exists(curves_path):
        data = json.load(open(curves_path, encoding="utf-8"))
        series = []
        for p in data["points"]:
            series.append({
                "iter": p["iter"],
                "my_score": p.get("my_score", 0),
                "hu_score": p.get("hu_score", 0),
                "won": p.get("won", 0),
                "rounds": p.get("rounds", 0),
                "ig_kl": p.get("ig_kl"),
                "ig_status": p.get("ig_status", "ok"),
                "replay": "curves.json",
                "error": p.get("error"),
            })
        emit(series, title="snakego iteration (eval vs sample_ai)")
    else:
        emit(from_replay_dir(d), title="snakego iteration (learn replays)")
