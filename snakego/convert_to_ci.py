#!/usr/bin/env python3
"""
Convert flat SnakeGo HL iteration data -> AgentBenchFrame CI data contract.

Reads from the flat runs/ directory (versions.json, multiseed_summary.json,
curves.json, experience.json) and produces:
  agentbench_data/runs/26_snakego/{agent}/{run_id}/run.toml + summary.json + events.jsonl

Agents produced:
  - rule_v0 ... rule_v3   (type=rule_iter, HL iteration snapshots)
  - rule_v3_eval          (type=eval, multiseed vs rank15 + sample_ai)
  - human_rank15          (type=eval, human baseline from our perspective)
  - human_sample_ai       (type=eval, human baseline from our perspective)
"""
import json, os, hashlib

DEFAULT_RUNS = r"C:\Users\53125\Desktop\snakego\runs"
RUNS = os.environ.get("SNAKEGO_RUNS", DEFAULT_RUNS)

CI_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "agentbench_data", "runs", "26_snakego")

with open(os.path.join(RUNS, "versions.json"), encoding="utf-8") as f:
    versions = json.load(f)
with open(os.path.join(RUNS, "multiseed_summary.json"), encoding="utf-8") as f:
    multiseed = json.load(f)
with open(os.path.join(RUNS, "curves.json"), encoding="utf-8") as f:
    curves = json.load(f)
with open(os.path.join(RUNS, "experience.json"), encoding="utf-8") as f:
    experience = json.load(f)


def make_run_id(label, hour=10):
    ts = "20260727_%02d00" % hour
    h = hashlib.md5(label.encode()).hexdigest()[:8]
    return "%s_%s" % (ts, h)


def _toml_val(v):
    """Format a Python value as a TOML literal."""
    if v is None:
        return '""'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_val(x) for x in v) + "]"
    # string
    return '"%s"' % str(v).replace('"', "'")


def write_run(agent, run_id, run_type, created, weights, ig, lesson,
              total_steps, total_episodes, win_rate, opponent, h2h,
              extra_config=None, events=None):
    run_dir = os.path.join(CI_ROOT, agent, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # run.toml
    toml_lines = [
        "[run]",
        'run_id = "%s"' % run_id,
        'game = "26_snakego"',
        'agent = "%s"' % agent,
        'type = "%s"' % run_type,
        'created = "%s"' % created,
        'git_commit = "hl_iter"',
        "started_at = 1785000000.0",
        "finished_at = 1785000120.0",
        "total_steps = %d" % total_steps,
        "total_episodes = %d" % total_episodes,
        "",
        "[config]",
    ]
    for wk, wv in weights.items():
        toml_lines.append("%s = %s" % (wk, _toml_val(wv)))
    if ig:
        igv = ig.get("ig_kl")
        if igv is not None:
            toml_lines.append("ig_kl = %s" % _toml_val(igv))
        else:
            toml_lines.append('ig_kl = ""')
        toml_lines.append('ig_status = "%s"' % ig.get("status", ""))
    if lesson:
        toml_lines.append('lesson = "%s"' % lesson.replace('"', "'"))
    if extra_config:
        for ck, cv in extra_config.items():
            toml_lines.append("%s = %s" % (ck, _toml_val(cv)))
    with open(os.path.join(run_dir, "run.toml"), "w", encoding="utf-8") as f:
        f.write("\n".join(toml_lines) + "\n")

    # summary.json
    summary = {
        "run_id": run_id,
        "game": "26_snakego",
        "agent": agent,
        "run_type": run_type,
        "created": created,
        "git_commit": "hl_iter",
        "wall_hours": 0.03,
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "win_rate": win_rate,
        "best_elo": 1000,
        "final_elo": 1000,
        "elo_history": [{"step": total_steps, "elo": 1000}],
        "h2h": h2h,
        "resource_summary": {},
    }
    summary["avg_reward_per_episode"] = round(
        summary.get("total_reward", 0) / max(1, total_episodes), 2) if "total_reward" in summary else 0.0
    if ig:
        summary["ig_kl"] = ig.get("ig_kl")
        summary["ig_status"] = ig.get("status")
    if extra_config:
        summary["config"] = extra_config
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # events.jsonl
    with open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8") as f:
        if events:
            for ev in events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        else:
            f.write(json.dumps({"event": "episode", "opponent": opponent,
                 "won": 1 if win_rate >= 0.5 else 0,
                 "rounds": total_steps}, ensure_ascii=False) + "\n")
    return run_dir


# 1. HL iteration runs (rule_v0 .. rule_v3)
for snap in versions:
    it = snap["iteration"]
    agent = "rule_v%d" % it
    run_id = make_run_id(agent, hour=10 + it)
    created = "2026-07-27T%02d:00:00Z" % (10 + it)
    es = snap["eval_summary"]
    ig = snap.get("ig", {})
    opp = es.get("human", "sample_ai")
    events = [{"event": "episode", "iteration": it, "opponent": opp,
        "my_score": es.get("my", 0), "hu_score": es.get("hu", 0),
        "won": es.get("won", 0), "rounds": es.get("rounds", 0),
        "ig_kl": ig.get("ig_kl"), "ig_status": ig.get("status"),
        "lesson": snap.get("note", "")}]
    d = write_run(agent, run_id, "rule_iter", created, snap["weights"], ig,
        snap.get("note", ""), es.get("rounds", 0), 1,
        float(es.get("won", 0)), opp, {agent: {opp: float(es.get("won", 0))}},
        events=events)
    print("  HL  %s -> %s" % (agent, d))

# 2. Eval run: rule_v3 multiseed
eval_agent = "rule_v3_eval"
eval_id = make_run_id("rule_v3_eval", hour=14)
h2h_eval = {"rule_v3": {}}
te, ts, we = 0, 0, 0
eval_events = []
for opp, games in multiseed.items():
    ow = 0
    for g in games:
        te += 1; ts += g.get("rounds", 0)
        if g.get("win") == 1: ow += 1; we += 1
        eval_events.append({"event": "episode", "opponent": opp,
            "seed": g.get("seed"), "my_score": g.get("my", 0),
            "hu_score": g.get("hu", 0), "win": g.get("win", 0),
            "rounds": g.get("rounds", 0), "error": g.get("err", False)})
    h2h_eval["rule_v3"][opp] = round(ow / len(games), 4) if games else 0
d = write_run(eval_agent, eval_id, "eval", "2026-07-27T14:00:00Z", {},
    None, "", ts, te, round(we / max(1, te), 4), "multiseed", h2h_eval,
    extra_config={"seeds": [1, 2, 3, 4, 5], "opponents": list(multiseed.keys())},
    events=eval_events)
print("  EVAL %s -> %s" % (eval_agent, d))

# 3. Human baseline runs
for opp_name, games in multiseed.items():
    ha = "human_%s" % opp_name
    run_id = make_run_id(ha, hour=15)
    teh = len(games)
    tsh = sum(g.get("rounds", 0) for g in games)
    wh = sum(1 for g in games if g.get("win") != 1)
    wrh = round(wh / max(1, teh), 4)
    he = [{"event": "episode", "opponent": "rule_v3", "seed": g.get("seed"),
        "my_score": g.get("hu", 0), "hu_score": g.get("my", 0),
        "win": 0 if g.get("win") == 1 else 1, "rounds": g.get("rounds", 0),
        "error": g.get("err", False)} for g in games]
    d = write_run(ha, run_id, "eval", "2026-07-27T15:00:00Z", {}, None,
        "Human baseline (ranked player)", tsh, teh, wrh, "rule_v3",
        {ha: {"rule_v3": wrh}},
        extra_config={"source": "multiseed_eval", "player_rank": opp_name},
        events=he)
    print("  HUMAN %s -> %s" % (ha, d))

print("\n=== CI conversion complete ===")
print("Source: %s" % RUNS)
print("Output: %s" % CI_ROOT)
print("Agents: rule_v0-v3 (HL), rule_v3_eval (eval), human_rank15, human_sample_ai")
print("Lessons: %d active, %d archived" % (
    len(experience.get("lessons", [])), len(experience.get("archived", []))))
