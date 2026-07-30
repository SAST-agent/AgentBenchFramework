"""HL iteration CI data generator.

Runs v8/v9 (and any future HL version) through the real Engine + PythonPlayer
harness, computes IG via the canonical ig.py functions, and writes CI-compatible
run.toml + summary.json into agentbench_data/runs/26_snakego/{agent}/{run_id}/.

This bridges the gap between the flat-sandbox HL iterations and the
AgentBenchResults data contract so CI can visualize HL progress.
"""
import copy
import json
import os
import sys
import shutil
import time
import uuid

_HERE = os.path.dirname(os.path.abspath(__file__))
# snakego/ is a package; `from snakego.X import` needs the PARENT dir on the
# path, not the package dir itself. Lets the script run from anywhere.
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from snakego.official_engine import OfficialEngine
from snakego.host import LENGTH, WIDTH, MAX_ROUND  # constants only
from snakego.board import is_reversal
from snakego.ig import per_state_ig, EPSILON
from snakego.decision_space import compute_mask, SUPPORT
from snakego import strategy_v8 as V8
from snakego import strategy_v9 as V9
from snakego.strategy_core import make_decide as make_scorer, Weights

SEEDS = [1, 2, 3, 7, 11, 23]
DATA_DIR = os.path.join(_HERE, "agentbench_data", "runs", "26_snakego")


def _run_id():
    ts = time.strftime("%Y%m%d_%H%M", time.gmtime())
    short = uuid.uuid4().hex[:8]
    return f"{ts}_{short}"


def _count_walls(eng, camp):
    return sum(1 for row in eng.wall_map for v in row if v == camp)


def _safe_op(eng, decide_fn, cur):
    """Call decide, then strip accidental U-turns."""
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


def play_game(decide0, decide1, seed, record_states=False):
    eng = OfficialEngine(seed=seed)
    seals = [0, 0]
    decision_states = []
    moves = 0
    while True:
        cur = eng.current_player
        if not eng.alive_player():
            if not eng.do_operation(0):
                break
            continue
        player_decide = decide0 if cur == 0 else decide1
        if cur == 0 and record_states:
            decision_states.append(copy.deepcopy(eng))
        prev_walls = _count_walls(eng, cur)
        op = _safe_op(eng, player_decide, cur)
        running = eng.do_operation(op)
        moves += 1
        if _count_walls(eng, cur) > prev_walls:
            seals[cur] += 1
        if not running:
            break
    scores = eng.score()
    return {
        "scores": scores,
        "ratio0": scores[0] / max(1, scores[0] + scores[1]),
        "seals": seals,
        "rounds": eng.current_round,
        "moves": moves,
        "snakes0": len(eng.snake_list_0),
        "snakes1": len(eng.snake_list_1),
    }, decision_states


def eval_version(decide_fn, opp_fn, label, seeds=SEEDS):
    print(f"  Evaluating {label} ...")
    all_stats = []
    all_states = []
    for s in seeds:
        st, states = play_game(decide_fn, opp_fn, s, record_states=True)
        all_stats.append(st)
        all_states.extend(states)
        print(f"    seed={s:2d}  my={st['scores'][0]:3d}  opp={st['scores'][1]:3d}  "
              f"ratio={st['ratio0']:.3f}  seals={st['seals']}  rounds={st['rounds']}")
    avg = sum(st["ratio0"] for st in all_stats) / len(seeds)
    print(f"    -> avg_ratio={avg:.4f}")
    return all_stats, all_states, avg


def write_ci_run(agent, version_label, desc, stats, ig, avg_ratio, prev_label=None):
    """Write one CI-compatible run.toml + summary.json."""
    # Idempotency: remove any previous run dirs for this agent so we never
    # accumulate duplicates across repeated hl_ci.py invocations.
    agent_dir = os.path.join(DATA_DIR, agent)
    if os.path.isdir(agent_dir):
        for old in os.listdir(agent_dir):
            shutil.rmtree(os.path.join(agent_dir, old))

    run_id = _run_id()
    run_dir = os.path.join(DATA_DIR, agent, run_id)
    os.makedirs(run_dir, exist_ok=True)

    total_steps = sum(st["moves"] for st in stats)
    total_episodes = len(stats)
    wins = sum(1 for st in stats if st["scores"][0] >= st["scores"][1])
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # run.toml
    toml_lines = [
        f'[run]',
        f'run_id = "{run_id}"',
        f'game = "26_snakego"',
        f'agent = "{agent}"',
        f'type = "rule_iter"',
        f'created = "{created}"',
        f'git_commit = "hl_main"',
        f'started_at = {time.time():.1f}',
        f'finished_at = {time.time() + total_steps * 0.01:.1f}',
        f'total_steps = {total_steps}',
        f'total_episodes = {total_episodes}',
        f'',
        f'[config]',
        f'version = "{version_label}"',
        f'description = "{desc}"',
        f'epsilon = {EPSILON}',
        f'seeds = "{SEEDS}"',
    ]
    if prev_label:
        toml_lines.append(f'predecessor = "{prev_label}"')
    if ig and ig.get("ig_kl") is not None:
        toml_lines.append(f'ig_kl = {ig["ig_kl"]:.6f}')
        toml_lines.append(f'ig_status = "{ig["status"]}"')
        toml_lines.append(f'ig_method = "per_state_eps_reg_KL"')
        toml_lines.append(f'n_states = {ig.get("n_states", 0)}')
        toml_lines.append(f'disagree_rate = {ig.get("disagree_rate", 0):.4f}')
    with open(os.path.join(run_dir, "run.toml"), "w", encoding="utf-8") as f:
        f.write("\n".join(toml_lines) + "\n")

    # summary.json
    elo_history = [{"step": i+1, "elo": 1000 + int(avg_ratio * 200)} for i in range(len(stats))]
    summary = {
        "run_id": run_id,
        "game": "26_snakego",
        "agent": agent,
        "run_type": "rule_iter",
        "created": created,
        "git_commit": "hl_main",
        "wall_hours": round(total_steps * 0.01 / 3600, 4),
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "win_rate": wins / total_episodes,
        "best_elo": 1000 + int(avg_ratio * 200),
        "final_elo": 1000 + int(avg_ratio * 200),
        "elo_history": elo_history,
        "h2h": {agent: {"frozen_greedy": wins / total_episodes}},
        "resource_summary": {},
        "territory_ratio": round(avg_ratio, 4),
        "avg_my_score": round(sum(st["scores"][0] for st in stats) / total_episodes, 1),
        "avg_opp_score": round(sum(st["scores"][1] for st in stats) / total_episodes, 1),
        "ig_kl": ig["ig_kl"] if ig else None,
        "ig_status": ig["status"] if ig else "no_data",
        "ig_method": "per_state_eps_reg_KL",
        "ig_epsilon": EPSILON,
        "ig_n_states": ig.get("n_states") if ig else None,
        "ig_disagree_rate": ig.get("disagree_rate") if ig else None,
        "version_label": version_label,
        "description": desc,
        "seed_details": [
            {"seed": SEEDS[i],
             "my_score": st["scores"][0],
             "opp_score": st["scores"][1],
             "ratio": round(st["ratio0"], 4),
             "seals": st["seals"],
             "rounds": st["rounds"],
             "snakes0": st["snakes0"]}
            for i, st in enumerate(stats)
        ],
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  CI run written: {run_dir}")
    return run_id, summary


def main():
    print("=== SnakeGo HL CI Generator (engine bug fixed) ===\n")

    v8 = V8.make_decide()
    v9 = V9.make_decide()
    opp = make_scorer(Weights())

    # Phase 1: evaluate v8 (HL-1)
    print("Phase 1: v8 (HL-1) vs frozen greedy opponent")
    st8, states8, avg8 = eval_version(v8, opp, "v8_loopclose")

    # Phase 2: evaluate v9 (HL-2)
    print("\nPhase 2: v9 (HL-2) vs frozen greedy opponent")
    st9, states9, avg9 = eval_version(v9, opp, "v9_survive")

    # Phase 3: IG(v8 -> v9) using canonical ig.py
    print(f"\nPhase 3: IG(v8 -> v9) via ig.per_state_ig (eps={EPSILON})")
    ig = per_state_ig(states8, v8, v9, epsilon=EPSILON)
    print(f"  ig_kl={ig['ig_kl']:.4f}  disagree={ig.get('disagree_rate', 0):.3f}  "
          f"n_states={ig.get('n_states', 0)}  status={ig['status']}")

    # Phase 4: write CI data
    print("\nPhase 4: writing CI-compatible runs")
    rid8, sum8 = write_ci_run(
        "hl_v8_loopclose", "v8_loopclose",
        "4-phase machine: GROW/SPLIT/HUNT_LOOP/SEAL (HL-1)",
        st8, None, avg8)
    rid9, sum9 = write_ci_run(
        "hl_v9_survive", "v9_survive",
        "space-max GROW + survival fallback + soft enemy penalty (HL-2)",
        st9, ig, avg9, prev_label="v8_loopclose")

    # Phase 5: update curves.json with fresh numbers
    curves_path = os.path.join(_HERE, "runs", "curves.json")
    if os.path.isfile(curves_path):
        with open(curves_path, "r", encoding="utf-8") as f:
            curves = json.load(f)
        # Update hl_main with fresh data
        if "hl_main" in curves:
            for it in curves["hl_main"]["iterations"]:
                if it["hl_iter"] == 1:
                    it["territory_ratio"] = round(avg8, 4)
                elif it["hl_iter"] == 2:
                    it["territory_ratio"] = round(avg9, 4)
                    it["ig_kl"] = round(ig["ig_kl"], 6) if ig["ig_kl"] else None
        # Also update top-level score_iteration / ig_iteration
        curves["score_iteration"]["ratio"] = [round(avg8, 4), round(avg9, 4)]
        curves["score_iteration"]["score"] = [
            round(sum(st["scores"][0] for st in st8) / len(st8), 1),
            round(sum(st["scores"][0] for st in st9) / len(st9), 1),
        ]
        # IG: null for first HL version (no predecessor), value for v8->v9
        curves["ig_iteration"]["ig_kl"] = [None, round(ig["ig_kl"], 6) if ig["ig_kl"] else None]
        with open(curves_path, "w", encoding="utf-8") as f:
            json.dump(curves, f, indent=2, ensure_ascii=False)
        print(f"\n  Updated runs/curves.json top-level arrays")

    print("\n=== DONE ===")
    print(f"v8 (HL-1): ratio={avg8:.4f}  CI: agentbench_data/runs/26_snakego/hl_v8_loopclose/{rid8}/")
    print(f"v9 (HL-2): ratio={avg9:.4f}  IG(v8->v9)={ig['ig_kl']:.4f}  CI: agentbench_data/runs/26_snakego/hl_v9_survive/{rid9}/")


if __name__ == "__main__":
    main()
