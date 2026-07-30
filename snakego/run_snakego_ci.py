"""
End-to-end runner: SnakeGo match through framework layers -> CI data.

Produces AgentBenchResults data-contract files:
    runs/26_snakego/{agent}/{run_id}/run.toml
    runs/26_snakego/{agent}/{run_id}/summary.json
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_FW_SRC = os.path.join(_HERE, 'AgentBenchFramework', 'src')
# PARENT for `from snakego.X import`; _FW_SRC for the framework package.
_PARENT = os.path.dirname(_HERE)
for p in (_PARENT, _FW_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from agentbench_frame.env import make_env
from agentbench_frame.agent import SnakeGoAgent
from agentbench_frame.tracking.run import Run
from snakego.strategy_core import Weights
from snakego.host import socket_player


def run_game(env, agent0, agent1, seed):
    obs = env.reset(seed=seed)
    agents = {0: agent0, 1: agent1}
    done = False
    while not done:
        pid = obs.player_id
        action = agents[pid].act(obs.to_dict())
        obs, reward, done, info = env.step(action)
    winner = info.get('winner', -1)
    scores = info.get('scores', [0, 0])
    return winner, scores, env.ops, env.eng.current_round


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent', default='rule_v0')
    ap.add_argument('--opponent', default=None)
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--run-type', default='rule_iter')
    args = ap.parse_args()

    run = Run.start(
        game='26_snakego', agent=args.agent, run_type=args.run_type,
        data_dir=args.data_dir,
        config={'opponent': args.opponent or 'self', 'seeds': args.seeds},
    )
    print(f'[ci] run_id={run.run_id}')
    print(f'[ci] run_dir={run.run_dir}')

    my_agent = SnakeGoAgent(name=args.agent, weights=Weights())

    if args.opponent:
        sp = socket_player(args.opponent, os.path.join(_HERE, 'bin_humans'))
        from agentbench_frame.agent import SnakeGoSubprocessAgent
        opp_agent = SnakeGoSubprocessAgent(args.opponent, sp)
        env = make_env('26_snakego', subprocess_players={1: sp})
    else:
        sp = None
        opp_agent = SnakeGoAgent(name='self_play', weights=Weights())
        env = make_env('26_snakego')

    wins = 0
    h2h = {}
    t0 = time.time()
    for i, seed in enumerate(args.seeds):
        try:
            winner, scores, ops, rounds = run_game(env, my_agent, opp_agent, seed)
        except Exception as e:
            print(f'[ci] seed {seed}: ERROR {e}')
            run.log_episode(reward=0, steps=0, winner=-1, info={'error': str(e)})
            continue
        won = 1 if winner == 0 else 0
        wins += won
        steps = len(ops)
        print(f'[ci] seed {seed}: my={scores[0]} opp={scores[1]} winner=P{winner} rounds={rounds} ops={steps}')
        run.log_episode(reward=float(scores[0]), steps=steps, winner=winner,
                       info={'my_score': scores[0], 'opp_score': scores[1], 'rounds': rounds})
        h2h.setdefault(args.agent, {})[args.opponent or 'self'] = won
        if sp:
            try: sp.close()
            except Exception: pass
            if i < len(args.seeds) - 1:
                sp2 = socket_player(args.opponent, os.path.join(_HERE, 'bin_humans'))
                env._subprocs = {1: sp2}
                opp_agent.sp = sp2

    elapsed = time.time() - t0
    n = max(1, len(args.seeds))
    win_rate = wins / n
    run.log_h2h(h2h)
    if h2h:
        opp_name = args.opponent or 'self'
        wr = h2h.get(args.agent, {}).get(opp_name, 0)
        run.log_elo(1000 + wr * 200)
    summary = run.finish()
    print()
    print(f'[ci] DONE in {elapsed:.1f}s')
    print(f'[ci] win_rate={win_rate:.1%} ({wins}/{n})')
    for fname in ('run.toml', 'summary.json'):
        p = os.path.join(run.run_dir, fname)
        print(f'[ci] {fname}: ' + ('OK' if os.path.isfile(p) else 'MISSING') + f' ({p})')
    env.close()


if __name__ == '__main__':
    main()
