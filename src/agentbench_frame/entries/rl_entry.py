"""
RL training entry point for AntWAR2.

When torch is available this runs a lightweight PPO loop over the
single-agent AntWar2Env (controlled player vs. an opponent callable) using
RLStrategy's AntWar2Policy. When torch is absent it falls back to a greedy
smoke test that simply plays a few episodes to verify the full pipeline
(env -> strategy -> tracking) works end to end.

Both paths produce a CI-compatible run.toml + summary.json with a
cost_summary block, and register the resulting strategy/checkpoint in the
Population.

Usage:
    python -m agentbench_frame.entries.rl_entry --episodes 4 --smoke
    python -m agentbench_frame.entries.rl_entry --episodes 200 \\
        --lr 3e-4 --ppo-epochs 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from agentbench_frame.env.antwar2_env import AntWar2Env
from agentbench_frame.env._antwar2.backend.model import Operation
from agentbench_frame.env._antwar2.utils.actions import ActionCatalog
from agentbench_frame.env._antwar2.utils.features import FeatureExtractor
from agentbench_frame.env._antwar2.utils.constants import MAX_ACTIONS
from agentbench_frame.population.store import Population
from agentbench_frame.strategy.rl_strategy import RLStrategy, HAS_TORCH, HAS_NUMPY
from agentbench_frame.strategy.rule_strategy import RuleStrategy
from agentbench_frame.tracking.run import Run

GAME = "30_antwar2"


def _greedy_opponent(state, player: int):
    catalog = ActionCatalog(feature_extractor=FeatureExtractor())
    bundles = catalog.build(state, player) or []
    if not bundles:
        return []
    return list(max(bundles, key=lambda b: b.score).operations)


def _collect_episode(env: AntWar2Env, strategy: RLStrategy,
                     seed: int) -> Tuple[float, int, int]:
    obs = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    steps = 0
    winner = -1
    me = env._controlled_player
    while not done:
        action = strategy.act(obs.to_dict())
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            winner = info.get("winner", -1)
    return total_reward, steps, winner


def _ppo_episode(env: AntWar2Env, policy, optimizer,
                 gamma: float = 0.99,
                 clip: float = 0.2) -> Dict[str, float]:
    import torch
    import numpy as np

    feature = FeatureExtractor(max_actions=MAX_ACTIONS)
    catalog = ActionCatalog(max_actions=MAX_ACTIONS, feature_extractor=feature)
    device = next(policy.parameters()).device

    obs = env.reset(seed=int(torch.randint(0, 1 << 30, (1,)).item()))
    done = False
    me = env._controlled_player

    boards, stats_vec, masks, actions, rewards, values, log_probs = \
        [], [], [], [], [], [], []

    while not done:
        backend = env.get_backend_state()
        bundles = catalog.build(backend, me) or []
        if not bundles:
            break
        mask = catalog.action_mask(bundles)
        board = feature.encode_board(backend, me)
        st = feature.encode_stats(backend, me)
        b = torch.from_numpy(board).float().unsqueeze(0).to(device)
        s = torch.from_numpy(st).float().unsqueeze(0).to(device)
        m = torch.from_numpy(mask.astype("float32")).unsqueeze(0).to(device)

        logits, val = policy.forward(b, s, m)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        idx = dist.sample()
        log_p = dist.log_prob(idx)
        idx_clamped = min(int(idx.item()), len(bundles) - 1)
        ops = bundles[idx_clamped].operations

        boards.append((b, s, m))
        actions.append(idx)
        values.append(val.squeeze())
        log_probs.append(log_p)

        action = [[int(t) for t in op.to_protocol_tokens()] for op in ops]
        obs, reward, done, info = env.step(action)
        rewards.append(reward)

    if not rewards:
        return {"episode_reward": 0.0, "steps": 0, "loss": 0.0}

    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    if returns.numel() > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    old_log_probs = torch.stack(log_probs).detach()
    values_t = torch.stack(values).detach()
    advantages = (returns - values_t).detach()

    loss_total = torch.tensor(0.0, device=device)
    for (b, s, m), act, adv, old_lp in zip(boards, actions, advantages, old_log_probs):
        logits, val = policy.forward(b, s, m)
        dist = torch.distributions.Categorical(torch.softmax(logits, dim=-1))
        new_lp = dist.log_prob(act)
        ratio = torch.exp(new_lp - old_lp)
        surr = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
        actor_loss = -torch.min(ratio * adv, surr)
        critic_loss = (val.squeeze() - returns.mean()) ** 2
        loss_total = loss_total + actor_loss + 0.5 * critic_loss
    loss_total = loss_total / max(1, len(actions))

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    return {
        "episode_reward": sum(rewards),
        "steps": len(rewards),
        "loss": float(loss_total.item()),
    }


def run_rl(episodes: int,
           lr: float,
           ppo_epochs: int,
           seed: int,
           data_dir: Optional[str],
           smoke: bool,
           population_root: Optional[str] = None,
           agent_name: str = "rl_antwar2") -> Dict[str, Any]:
    env = AntWar2Env(opponent=_greedy_opponent, controlled_player=0)
    pop = Population(root=population_root, game=GAME)

    run = Run.start(
        game=GAME, agent=agent_name, run_type="rl", data_dir=data_dir,
        config={"episodes": episodes, "lr": lr, "ppo_epochs": ppo_epochs,
                "seed": seed, "smoke": smoke,
                "torch_available": bool(HAS_TORCH)},
    )

    strategy = RLStrategy(name=agent_name, deterministic=False)

    if HAS_TORCH and HAS_NUMPY and not smoke:
        import torch
        policy = strategy.policy
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        for ep in range(episodes):
            metrics = _ppo_episode(env, strategy, optimizer)
            run.log_episode(metrics["episode_reward"], metrics["steps"], -1)
            run.log_interactions(metrics["steps"])
            run.write("train_step", episode=ep, **metrics)
            if ep % max(1, episodes // 10) == 0:
                run.log_elo(1500.0 + metrics["episode_reward"] / 10.0)

        ckpt_dir = os.path.join(run.run_dir, "checkpoint")
        strategy.save(ckpt_dir)
        pop.register(strategy, note=f"PPO checkpoint after {episodes} episodes")
        run.log_cost(label="gpu", gpu_seconds=run.budget.stop_clock(),
                     interactions=run.budget.entries[-1].interactions if run.budget.entries else 0)
    else:
        for ep in range(episodes):
            reward, steps, winner = _collect_episode(env, strategy, seed + ep)
            run.log_episode(reward, steps, winner)
            run.log_interactions(steps)
            run.write("smoke_episode", episode=ep, reward=reward,
                      steps=steps, winner=winner)

        strategy.deterministic = True
        pop.register(strategy, note=f"smoke test ({episodes} episodes, torch={HAS_TORCH})")

    summary = run.finish()
    summary["torch_available"] = bool(HAS_TORCH)
    summary["smoke_mode"] = smoke or not HAS_TORCH
    return summary


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        prog="antwar2-rl",
        description="AntWAR2 RL training entry point",
    )
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="force greedy smoke test (skip PPO)")
    parser.add_argument("--population-root", default=None)
    parser.add_argument("--agent", default="rl_antwar2")
    args = parser.parse_args(argv)

    summary = run_rl(
        episodes=args.episodes, lr=args.lr, ppo_epochs=args.ppo_epochs,
        seed=args.seed, data_dir=args.data_dir, smoke=args.smoke,
        population_root=args.population_root, agent_name=args.agent,
    )

    print(json.dumps({
        "run_id": summary.get("run_id"),
        "torch_available": summary.get("torch_available"),
        "smoke_mode": summary.get("smoke_mode"),
        "total_episodes": summary.get("total_episodes"),
        "total_steps": summary.get("total_steps"),
        "cost_summary": summary.get("cost_summary"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
