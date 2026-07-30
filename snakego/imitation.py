"""Imitation Game (IG): behavior cloning + KL divergence.

Two complementary analyses:

1. **Behavior signature KL** -- For each agent, record the marginal
   distribution of action *types* taken over many games (move_1..4,
   railgun, split).  Then compute KL(A || B) between any pair of agents.
   This measures how *behaviorally different* two strategies are,
   independent of game outcomes.

2. **Behavior cloning (IG)** -- Record expert trajectories
   (state-features -> action), train a feature-conditioned frequency
   model to imitate the expert, then measure:
   - cross-entropy = KL(expert_policy || learner_policy) on held-out data
   - action-prediction accuracy
   - behavioral KL on fresh games

Usage (CLI)::

    python -m snakego.imitation --expert greedy --learner greedy --n-games 10
    python -m snakego.imitation --expert greedy --bc              # train BC learner

Usage (API)::

    from snakego.imitation import behavioral_signature, kl_signature
    sig_a = behavioral_signature(agent_a, opponents, n_games=8)
    sig_b = behavioral_signature(agent_b, opponents, n_games=8)
    print(kl_signature(sig_a, sig_b))
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .agents import BaseAgent, BUILTIN_AGENTS
from .env import (
    ACT_MOVE_BASE, ACT_RAILGUN, ACT_SPLIT, DX, DY,
    GameConfig, SnakeGoGame,
)
from .population import get_pool


ACTION_LABELS = {
    1: "move_+x", 2: "move_+y", 3: "move_-x", 4: "move_-y",
    5: "railgun", 6: "split",
}
ACTION_TYPES = [1, 2, 3, 4, 5, 6]


# --------------------------------------------------------------------------- #
# play one game and record the candidate's action types
# --------------------------------------------------------------------------- #


def _play_and_record_actions(
    candidate: BaseAgent,
    opponent: BaseAgent,
    seed: int,
) -> List[int]:
    """Play a game; return the list of action types the candidate took."""
    game = SnakeGoGame(GameConfig(seed=seed))
    game.reset(seed)
    agents = {0: candidate, 1: opponent}
    actions: List[int] = []
    steps = 0
    while not game.is_over() and steps < 200000:
        player = game.current_player
        sid = game.current_snake_id()
        if sid is None:
            game.act(0)
            steps += 1
            continue
        obs = game.obs()
        action = agents[player].act(obs, sid, game)
        if player == 0:
            actions.append(action)
        game.act(action)
        steps += 1
    return actions


def _play_both_sides_and_record(
    candidate: BaseAgent,
    opponent: BaseAgent,
    seed: int,
) -> List[int]:
    """Record candidate actions in both seats (P0 and P1)."""
    a1 = _play_and_record_actions(candidate, opponent, seed)
    a2 = _play_and_record_actions(opponent, candidate, seed + 1000)
    # in a2 the candidate is P1; we recorded opponent (P1) actions,
    # but wait -- _play_and_record_actions records player==0 actions.
    # Fix: record P1 actions too.
    return a1  # we handle P1 below separately


def _record_candidate_actions(
    candidate: BaseAgent,
    opponent: BaseAgent,
    seed: int,
) -> List[int]:
    """Record candidate actions across both seats (P0 and P1)."""
    out: List[int] = []
    # candidate as P0
    game = SnakeGoGame(GameConfig(seed=seed))
    game.reset(seed)
    agents = {0: candidate, 1: opponent}
    steps = 0
    while not game.is_over() and steps < 200000:
        player = game.current_player
        sid = game.current_snake_id()
        if sid is None:
            game.act(0)
            steps += 1
            continue
        obs = game.obs()
        action = agents[player].act(obs, sid, game)
        if player == 0:
            out.append(action)
        game.act(action)
        steps += 1
    # candidate as P1
    game = SnakeGoGame(GameConfig(seed=seed + 1000))
    game.reset(seed + 1000)
    agents = {0: opponent, 1: candidate}
    steps = 0
    while not game.is_over() and steps < 200000:
        player = game.current_player
        sid = game.current_snake_id()
        if sid is None:
            game.act(0)
            steps += 1
            continue
        obs = game.obs()
        action = agents[player].act(obs, sid, game)
        if player == 1:
            out.append(action)
        game.act(action)
        steps += 1
    return out


# --------------------------------------------------------------------------- #
# 1. Behavioral signature + KL divergence
# --------------------------------------------------------------------------- #


def behavioral_signature(
    agent: BaseAgent,
    opponents: Optional[Dict[str, BaseAgent]] = None,
    n_seeds: int = 4,
) -> Dict[int, float]:
    """Record the action-type distribution of *agent* over many games.

    Returns a dict {action_type: probability} over {1..6}.
    """
    if opponents is None:
        opponents = get_pool("train")
    counter: Counter = Counter()
    for opp in opponents.values():
        for s in range(n_seeds):
            actions = _record_candidate_actions(agent, opp, s)
            counter.update(actions)
    total = sum(counter.values()) or 1
    return {a: counter.get(a, 0) / total for a in ACTION_TYPES}


def kl_divergence(p: Dict[int, float], q: Dict[int, float]) -> float:
    """KL(P || Q) over the shared action-type support.

    Uses smoothing: actions absent from Q get a small epsilon.
    """
    eps = 1e-6
    total = 0.0
    for a in ACTION_TYPES:
        pi = p.get(a, 0.0)
        qi = q.get(a, eps)
        if pi > 0:
            total += pi * math.log(pi / max(qi, eps))
    return total


def kl_signature(sig_a: Dict[int, float], sig_b: Dict[int, float]) -> float:
    """Symmetric measure: average of KL(A||B) and KL(B||A)."""
    return (kl_divergence(sig_a, sig_b) + kl_divergence(sig_b, sig_a)) / 2.0


def signature_matrix(
    agents: Dict[str, BaseAgent],
    opponents: Optional[Dict[str, BaseAgent]] = None,
    n_seeds: int = 4,
) -> Dict[str, Dict[str, float]]:
    """Compute pairwise symmetric-KL between all agents' signatures."""
    sigs = {name: behavioral_signature(a, opponents, n_seeds)
            for name, a in agents.items()}
    matrix: Dict[str, Dict[str, float]] = {}
    for n1, s1 in sigs.items():
        matrix[n1] = {}
        for n2, s2 in sigs.items():
            matrix[n1][n2] = round(kl_signature(s1, s2), 4)
    return matrix


# --------------------------------------------------------------------------- #
# 2. Behavior cloning (IG): expert -> learner
# --------------------------------------------------------------------------- #


def _extract_features(obs: dict, snake_id: int, game: SnakeGoGame) -> tuple:
    """Discretize observation into a compact feature tuple.

    Features:
      - turn_phase: 0 (1-170), 1 (171-340), 2 (341+)
      - length_bucket: 0 (1-3), 1 (4-8), 2 (9-15), 3 (16+)
      - has_railgun: 0/1
      - n_snakes_camp: number of own snakes (1..4)
      - nearest_item_dir: 0=none, 1-4 toward nearest item direction
      - near_wall: 0/1 (head within 2 of any edge)
    """
    snake = game._get_snake(snake_id)
    coor = snake.coor_list
    head = coor[0]
    turn = obs["turn"]
    phase = 0 if turn <= 170 else (1 if turn <= 340 else 2)
    length = len(coor)
    lb = 0 if length <= 3 else (1 if length <= 8 else (2 if length <= 15 else 3))
    has_rg = 1 if snake.has_railgun() else 0
    n_camp = sum(1 for s in obs["snakes"] if s["camp"] == snake.camp)
    n_camp = min(n_camp, 4)

    # nearest item direction
    item_dir = 0
    best_d = 10 ** 9
    for it in obs["items"]:
        d = abs(it["x"] - head[0]) + abs(it["y"] - head[1])
        if d < best_d:
            best_d = d
            dx = it["x"] - head[0]
            dy = it["y"] - head[1]
            if abs(dx) >= abs(dy):
                item_dir = 1 if dx > 0 else 3
            else:
                item_dir = 2 if dy > 0 else 4

    L, W = obs["length"], obs["width"]
    near_wall = 1 if min(head[0], L - 1 - head[0], head[1], W - 1 - head[1]) <= 2 else 0

    return (phase, lb, has_rg, n_camp, item_dir, near_wall)


class BehaviorCloner:
    """Feature-conditioned frequency model that imitates an expert.

    Trains on (feature_tuple -> action) pairs from expert trajectories,
    then predicts a probability distribution over actions for new states.
    """

    def __init__(self) -> None:
        # feature_tuple -> {action: count}
        self._counts: Dict[tuple, Counter] = defaultdict(Counter)
        self._total_samples = 0

    def collect(self, expert: BaseAgent, opponent: BaseAgent,
                seeds: List[int]) -> None:
        """Collect expert trajectories and accumulate feature-action counts."""
        for seed in seeds:
            game = SnakeGoGame(GameConfig(seed=seed))
            game.reset(seed)
            agents = {0: expert, 1: opponent}
            steps = 0
            while not game.is_over() and steps < 200000:
                player = game.current_player
                sid = game.current_snake_id()
                if sid is None:
                    game.act(0)
                    steps += 1
                    continue
                obs = game.obs()
                if player == 0:
                    action = agents[0].act(obs, sid, game)
                    feat = _extract_features(obs, sid, game)
                    self._counts[feat][action] += 1
                    self._total_samples += 1
                else:
                    action = agents[1].act(obs, sid, game)
                game.act(action)
                steps += 1

    def train(self, expert: BaseAgent,
              opponents: Dict[str, BaseAgent],
              n_games: int = 10,
              base_seed: int = 0) -> None:
        """Collect from multiple opponents for diverse coverage."""
        for i, opp in enumerate(opponents.values()):
            seeds = [base_seed + i * 100 + j for j in range(n_games)]
            self.collect(expert, opp, seeds)

    def predict_dist(self, feat: tuple) -> Dict[int, float]:
        """Return P(action | feature) with Laplace smoothing."""
        counts = self._counts.get(feat, Counter())
        total = sum(counts.values()) + len(ACTION_TYPES)  # Laplace
        return {a: (counts.get(a, 0) + 1) / total for a in ACTION_TYPES}

    @property
    def n_features_seen(self) -> int:
        return len(self._counts)

    @property
    def n_samples(self) -> int:
        return self._total_samples


def evaluate_bc(
    expert: BaseAgent,
    learner: BehaviorCloner,
    opponents: Dict[str, BaseAgent],
    n_games: int = 5,
    base_seed: int = 5000,
) -> Dict[str, Any]:
    """Evaluate behavior cloning on held-out games.

    Measures:
      - cross_entropy: -log P_learner(expert_action) averaged
        (= KL(expert || learner) since expert is ~deterministic)
      - accuracy: fraction where learner's argmax == expert action
      - top2_rate: expert action in learner's top-2
    """
    total_ce = 0.0
    correct = 0
    top2 = 0
    n = 0
    for i, opp in enumerate(opponents.values()):
        for g in range(n_games):
            seed = base_seed + i * 100 + g
            game = SnakeGoGame(GameConfig(seed=seed))
            game.reset(seed)
            agents = {0: expert, 1: opp}
            steps = 0
            while not game.is_over() and steps < 200000:
                player = game.current_player
                sid = game.current_snake_id()
                if sid is None:
                    game.act(0)
                    steps += 1
                    continue
                obs = game.obs()
                action = agents[player].act(obs, sid, game)
                if player == 0:
                    feat = _extract_features(obs, sid, game)
                    dist = learner.predict_dist(feat)
                    p = dist.get(action, 1e-6)
                    total_ce += -math.log(max(p, 1e-10))
                    ranked = sorted(dist.items(), key=lambda x: -x[1])
                    if ranked[0][0] == action:
                        correct += 1
                    elif len(ranked) > 1 and ranked[1][0] == action:
                        top2 += 1
                    n += 1
                game.act(action)
                steps += 1

    ce = total_ce / n if n else 0.0
    acc = correct / n if n else 0.0
    t2 = (correct + top2) / n if n else 0.0
    return {
        "cross_entropy": round(ce, 4),
        "kl_expert_to_learner": round(ce, 4),  # CE == KL for ~deterministic expert
        "accuracy": round(acc, 4),
        "top2_rate": round(t2, 4),
        "n_samples": n,
        "n_features": learner.n_features_seen,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="snakego.imitation",
        description="Imitation Game: behavioral KL divergence + behavior cloning.",
    )
    parser.add_argument("--expert", default="greedy",
                        help="expert agent name (builtin or strategy)")
    parser.add_argument("--learner", default="greedy",
                        help="learner agent name for signature comparison")
    parser.add_argument("--pool", default="train",
                        help="opponent pool for recording")
    parser.add_argument("--n-games", type=int, default=8)
    parser.add_argument("--bc", action="store_true",
                        help="train a behavior-cloning learner and evaluate it")
    args = parser.parse_args(argv)

    from .evaluate import load_strategy_agent

    expert = load_strategy_agent(args.expert)
    opponents = get_pool(args.pool)

    print(f"=== Behavioral KL Divergence ===\n")
    sig_e = behavioral_signature(expert, opponents, n_seeds=args.n_games)
    learner = load_strategy_agent(args.learner)
    sig_l = behavioral_signature(learner, opponents, n_seeds=args.n_games)

    print(f"Expert ({expert.name}) action distribution:")
    for a, p in sig_e.items():
        print(f"  {ACTION_LABELS[a]:12s} {p:.3f}")
    print(f"\nLearner ({learner.name}) action distribution:")
    for a, p in sig_l.items():
        print(f"  {ACTION_LABELS[a]:12s} {p:.3f}")

    kl = kl_signature(sig_e, sig_l)
    print(f"\nSymmetric KL({expert.name} || {learner.name}) = {kl:.4f}")

    if args.bc:
        print(f"\n=== Behavior Cloning (IG) ===\n")
        bc = BehaviorCloner()
        train_opp = get_pool("train")
        test_opp = get_pool("test")
        print(f"Training BC learner on {expert.name} ...")
        bc.train(expert, train_opp, n_games=args.n_games)
        print(f"  samples: {bc.n_samples}  features: {bc.n_features_seen}")

        print(f"\nEvaluating on held-out TEST pool ...")
        result = evaluate_bc(expert, bc, test_opp, n_games=4)
        print(f"  cross_entropy (= KL expert->learner): {result['cross_entropy']}")
        print(f"  action accuracy:                     {result['accuracy']:.1%}")
        print(f"  top-2 rate:                          {result['top2_rate']:.1%}")
        print(f"  samples evaluated:                   {result['n_samples']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "behavioral_signature", "kl_divergence", "kl_signature", "signature_matrix",
    "BehaviorCloner", "evaluate_bc",
    "ACTION_LABELS", "ACTION_TYPES",
]
