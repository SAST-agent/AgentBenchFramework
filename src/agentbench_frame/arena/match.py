"""
Match runner for 1v1 agent competitions.

Handles running multiple games between two agents, collecting results,
and computing win rates and statistics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

from agentbench_frame.env.base import BaseEnv
from agentbench_frame.agent.base import BaseAgent


@dataclass
class MatchResult:
    """Results from a match between two agents."""
    agent1_name: str
    agent2_name: str
    games_played: int
    agent1_wins: int
    agent2_wins: int
    draws: int
    win_rate: float
    avg_game_length: float
    avg_agent1_reward: float
    avg_agent2_reward: float
    game_results: List[Dict[str, Any]] = field(default_factory=list)
    elo_change: float = 0.0
    duration_seconds: float = 0.0
    benchmark_score: float = 0.0


class Match:
    """
    Runs a series of games between two agents.

    Supports:
    - Alternating who goes first (公平先后手)
    - Multiple games for statistical significance
    - Detailed per-game statistics
    - Configurable time limits

    Example:
        env = GeneralsEnv()
        agent1 = RuleBasedAgent("Aggressive")
        agent2 = RandomAgent("Random")

        match = Match(env, agent1, agent2)
        result = match.run(n_games=100)
        print(f"Win rate: {result.win_rate:.2%}")
    """

    def __init__(self,
                 env: BaseEnv,
                 agent1: BaseAgent,
                 agent2: BaseAgent,
                 alternate_starts: bool = True,
                 seed: int = 42):
        self.env = env
        self.agent1 = agent1
        self.agent2 = agent2
        self.alternate_starts = alternate_starts
        self.seed = seed

    def run(self, n_games: int = 10) -> MatchResult:
        """
        Run n_games between the two agents.

        Args:
            n_games: Number of games to play

        Returns:
            MatchResult with statistics
        """
        start_time = time.time()
        game_results = []
        agent1_wins = 0
        agent2_wins = 0
        draws = 0
        total_length = 0
        total_reward1 = 0
        total_reward2 = 0

        for game_idx in range(n_games):
            result = self._play_game(game_idx)
            game_results.append(result)

            # `winner` is the environment's player id.  That id changes
            # ownership when starts alternate, so aggregate by the mapped
            # agent identity instead.
            winner_agent = result.get("winner_agent")
            if winner_agent == self.agent1.name:
                agent1_wins += 1
            elif winner_agent == self.agent2.name:
                agent2_wins += 1
            else:
                draws += 1

            total_length += result.get("steps", 0)
            total_reward1 += result.get("agent1_reward", result.get("reward_0", 0))
            total_reward2 += result.get("agent2_reward", result.get("reward_1", 0))

        duration = time.time() - start_time
        n = max(1, n_games)

        return MatchResult(
            agent1_name=self.agent1.name,
            agent2_name=self.agent2.name,
            games_played=n_games,
            agent1_wins=agent1_wins,
            agent2_wins=agent2_wins,
            draws=draws,
            win_rate=(agent1_wins + 0.5 * draws) / n,
            avg_game_length=total_length / n,
            avg_agent1_reward=total_reward1 / n,
            avg_agent2_reward=total_reward2 / n,
            game_results=game_results,
            duration_seconds=duration,
            benchmark_score=(agent1_wins + 0.5 * draws) / n,
        )

    def _play_game(self, game_idx: int) -> Dict[str, Any]:
        """Play a single game between the two agents."""
        seed = self.seed + game_idx

        # Track which side the agents are on
        # Alternate who goes first
        if self.alternate_starts and game_idx % 2 == 1:
            player_agents = {0: self.agent2, 1: self.agent1}
        else:
            player_agents = {0: self.agent1, 1: self.agent2}

        for player_id, agent in player_agents.items():
            set_metadata = getattr(
                agent, "set_measurement_episode_metadata", None
            )
            if callable(set_metadata):
                opponent = player_agents[1 - player_id]
                set_metadata({
                    "game_index": game_idx,
                    "seed": seed,
                    "player_id": player_id,
                    "opponent_name": getattr(opponent, "name", "unknown"),
                })

        self.agent1.reset()
        self.agent2.reset()
        obs = self.env.reset(seed=seed)
        done = False
        current_player = getattr(obs, "player_id", 0)
        total_rewards = {0: 0.0, 1: 0.0}
        step_count = 0

        while not done:
            previous_obs = obs
            actor_player = current_player
            agent = player_agents[current_player]
            action = agent.act(obs.to_dict())
            obs, reward, done, info = self.env.step(action)
            step_count += 1

            total_rewards[actor_player] += reward
            transition = {
                "observation": previous_obs.to_dict(),
                "actor_player_id": actor_player,
                "action": action,
                "next_observation": obs.to_dict(),
                "reward": reward,
                "terminated": bool(done),
                "truncated": False,
                "done": bool(done),
                "info": info or {},
                "env_step": step_count,
            }
            notified = set()
            for participant in player_agents.values():
                identity = id(participant)
                if identity in notified:
                    continue
                notified.add(identity)
                observe = getattr(participant, "observe_transition", None)
                if callable(observe):
                    try:
                        observe(transition)
                    except Exception:
                        pass
            # Relay observation to the other agent's perspective
            current_player = obs.player_id

        # Map winner back to agent identity
        raw_winner = obs.state.get("winner", -1)
        if raw_winner == 0:
            winner_agent = player_agents[0].name
        elif raw_winner == 1:
            winner_agent = player_agents[1].name
        else:
            winner_agent = "draw"

        if player_agents[0] is self.agent1:
            agent1_reward = total_rewards[0]
            agent2_reward = total_rewards[1]
        else:
            agent1_reward = total_rewards[1]
            agent2_reward = total_rewards[0]

        measurements = []
        seen = set()
        for participant in player_agents.values():
            identity = id(participant)
            if identity in seen:
                continue
            seen.add(identity)
            measurement = getattr(
                participant, "latest_trajectory_kl_result", None
            )
            if measurement is not None:
                measurements.append(
                    measurement.to_dict()
                    if hasattr(measurement, "to_dict")
                    else measurement
                )

        return {
            "game_idx": game_idx,
            "winner": raw_winner,
            "winner_agent": winner_agent,
            "steps": step_count,
            "env_steps": step_count,
            "reward_0": total_rewards[0],
            "reward_1": total_rewards[1],
            "agent1_reward": agent1_reward,
            "agent2_reward": agent2_reward,
            "trajectory": self.env.get_trajectory(),
            "trajectory_kl_measurements": measurements,
        }

    def run_parallel(self, n_games: int = 10, n_workers: int = 4) -> MatchResult:
        """
        Run games in parallel using multiple processes.
        (For environments that support pickling)
        """
        # Simplified: run sequentially but could be parallelized
        return self.run(n_games)
