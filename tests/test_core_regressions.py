import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock


class _Observation:
    def __init__(self, player_id=0, done=False, winner=-1):
        self.player_id = player_id
        self.round_num = 0 if not done else 1
        self.done = done
        self.state = {"winner": winner}

    def to_dict(self):
        return {"state": self.state, "player_id": self.player_id}


class _OneStepEnv:
    def reset(self, seed=None):
        return _Observation()

    def step(self, action):
        return _Observation(player_id=1, done=True, winner=1), 2.0, True, {}

    def get_trajectory(self):
        return []


class _PPOCompatibleEnv:
    def reset(self):
        return _Observation()

    def step(self, action):
        return _Observation(player_id=1, done=True, winner=1), 0.0, True, {}

    def to_feature_vector(self, observation, player):
        return [0.0]


class _TwoStepEnv:
    def __init__(self):
        self.steps = 0

    def reset(self):
        self.steps = 0
        return _Observation()

    def step(self, action):
        self.steps += 1
        done = self.steps >= 2
        return _Observation(player_id=1, done=done, winner=1 if done else -1), 0.0, done, {}


class _DrawEnv:
    def reset(self, seed=None):
        return _Observation()

    def step(self, action):
        return _Observation(winner=-1), 0.0, True, {}

    def get_trajectory(self):
        return []


class CoreIntegrationTests(TestCase):
    def test_generals_is_registered_by_default(self):
        from agentbench_frame.env import ENV_REGISTRY, make_env

        self.assertIn("generals", ENV_REGISTRY)
        self.assertEqual(make_env("generals").game_name, "Generals")

    def test_default_mcp_server_is_public_api(self):
        from agentbench_frame.mcp import MCPServer, create_default_server

        self.assertIsInstance(create_default_server(), MCPServer)

    def test_match_counts_agent_identity_when_starts_alternate(self):
        from agentbench_frame.agent import RandomAgent
        from agentbench_frame.arena import Match

        result = Match(
            _OneStepEnv(),
            RandomAgent("agent-1"),
            RandomAgent("agent-2"),
            alternate_starts=True,
        ).run(n_games=2)

        self.assertEqual(result.agent1_wins, 1)
        self.assertEqual(result.agent2_wins, 1)
        self.assertEqual(result.win_rate, 0.5)
        self.assertEqual(result.avg_agent1_reward, 1.0)
        self.assertEqual(result.avg_agent2_reward, 1.0)
        self.assertEqual(
            [game["winner_agent"] for game in result.game_results],
            ["agent-2", "agent-1"],
        )

    def test_match_treats_draw_as_half_win_for_score(self):
        from agentbench_frame.agent import RandomAgent
        from agentbench_frame.arena import Match

        result = Match(
            _DrawEnv(), RandomAgent("agent-1"), RandomAgent("agent-2")
        ).run(n_games=1)

        self.assertEqual(result.draws, 1)
        self.assertAlmostEqual(result.win_rate, 0.5)

    def test_rl_runner_delegates_to_ppo(self):
        from agentbench_frame.env import GeneralsEnv
        from agentbench_frame.runner.rl_runner import BaseRLRunner

        class FakePPOTrainer:
            instances = []

            def __init__(self, env, config, eval_opponent=None):
                self.env = env
                self.config = config
                self.policy = object()
                self.__class__.instances.append(self)

            def train(self):
                return {"total_timesteps": self.config.total_timesteps}

            def get_episode_results(self):
                return [{"reward": 2.0, "steps": 3, "winner": 0}]

        logged_episodes = []
        run = SimpleNamespace(
            _tracked_env=GeneralsEnv(),
            _timed_agent=None,
            _total_steps=0,
            _episodes=0,
            run_dir="/tmp/agentbench-test-run",
            log_episode=lambda **kwargs: logged_episodes.append(kwargs),
            write=lambda *args, **kwargs: None,
        )
        runner = BaseRLRunner(
            game="generals",
            agent="ppo-test",
            total_timesteps=123,
            config={"learning_rate": 0.01, "max_episodes": 1},
        )

        with mock.patch("agentbench_frame.runner.rl_runner.HAS_TORCH", True), \
                mock.patch("agentbench_frame.training.ppo_trainer.HAS_NUMPY", True), \
                mock.patch("agentbench_frame.training.ppo_trainer.PPOTrainer", FakePPOTrainer):
            result = runner._execute(run)

        self.assertEqual(len(FakePPOTrainer.instances), 1)
        self.assertEqual(FakePPOTrainer.instances[0].config.total_timesteps, 123)
        self.assertEqual(result["total_timesteps"], 123)
        self.assertEqual(logged_episodes, [{"reward": 2.0, "steps": 3, "winner": 0}])

    def test_rl_runner_uses_fallback_for_non_generals_env(self):
        from agentbench_frame.runner.rl_runner import BaseRLRunner

        class UnexpectedPPOTrainer:
            instances = []

            def __init__(self, *args, **kwargs):
                self.__class__.instances.append(self)

            def train(self):
                return {}

        agent = SimpleNamespace(
            act=lambda observation: [[8]],
            learn=lambda batch: {},
        )
        run = SimpleNamespace(
            _tracked_env=_PPOCompatibleEnv(),
            _timed_agent=agent,
            _total_steps=0,
            _episodes=0,
            run_dir="/tmp/agentbench-test-run",
            log_episode=lambda **kwargs: None,
            write=lambda *args, **kwargs: None,
        )
        runner = BaseRLRunner(
            game="custom",
            agent="custom-agent",
            total_timesteps=1,
            config={"max_episodes": 1},
        )

        with mock.patch("agentbench_frame.runner.rl_runner.HAS_TORCH", True), \
                mock.patch("agentbench_frame.training.ppo_trainer.HAS_NUMPY", True), \
                mock.patch("agentbench_frame.training.ppo_trainer.PPOTrainer", UnexpectedPPOTrainer):
            runner._execute(run)

        self.assertEqual(UnexpectedPPOTrainer.instances, [])

    def test_rl_runner_fallback_respects_timestep_budget(self):
        from agentbench_frame.runner.rl_runner import BaseRLRunner

        logged_episodes = []
        run = SimpleNamespace(
            _tracked_env=_TwoStepEnv(),
            _timed_agent=SimpleNamespace(
                act=lambda observation: [[8]],
                learn=lambda batch: {},
            ),
            _total_steps=0,
            _episodes=0,
            log_episode=lambda **kwargs: (
                logged_episodes.append(kwargs),
                setattr(run, "_total_steps", run._total_steps + kwargs["steps"]),
            ),
            write=lambda *args, **kwargs: None,
        )
        runner = BaseRLRunner(
            game="generic",
            agent="fallback-test",
            total_timesteps=1,
            config={"max_episodes": 2},
        )

        with mock.patch("agentbench_frame.runner.rl_runner.HAS_TORCH", False):
            runner._execute(run)

        self.assertEqual(run._total_steps, 1)
        self.assertEqual(logged_episodes[0]["steps"], 1)

    def test_ppo_trainer_exposes_completed_episode_results(self):
        from agentbench_frame.training.ppo_trainer import PPOTrainer

        trainer = object.__new__(PPOTrainer)
        trainer._episode_results = [{"reward": 1.0, "steps": 2, "winner": 0}]

        self.assertEqual(trainer.get_episode_results(), trainer._episode_results)

    def test_rl_runner_fallback_calls_agent_learn(self):
        from agentbench_frame.runner.rl_runner import BaseRLRunner

        learned_batches = []
        agent = SimpleNamespace(
            act=lambda observation: [[8]],
            learn=lambda batch: learned_batches.append(batch) or {"loss": 0.0},
        )
        run = SimpleNamespace(
            _tracked_env=_PPOCompatibleEnv(),
            _timed_agent=agent,
            _total_steps=0,
            _episodes=0,
            log_episode=lambda **kwargs: None,
            write=lambda *args, **kwargs: None,
        )
        runner = BaseRLRunner(
            game="generic",
            agent="fallback-test",
            total_timesteps=1,
            config={"max_episodes": 1},
        )

        with mock.patch("agentbench_frame.runner.rl_runner.HAS_TORCH", False):
            runner._execute(run)

        self.assertEqual(len(learned_batches), 1)
        self.assertEqual(len(learned_batches[0]["episode"]), 1)

    def test_runner_persists_execute_result(self):
        from agentbench_frame.runner.base import BaseRunner

        class ResultRunner(BaseRunner):
            def _execute(self, run):
                return {"custom_metric": 7}

        with tempfile.TemporaryDirectory() as data_dir:
            result = ResultRunner(
                game="test-game",
                agent="test-agent",
                data_dir=data_dir,
            ).run()
            run_path = (
                Path(data_dir)
                / "runs"
                / "test-game"
                / "test-agent"
                / result["run_id"]
            )
            persisted = json.loads((run_path / "summary.json").read_text())

        self.assertEqual(result["custom_metric"], 7)
        self.assertEqual(persisted.get("custom_metric"), 7)


if __name__ == "__main__":
    import unittest

    unittest.main()
