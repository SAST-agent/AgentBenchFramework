import copy
import unittest


class _Observation:
    def __init__(self, player_id, step, done=False, winner=-1):
        self.player_id = player_id
        self.done = done
        self.state = {"step": step, "winner": winner}

    def to_dict(self):
        return {
            "player_id": self.player_id,
            "done": self.done,
            "state": dict(self.state),
        }


class _AlternatingEnv:
    def __init__(self, total_steps=3):
        self.total_steps = total_steps
        self.actions = []
        self.transitions = []

    def reset(self, seed=None):
        self.actions = []
        self.transitions = []
        return _Observation(player_id=0, step=0)

    def step(self, action):
        actor = len(self.actions) % 2
        self.actions.append(action)
        step = len(self.actions)
        done = step >= self.total_steps
        next_player = 1 - actor
        obs = _Observation(
            player_id=next_player,
            step=step,
            done=done,
            winner=0 if done else -1,
        )
        info = {"termination_reason": "rule"} if done else {}
        self.transitions.append(
            {"actor": actor, "action": action, "done": done, "step": step}
        )
        return obs, 1.0 if done and actor == 0 else 0.0, done, info

    def get_trajectory(self):
        return list(self.transitions)


class _Opponent:
    def __init__(self):
        self.name = "opponent"
        self.transitions = []

    def reset(self):
        self.transitions = []

    def act(self, _observation):
        return "opponent-action"

    def observe_transition(self, transition):
        self.transitions.append(transition)


class _ActivePolicy:
    def __init__(self):
        self.name = "new-policy"
        self.calls = []
        self.transitions = []
        self._decision_index = 0

    def reset(self):
        self.calls = []
        self.transitions = []
        self._decision_index = 0

    def decide_with_distribution(self, observation, support):
        from agentbench_frame.eval.measurement import PolicyDecision

        self.calls.append((observation, support.action_ids))
        decisions = (
            PolicyDecision("a", {"a": 0.75, "b": 0.25}),
            PolicyDecision("b", {"a": 0.50, "b": 0.50}),
        )
        decision = decisions[self._decision_index]
        self._decision_index += 1
        return decision

    def observe_transition(self, transition):
        self.transitions.append(transition)


class _MutatingActivePolicy(_ActivePolicy):
    def observe_transition(self, transition):
        transition["info"]["mutated_by_active"] = True
        transition["next_observation"]["state"]["step"] = -999
        self.transitions.append(transition)


class _ReferencePolicy:
    def __init__(self, fail=False):
        self.name = "old-policy"
        self.fail = fail
        self.calls = []
        self.transitions = []

    def reset(self):
        self.calls = []
        self.transitions = []

    def distribution_for_measurement(self, observation, support):
        self.calls.append((observation, support.action_ids))
        if self.fail:
            raise RuntimeError("reference unavailable")
        return {"a": 0.5, "b": 0.5}

    def observe_transition(self, transition):
        self.transitions.append(transition)


def _support(_observation):
    from agentbench_frame.eval.measurement import ActionCandidate, ActionSupport

    return ActionSupport(
        actions=[
            ActionCandidate("a", "environment-action-a"),
            ActionCandidate("b", "environment-action-b"),
        ],
        schema_version="fake-actions-v1",
    )


class TrajectoryKLRuntimeTests(unittest.TestCase):
    def test_decision_record_freezes_source_distributions_and_metadata(self):
        from agentbench_frame.eval.trajectory_kl import (
            TrajectoryKLDecisionRecord,
        )

        new_distribution = {"a": 0.75, "b": 0.25}
        old_distribution = {"a": 0.5, "b": 0.5}
        record = TrajectoryKLDecisionRecord(
            decision_step=1,
            context_ref="context",
            action_schema_version="actions-v1",
            support_id="support",
            legal_action_ids=("a", "b"),
            selected_action_id="a",
            new_distribution=new_distribution,
            old_distribution=old_distribution,
            new_probabilities=(0.75, 0.25),
            old_probabilities=(0.5, 0.5),
            local_policy_kl=0.1,
        )

        new_distribution["a"] = 0.0
        old_distribution["a"] = 1.0
        self.assertEqual(record.to_dict()["new_distribution"]["a"], 0.75)
        self.assertEqual(record.to_dict()["old_distribution"]["a"], 0.5)
        with self.assertRaises(TypeError):
            record.new_distribution["a"] = 0.0

    def test_episode_result_rejects_forged_summary_derived_from_records(self):
        from agentbench_frame.eval.information_gain import policy_kl
        from agentbench_frame.eval.trajectory_kl import (
            TrajectoryKLDecisionRecord,
            TrajectoryKLEpisodeResult,
        )

        local = policy_kl([0.75, 0.25], [0.5, 0.5], epsilon=0.01)
        record = TrajectoryKLDecisionRecord(
            decision_step=1,
            context_ref="context",
            action_schema_version="actions-v1",
            support_id="support",
            legal_action_ids=("a", "b"),
            selected_action_id="a",
            new_distribution={"a": 0.75, "b": 0.25},
            old_distribution={"a": 0.5, "b": 0.5},
            new_probabilities=(0.75, 0.25),
            old_probabilities=(0.5, 0.5),
            local_policy_kl=local,
        )
        with self.assertRaisesRegex(ValueError, "mean|summary|aggregate"):
            TrajectoryKLEpisodeResult(
                episode=1,
                version_before="v1",
                version_after="v2",
                epsilon=0.01,
                status="complete",
                decisions=(record,),
                trace=(local,),
                trajectory_kl_episode=local,
                mean_local_policy_kl=local + 0.25,
                errors=(),
                metadata={},
            )

    def test_episode_result_rejects_bool_aggregate_aliases(self):
        from agentbench_frame.eval.trajectory_kl import (
            TrajectoryKLDecisionRecord,
            TrajectoryKLEpisodeResult,
        )

        record = TrajectoryKLDecisionRecord(
            decision_step=1,
            context_ref="context",
            action_schema_version="actions-v1",
            support_id="support",
            legal_action_ids=("a", "b"),
            selected_action_id="a",
            new_distribution={"a": 0.5, "b": 0.5},
            old_distribution={"a": 0.5, "b": 0.5},
            new_probabilities=(0.5, 0.5),
            old_probabilities=(0.5, 0.5),
            local_policy_kl=0.0,
        )
        for field in ("trajectory_kl_episode", "mean_local_policy_kl"):
            values = {
                "trajectory_kl_episode": 0.0,
                "mean_local_policy_kl": 0.0,
            }
            values[field] = False
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "aggregate|mean"
            ):
                TrajectoryKLEpisodeResult(
                    episode=1,
                    version_before="v1",
                    version_after="v2",
                    epsilon=0.01,
                    status="complete",
                    decisions=(record,),
                    trace=(0.0,),
                    errors=(),
                    metadata={},
                    **values,
                )

    def test_payload_rejects_bool_numeric_aliases(self):
        from agentbench_frame.eval.trajectory_kl import (
            TrajectoryKLDecisionRecord,
            TrajectoryKLEpisodeResult,
            trajectory_kl_result_from_payload,
        )

        record = TrajectoryKLDecisionRecord(
            decision_step=1,
            context_ref="context",
            action_schema_version="actions-v1",
            support_id="support",
            legal_action_ids=("a", "b"),
            selected_action_id="a",
            new_distribution={"a": 0.5, "b": 0.5},
            old_distribution={"a": 0.5, "b": 0.5},
            new_probabilities=(0.5, 0.5),
            old_probabilities=(0.5, 0.5),
            local_policy_kl=0.0,
        )
        result = TrajectoryKLEpisodeResult(
            episode=1,
            version_before="v1",
            version_after="v2",
            epsilon=0.01,
            status="complete",
            decisions=(record,),
            trace=(0.0,),
            trajectory_kl_episode=0.0,
            mean_local_policy_kl=0.0,
            errors=(),
            metadata={},
            measurement_profile="24_miracle_policy_information_gain_v2",
        )
        payload = result.to_dict()

        replacements = {
            "decision_steps": True,
            "trajectory_kl_episode": False,
            "mean_local_policy_kl": False,
            "information_gain": False,
            "local_policy_kl_sum": False,
        }
        for field, forged in replacements.items():
            candidate = copy.deepcopy(payload)
            candidate[field] = forged
            with self.subTest(field=field), self.assertRaises(
                (TypeError, ValueError)
            ):
                trajectory_kl_result_from_payload(candidate)

    def test_match_measures_only_new_policy_decisions_on_the_actual_rollout(self):
        from agentbench_frame.arena.match import Match
        from agentbench_frame.eval.information_gain import policy_kl
        from agentbench_frame.eval.trajectory_kl import (
            TrajectoryKLAgent,
            TrajectoryKLConfig,
        )

        active = _ActivePolicy()
        reference = _ReferencePolicy()
        completed = []
        measured = TrajectoryKLAgent(
            active_policy=active,
            reference_policy=reference,
            support_provider=_support,
            config=TrajectoryKLConfig.for_policy_information_gain(
                version_before="v1",
                version_after="v2",
            ),
            on_episode_complete=completed.append,
        )
        env = _AlternatingEnv(total_steps=3)

        result = Match(
            env,
            measured,
            _Opponent(),
            alternate_starts=False,
            seed=17,
        ).run(n_games=1)

        self.assertEqual(result.games_played, 1)
        self.assertEqual(env.actions, [
            "environment-action-a",
            "opponent-action",
            "environment-action-b",
        ])
        self.assertEqual(len(active.calls), 2)
        self.assertEqual(len(reference.calls), 2)
        self.assertEqual(
            [call[1] for call in active.calls],
            [("a", "b"), ("a", "b")],
        )
        self.assertEqual(
            [call[1] for call in reference.calls],
            [("a", "b"), ("a", "b")],
        )
        self.assertEqual(len(active.transitions), 3)
        self.assertEqual(len(reference.transitions), 3)

        episode = completed[0]
        expected = (
            policy_kl([0.75, 0.25], [0.5, 0.5], epsilon=0.01)
            + policy_kl([0.5, 0.5], [0.5, 0.5], epsilon=0.01)
        )
        self.assertEqual(episode.status, "complete")
        self.assertEqual(episode.decision_steps, 2)
        self.assertAlmostEqual(episode.trajectory_kl_episode, expected)
        self.assertAlmostEqual(episode.mean_local_policy_kl, expected / 2)
        self.assertAlmostEqual(episode.information_gain, expected / 2)
        self.assertAlmostEqual(episode.local_policy_kl_sum, expected)
        self.assertEqual(episode.information_gain_unit, "nats / decision")
        self.assertEqual(episode.local_policy_kl_sum_unit, "nats / episode")
        self.assertEqual(episode.metadata["seed"], 17)
        self.assertEqual(episode.metadata["player_id"], 0)
        self.assertEqual(episode.metadata["opponent_name"], "opponent")
        self.assertEqual(
            episode.estimand,
            "epsilon_regularized_local_kl_sum_under_new_policy_occupancy",
        )
        self.assertEqual(
            [record.selected_action_id for record in episode.decisions],
            ["a", "b"],
        )

    def test_reference_failure_marks_episode_incomplete_without_replacing_active_action(self):
        from agentbench_frame.arena.match import Match
        from agentbench_frame.eval.trajectory_kl import (
            TrajectoryKLAgent,
            TrajectoryKLConfig,
        )

        completed = []
        measured = TrajectoryKLAgent(
            active_policy=_ActivePolicy(),
            reference_policy=_ReferencePolicy(fail=True),
            support_provider=_support,
            config=TrajectoryKLConfig.for_policy_information_gain(
                version_before="v1",
                version_after="v2",
            ),
            on_episode_complete=completed.append,
        )
        env = _AlternatingEnv(total_steps=1)

        result = Match(
            env,
            measured,
            _Opponent(),
            alternate_starts=False,
        ).run(n_games=1)

        self.assertEqual(result.games_played, 1)
        self.assertEqual(env.actions, ["environment-action-a"])
        episode = completed[0]
        self.assertEqual(episode.status, "incomplete")
        self.assertEqual(episode.decision_steps, 1)
        self.assertIsNone(episode.trajectory_kl_episode)
        self.assertIsNone(episode.mean_local_policy_kl)
        self.assertEqual(episode.trace, (None,))
        self.assertIn("reference unavailable", episode.errors[0])

    def test_new_and_reference_sessions_receive_isolated_transition_snapshots(self):
        from agentbench_frame.arena.match import Match
        from agentbench_frame.eval.trajectory_kl import (
            TrajectoryKLAgent,
            TrajectoryKLConfig,
        )

        active = _MutatingActivePolicy()
        reference = _ReferencePolicy()
        measured = TrajectoryKLAgent(
            active_policy=active,
            reference_policy=reference,
            support_provider=_support,
            config=TrajectoryKLConfig.for_policy_information_gain("v1", "v2"),
        )

        Match(
            _AlternatingEnv(total_steps=1),
            measured,
            _Opponent(),
            alternate_starts=False,
        ).run(n_games=1)

        self.assertTrue(active.transitions[0]["info"]["mutated_by_active"])
        self.assertNotIn("mutated_by_active", reference.transitions[0]["info"])
        self.assertEqual(
            reference.transitions[0]["next_observation"]["state"]["step"],
            1,
        )

    def test_match_aborts_and_emits_incomplete_measurement_on_environment_failure(self):
        from agentbench_frame.arena.match import Match
        from agentbench_frame.eval.trajectory_kl import (
            TrajectoryKLAgent,
            TrajectoryKLConfig,
        )

        class FailingEnv(_AlternatingEnv):
            def step(self, action):
                self.actions.append(action)
                raise RuntimeError("environment crashed")

        completed = []
        measured = TrajectoryKLAgent(
            active_policy=_ActivePolicy(),
            reference_policy=_ReferencePolicy(),
            support_provider=_support,
            config=TrajectoryKLConfig.for_policy_information_gain("v1", "v2"),
            on_episode_complete=completed.append,
        )

        with self.assertRaisesRegex(RuntimeError, "environment crashed"):
            Match(
                FailingEnv(),
                measured,
                _Opponent(),
                alternate_starts=False,
            ).run(n_games=1)

        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].status, "incomplete")
        self.assertIsNone(completed[0].trajectory_kl_episode)
        self.assertIn("environment crashed", completed[0].errors[-1])

    def test_measurement_persistence_callback_failure_is_not_silenced(self):
        from agentbench_frame.arena.match import Match
        from agentbench_frame.eval.trajectory_kl import (
            TrajectoryKLAgent,
            TrajectoryKLConfig,
        )

        def fail_persistence(_result):
            raise OSError("cannot persist trajectory KL")

        measured = TrajectoryKLAgent(
            active_policy=_ActivePolicy(),
            reference_policy=_ReferencePolicy(),
            support_provider=_support,
            config=TrajectoryKLConfig.for_policy_information_gain("v1", "v2"),
            on_episode_complete=fail_persistence,
        )

        with self.assertRaisesRegex(OSError, "cannot persist trajectory KL"):
            Match(
                _AlternatingEnv(total_steps=1),
                measured,
                _Opponent(),
                alternate_starts=False,
            ).run(n_games=1)

    def test_reset_aborts_unfinished_episode_before_clearing_decisions(self):
        from agentbench_frame.eval.trajectory_kl import (
            TrajectoryKLAgent,
            TrajectoryKLConfig,
        )

        completed = []
        measured = TrajectoryKLAgent(
            active_policy=_ActivePolicy(),
            reference_policy=_ReferencePolicy(),
            support_provider=_support,
            config=TrajectoryKLConfig.for_policy_information_gain("v1", "v2"),
            on_episode_complete=completed.append,
        )
        measured.reset()
        measured.act(_Observation(player_id=0, step=0).to_dict())
        measured.reset()

        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].status, "incomplete")
        self.assertIn("reset before terminal transition", completed[0].errors[-1])

    def test_generic_config_accepts_research_epsilon_but_rejects_invalid_values(self):
        from agentbench_frame.eval.trajectory_kl import TrajectoryKLConfig

        for epsilon in (0.0, 0.001, 0.05, 1.0):
            config = TrajectoryKLConfig("v1", "v2", epsilon)
            self.assertFalse(config.is_formal_policy_information_gain)
        for epsilon in (-0.1, 1.1, True):
            with self.subTest(epsilon=epsilon):
                with self.assertRaises((TypeError, ValueError)):
                    TrajectoryKLConfig("v1", "v2", epsilon)
        with self.assertRaises(ValueError):
            TrajectoryKLConfig("", "v2", 0.01)


if __name__ == "__main__":
    unittest.main()
