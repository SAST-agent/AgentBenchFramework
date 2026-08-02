import unittest


class HLConfigTests(unittest.TestCase):
    def test_native_rollout_budget_is_strict_and_serializable(self):
        from agentbench_frame.hl.config import HLRunConfig

        config = HLRunConfig.from_mapping(
            {
                "game": "29_rollman",
                "provider": {
                    "kind": "codex",
                    "expected_cli_version": "codex-cli 0.146.0-alpha.9.2",
                    "rollout_budget": {
                        "enabled": True,
                        "limit_tokens": 70000,
                        "reminder_at_remaining_tokens": [20000, 10000, 5000],
                        "sampling_token_weight": 1.0,
                        "prefill_token_weight": 1.0,
                    },
                },
            }
        )

        budget = config.provider.rollout_budget
        self.assertTrue(budget.enabled)
        self.assertEqual(budget.limit_tokens, 70000)
        self.assertEqual(
            budget.reminder_at_remaining_tokens,
            (20000, 10000, 5000),
        )
        self.assertEqual(
            config.to_dict()["provider"]["rollout_budget"],
            {
                "enabled": True,
                "limit_tokens": 70000,
                "reminder_at_remaining_tokens": (20000, 10000, 5000),
                "sampling_token_weight": 1.0,
                "prefill_token_weight": 1.0,
            },
        )

    def test_native_rollout_budget_rejects_ambiguous_limits_and_weights(self):
        from agentbench_frame.hl.config import HLRunConfig

        invalid_budgets = (
            {"enabled": True, "limit_tokens": None},
            {"enabled": True, "limit_tokens": 0},
            {
                "enabled": True,
                "limit_tokens": 100,
                "reminder_at_remaining_tokens": [100],
            },
            {
                "enabled": True,
                "limit_tokens": 100,
                "reminder_at_remaining_tokens": [10, 20],
            },
            {
                "enabled": True,
                "limit_tokens": 100,
                "sampling_token_weight": 0,
            },
            {
                "enabled": True,
                "limit_tokens": 100,
                "prefill_token_weight": float("inf"),
            },
        )
        for rollout_budget in invalid_budgets:
            with self.subTest(rollout_budget=rollout_budget), self.assertRaises(
                ValueError
            ):
                HLRunConfig.from_mapping(
                    {
                        "game": "29_rollman",
                        "provider": {
                            "kind": "codex",
                            "rollout_budget": rollout_budget,
                        },
                    }
                )

    def test_defaults_keep_open_ended_iteration_and_enable_rollback(self):
        from agentbench_frame.hl.config import HLRunConfig

        config = HLRunConfig.from_mapping(
            {
                "game": "29_rollman",
                "provider": {
                    "kind": "codex",
                    "model": "gpt-5.5",
                    "env_key": "AGENTBENCH_API_KEY",
                },
            }
        )

        self.assertIsNone(config.iteration.max_acts)
        self.assertEqual(config.iteration.candidates_per_act, 1)
        self.assertTrue(config.rollback.enabled)
        self.assertEqual(config.rollback.patience, 3)
        self.assertAlmostEqual(config.rollback.score_margin, 0.05)
        self.assertEqual(config.provider.context_mode, "resumable")

    def test_invalid_ablation_values_fail_closed(self):
        from agentbench_frame.hl.config import HLRunConfig

        base = {
            "game": "29_rollman",
            "provider": {"kind": "codex", "env_key": "AGENTBENCH_API_KEY"},
        }
        for override in (
            {"iteration": {"candidates_per_act": 0}},
            {"iteration": {"max_acts": 0}},
            {"provider": {"kind": "codex", "env_key": "KEY", "context_mode": "hidden"}},
            {"provider": {"kind": "codex", "idle_timeout_seconds": 0}},
            {
                "provider": {
                    "kind": "codex",
                    "timeout_seconds": 10,
                    "idle_timeout_seconds": 11,
                }
            },
            {"rollback": {"patience": 0}},
            {"rollback": {"score_margin": -0.01}},
            {"measurement": {"epsilon": 1.01}},
        ):
            mapping = {**base, **override}
            with self.subTest(override=override), self.assertRaises(ValueError):
                HLRunConfig.from_mapping(mapping)

    def test_unknown_fields_and_embedded_secrets_are_rejected(self):
        from agentbench_frame.hl.config import HLRunConfig

        with self.assertRaises(ValueError):
            HLRunConfig.from_mapping(
                {
                    "game": "29_rollman",
                    "provider": {
                        "kind": "codex",
                        "env_key": "AGENTBENCH_API_KEY",
                        "api_key": "sk-should-never-be-here",
                    },
                }
            )
        with self.assertRaises(ValueError):
            HLRunConfig.from_mapping(
                {
                    "game": "29_rollman",
                    "provider": {"kind": "codex", "env_key": "KEY"},
                    "invented": True,
                }
            )

    def test_serialized_config_contains_only_environment_variable_name(self):
        from agentbench_frame.hl.config import HLRunConfig

        config = HLRunConfig.from_mapping(
            {
                "game": "29_rollman",
                "provider": {
                    "kind": "codex",
                    "model": "gpt-5.5",
                    "env_key": "AGENTBENCH_API_KEY",
                    "base_url": "https://example.test/responses",
                },
            }
        )

        serialized = config.to_dict()
        self.assertEqual(serialized["provider"]["env_key"], "AGENTBENCH_API_KEY")
        self.assertNotIn("api_key", serialized["provider"])

    def test_defaults_keep_model_bootstrap_and_fixed_opponent_mode(self):
        from agentbench_frame.hl.config import HLRunConfig

        config = HLRunConfig.from_mapping(
            {
                "game": "29_rollman",
                "provider": {"kind": "codex"},
            }
        )

        self.assertEqual(config.origin.mode, "model_bootstrap")
        self.assertIsNone(config.origin.source_run)
        self.assertIsNone(config.origin.source_version)
        self.assertEqual(config.curriculum.mode, "fixed")

    def test_imported_origin_requires_source_run_and_version(self):
        from agentbench_frame.hl.config import HLRunConfig

        base = {
            "game": "29_rollman",
            "provider": {"kind": "codex"},
        }
        for origin in (
            {"mode": "imported_version"},
            {
                "mode": "imported_version",
                "source_run": "run-a",
            },
            {
                "mode": "imported_version",
                "source_version": "v000001",
            },
        ):
            with self.subTest(origin=origin), self.assertRaisesRegex(
                ValueError, "source_run and source_version"
            ):
                HLRunConfig.from_mapping({**base, "origin": origin})

    def test_imported_origin_resets_research_state_by_default(self):
        from agentbench_frame.hl.config import OriginConfig

        origin = OriginConfig(
            mode="imported_version",
            source_run="run-a",
            source_version="v000001",
        )

        self.assertTrue(origin.reset_research_state)

    def test_weakest_failed_curriculum_requires_preservation_and_valid_bounds(self):
        from agentbench_frame.hl.config import HLRunConfig

        base = {
            "game": "29_rollman",
            "provider": {"kind": "codex"},
        }
        invalid_curricula = (
            {
                "mode": "weakest_failed",
                "preserve_passed_opponents": False,
            },
            {
                "mode": "weakest_failed",
                "required_human_opponents": 17,
            },
            {
                "mode": "weakest_failed",
                "stagnation_patience": 0,
            },
        )

        for curriculum in invalid_curricula:
            with self.subTest(curriculum=curriculum), self.assertRaises(
                ValueError
            ):
                HLRunConfig.from_mapping(
                    {**base, "curriculum": curriculum}
                )

    def test_weakest_failed_curriculum_accepts_bootstrap_or_clean_import(self):
        from agentbench_frame.hl.config import HLRunConfig

        base = {
            "game": "29_rollman",
            "provider": {"kind": "codex"},
            "curriculum": {"mode": "weakest_failed"},
        }
        bootstrap = HLRunConfig.from_mapping(base)
        self.assertEqual(bootstrap.origin.mode, "model_bootstrap")
        with self.assertRaisesRegex(ValueError, "reset_session"):
            HLRunConfig.from_mapping(
                {
                    **base,
                    "origin": {
                        "mode": "imported_version",
                        "source_run": "run-a",
                        "source_version": "v000001",
                        "reset_session": False,
                    },
                }
            )
        continued = HLRunConfig.from_mapping(
            {
                **base,
                "origin": {
                    "mode": "imported_version",
                    "source_run": "run-a",
                    "source_version": "v000001",
                    "reset_experience": False,
                },
            }
        )
        self.assertFalse(continued.origin.reset_experience)

    def test_k4_config_parses_linear_proposal_cycle(self):
        from agentbench_frame.hl.config import HLRunConfig

        config = HLRunConfig.from_mapping(
            {
                "game": "29_rollman",
                "provider": {
                    "kind": "codex",
                    "model": "gpt-5.5",
                    "context_mode": "fresh",
                },
                "iteration": {
                    "candidates_per_cycle": 4,
                    "planner_enabled": True,
                    "reducer_enabled": True,
                    "quick_screen_seeds": 1,
                    "finalist_count": 2,
                    "finalist_seeds": 3,
                },
                "selection": {
                    "mode": "linear_lexicographic",
                    "exploration_debt_cycles": 3,
                    "source_size_penalty": False,
                },
                "context": {
                    "use_game_digest": True,
                    "research_state_max_bytes": 16384,
                    "reduction_token_threshold": 250000,
                },
                "evaluation": {
                    "reporting_panel_every_cycle": True,
                    "reporting_seeds_per_opponent": 1,
                    "max_parallel_matches": 4,
                },
            }
        )

        self.assertEqual(config.iteration.candidates_per_cycle, 4)
        self.assertEqual(config.iteration.candidates_per_act, 4)
        self.assertEqual(config.iteration.quick_screen_seeds, 1)
        self.assertEqual(config.iteration.finalist_count, 2)
        self.assertEqual(config.iteration.finalist_seeds, 3)
        self.assertEqual(config.selection.mode, "linear_lexicographic")
        self.assertFalse(config.selection.source_size_penalty)
        self.assertEqual(config.context.research_state_max_bytes, 16384)
        self.assertTrue(config.evaluation.reporting_panel_every_cycle)
        self.assertEqual(config.evaluation.max_parallel_matches, 4)

    def test_linear_k4_rejects_more_finalists_than_candidates(self):
        from agentbench_frame.hl.config import IterationConfig

        with self.assertRaisesRegex(ValueError, "finalist_count"):
            IterationConfig(candidates_per_cycle=4, finalist_count=5)

    def test_repair_config_defaults_are_ablatable(self):
        from agentbench_frame.hl.config import IterationConfig

        value = IterationConfig(candidates_per_cycle=4)

        self.assertTrue(value.scope_contract_required)
        self.assertFalse(value.repair_enabled)
        self.assertEqual(value.repair_top_k, 2)
        self.assertEqual(value.repair_rounds, 1)

    def test_enabled_repair_requires_valid_top_k(self):
        from agentbench_frame.hl.config import IterationConfig

        with self.assertRaisesRegex(ValueError, "repair_top_k"):
            IterationConfig(
                candidates_per_cycle=4,
                planner_enabled=True,
                reducer_enabled=True,
                repair_enabled=True,
                repair_top_k=5,
            )

    def test_enabled_repair_requires_planner(self):
        from agentbench_frame.hl.config import IterationConfig

        with self.assertRaisesRegex(ValueError, "planner"):
            IterationConfig(
                candidates_per_cycle=4,
                planner_enabled=False,
                reducer_enabled=True,
                repair_enabled=True,
            )

    def test_repair_rounds_accepts_only_zero_or_one(self):
        from agentbench_frame.hl.config import IterationConfig

        with self.assertRaisesRegex(ValueError, "repair_rounds"):
            IterationConfig(candidates_per_cycle=4, repair_rounds=2)

    def test_source_size_penalty_is_forbidden_for_linear_search(self):
        from agentbench_frame.hl.config import SelectionConfig

        with self.assertRaisesRegex(ValueError, "source_size_penalty"):
            SelectionConfig(source_size_penalty=True)


if __name__ == "__main__":
    unittest.main()
