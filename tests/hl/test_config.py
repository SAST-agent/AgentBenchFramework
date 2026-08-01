import unittest


class HLConfigTests(unittest.TestCase):
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
        for reset_field in ("reset_session", "reset_experience"):
            with self.subTest(reset_field=reset_field), self.assertRaisesRegex(
                ValueError, reset_field
            ):
                HLRunConfig.from_mapping(
                    {
                        **base,
                        "origin": {
                            "mode": "imported_version",
                            "source_run": "run-a",
                            "source_version": "v000001",
                            reset_field: False,
                        },
                    }
                )

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
                },
            }
        )

        self.assertEqual(config.iteration.candidates_per_cycle, 4)
        self.assertEqual(config.iteration.candidates_per_act, 4)
        self.assertEqual(config.iteration.finalist_count, 2)
        self.assertEqual(config.selection.mode, "linear_lexicographic")
        self.assertFalse(config.selection.source_size_penalty)
        self.assertEqual(config.context.research_state_max_bytes, 16384)
        self.assertTrue(config.evaluation.reporting_panel_every_cycle)

    def test_linear_k4_rejects_more_finalists_than_candidates(self):
        from agentbench_frame.hl.config import IterationConfig

        with self.assertRaisesRegex(ValueError, "finalist_count"):
            IterationConfig(candidates_per_cycle=4, finalist_count=5)

    def test_source_size_penalty_is_forbidden_for_linear_search(self):
        from agentbench_frame.hl.config import SelectionConfig

        with self.assertRaisesRegex(ValueError, "source_size_penalty"):
            SelectionConfig(source_size_penalty=True)


if __name__ == "__main__":
    unittest.main()
