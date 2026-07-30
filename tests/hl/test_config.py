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


if __name__ == "__main__":
    unittest.main()
