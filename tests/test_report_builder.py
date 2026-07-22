import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from aggregate import find_runs  # noqa: E402
from report_builder import build_site, load_run  # noqa: E402


class ResearchDashboardTests(unittest.TestCase):
    def _make_data(self, root: Path) -> None:
        run_dir = root / "runs" / "arena" / "rl-v1" / "run-1"
        run_dir.mkdir(parents=True)
        (root / "registry.toml").write_text(
            """[[runs]]
run_id = "run-1"
game = "arena"
agent = "rl-v1"
type = "eval"
path = "runs/arena/rl-v1/run-1"
created = "2026-07-23T08:00:00+00:00"

[runs.summary]
benchmark_score = 0.75
"""
        )
        (run_dir / "summary.json").write_text(json.dumps({
            "run_id": "run-1",
            "game": "arena",
            "agent": "rl-v1",
            "run_type": "eval",
            "created": "2026-07-23T08:00:00+00:00",
            "benchmark_score": 0.75,
            "raw_score": 0.5,
            "evo_score": 0.75,
            "gain": 0.25,
            "evaluation_status": "complete",
            "budget": {
                "learning_episodes": 8,
                "learning_env_steps": 64,
                "learning_coding_agent_acts": 2,
                "learning_total_tokens": 1200,
                "learning_time_s": 31.5,
                "total_episodes": 10,
            },
        }))
        events = [
            {"event_type": "coding_agent_act", "act_id": "a1", "version_after": "v1"},
            {"event_type": "act_evaluation", "act_id": "a1", "benchmark_score": 0.5},
            {"event_type": "policy_kl_trace", "episode": 1, "trace": [0.1, 0.3]},
            {"event_type": "coding_agent_act", "act_id": "a2", "version_after": "v2"},
            {"event_type": "act_evaluation", "act_id": "a2", "benchmark_score": 0.75},
            {"event_type": "policy_kl_trace", "episode": 2, "trace": [0.2]},
        ]
        (run_dir / "events.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events)
        )

    def test_run_derives_research_series_from_summary_and_raw_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_data(root)
            entry = {
                "run_id": "run-1",
                "game": "arena",
                "agent": "rl-v1",
                "type": "eval",
                "path": "runs/arena/rl-v1/run-1",
                "summary": {},
            }
            run = load_run(root, entry)

            self.assertEqual(run.research["score_history"], [
                {"x": 1, "score": 0.5, "act_id": "a1"},
                {"x": 2, "score": 0.75, "act_id": "a2"},
            ])
            self.assertEqual(run.research["ig_history"], [
                {"episode": 1, "ig": 0.2, "decision_steps": 2},
                {"episode": 2, "ig": 0.2, "decision_steps": 1},
            ])
            self.assertAlmostEqual(run.research["auc_coding_agent_act"], 0.625)
            self.assertEqual(run.research["budget"]["learning_episodes"], 8)

    def test_build_renders_research_dashboard_and_inline_icon_sprite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "site"
            self._make_data(root)
            build_site(root, output)

            html = (output / "index.html").read_text()
            self.assertIn("Research CI", html)
            self.assertIn("Benchmark score", html)
            self.assertIn("Learning-only budget", html)
            self.assertIn("Policy change", html)
            self.assertIn("AUC / act", html)
            self.assertNotIn("&#34;", html)
            self.assertIn('id="icon-trophy"', html)
            self.assertIn('id="icon-spark"', html)
            self.assertIn("0.75", html)
            self.assertTrue((output / "data.json").exists())

    def test_aggregate_preserves_research_summary_scalars(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "arena" / "rl-v1" / "run-1"
            run_dir.mkdir(parents=True)
            (run_dir / "run.toml").write_text("[run]\ntype = \"eval\"\n")
            (run_dir / "summary.json").write_text(json.dumps({
                "run_type": "eval",
                "benchmark_score": 0.75,
                "raw_score": 0.5,
                "evo_score": 0.75,
                "gain": 0.25,
                "evaluation_status": "complete",
            }))

            records = find_runs(root)

            self.assertEqual(records[0]["summary"]["benchmark_score"], 0.75)
            self.assertEqual(records[0]["summary"]["gain"], 0.25)
            self.assertEqual(records[0]["summary"]["evaluation_status"], "complete")

    def test_research_derives_gain_from_raw_and_evolved_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "arena" / "rule-v2" / "run-2"
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(json.dumps({
                "raw_score": 0.4,
                "evo_score": 0.7,
            }))
            run = load_run(root, {
                "run_id": "run-2", "game": "arena", "agent": "rule-v2",
                "type": "eval", "path": "runs/arena/rule-v2/run-2", "summary": {},
            })

            self.assertAlmostEqual(run.gain, 0.3)

    def test_incomplete_evaluation_remains_an_explicit_score_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "arena" / "rl-v3" / "run-3"
            run_dir.mkdir(parents=True)
            (run_dir / "events.jsonl").write_text(
                json.dumps({"event_type": "coding_agent_act", "act_id": "a1"}) + "\n"
                + json.dumps({
                    "event_type": "act_evaluation",
                    "act_id": "a1",
                    "evaluation_status": "incomplete",
                }) + "\n"
            )
            run = load_run(root, {
                "run_id": "run-3", "game": "arena", "agent": "rl-v3",
                "type": "eval", "path": "runs/arena/rl-v3/run-3", "summary": {},
            })

            self.assertEqual(run.research["score_history"], [
                {"x": 1, "score": None, "act_id": "a1"},
            ])

    def test_report_keeps_per_case_results_and_separate_occupancy_shift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "arena" / "cc" / "run-4"
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(json.dumps({
                "benchmark_score": 0.5,
                "benchmark_results": [
                    {"case_id": "case-weak", "outcome": "win", "valid": True},
                    {"case_id": "case-strong", "outcome": "loss", "valid": True},
                ],
            }))
            (run_dir / "events.jsonl").write_text("\n".join([
                json.dumps({"event_type": "coding_agent_act", "act_id": "a1"}),
                json.dumps({"event_type": "act_evaluation", "act_id": "a1", "score": 0.5,
                            "coding_agent_act": 1, "episode": 2, "env_step": 5,
                            "token": 100, "time_s": 1.5}),
                json.dumps({"event_type": "occupancy", "episode": 1,
                            "occupancy_shift": 0.25, "state_count": 3}),
                "{bad-json",
                json.dumps({"event_type": "future_v2", "event_id": "x", "run_id": "r"}),
            ]) + "\n")
            run = load_run(root, {
                "run_id": "run-4", "game": "arena", "agent": "cc", "type": "eval",
                "path": "runs/arena/cc/run-4", "summary": {},
            })
            output = root / "site"
            build_site(root, output)
            html = (output / "index.html").read_text()

            self.assertEqual(run.research["benchmark_results"][1]["case_id"], "case-strong")
            self.assertEqual(run.research["occupancy_history"][0]["shift"], 0.25)
            self.assertEqual(run.research["quality"]["malformed_lines"], 1)
            self.assertEqual(run.research["quality"]["unknown_event_types"], 1)
            self.assertIn("case-strong", html)
            self.assertIn("Quality diagnostics", html)

    def test_report_derives_named_auc_axes_from_budget_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "arena" / "rl" / "run-5"
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(json.dumps({"benchmark_score": 1.0}))
            events = [
                {"event_type": "act_evaluation", "act_id": "a1", "score": 0.0,
                 "coding_agent_act": 0, "episode": 0, "env_step": 0,
                 "token": 0, "time_s": 0},
                {"event_type": "act_evaluation", "act_id": "a2", "score": 1.0,
                 "coding_agent_act": 2, "episode": 4, "env_step": 8,
                 "token": 20, "time_s": 2},
            ]
            (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
            run = load_run(root, {
                "run_id": "run-5", "game": "arena", "agent": "rl", "type": "eval",
                "path": "runs/arena/rl/run-5", "summary": {},
            })

            self.assertEqual(run.research["auc"]["auc_coding_agent_act"], 1.0)
            self.assertEqual(run.research["auc"]["auc_episode"], 2.0)
            self.assertEqual(run.research["auc"]["auc_env_step"], 4.0)
            self.assertEqual(run.research["auc"]["auc_token"], 10.0)
            self.assertEqual(run.research["auc"]["auc_time_s"], 1.0)


if __name__ == "__main__":
    unittest.main()
