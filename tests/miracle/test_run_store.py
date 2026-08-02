import json
import tomllib

import pytest

from agentbench_frame.miracle.loop_config import LoopConfig
from agentbench_frame.miracle.run_store import BudgetExceeded, BudgetLedger, MiracleRunStore


def _config(tmp_path):
    initial = tmp_path / "initial.py"
    initial.write_text("# strategy\n", encoding="utf-8")
    config = tmp_path / "loop.toml"
    config.write_text('''
agent = "tested_llm"
initial_strategy = "initial.py"
opponent = "sample"
[llm]
base_url = "http://localhost:1"
api_key_env = "KEY"
model = "model"
[evaluation]
seeds = [11]
seats = [0]
[budget]
max_iterations = 1
max_rollouts = 2
max_episode_reads = 1
max_decision_reads = 5
max_total_tokens = 30
max_wall_seconds = 60
''', encoding="utf-8")
    return LoopConfig.from_toml(config)


def test_creates_results_compatible_run_and_copies_skills(tmp_path):
    config = _config(tmp_path)
    store = MiracleRunStore.create(config, data_dir=tmp_path / "results", run_id="run-fixed")

    assert store.run_dir == tmp_path / "results/runs/24_miracle/tested_llm/run-fixed"
    assert store.iteration_dir(0).name == "iteration-0000"
    assert (store.run_dir / "skills/miracle-harness.SKILL.md").is_file()
    assert (store.run_dir / "skills/miracle-replay-reader.SKILL.md").is_file()
    meta = tomllib.loads((store.run_dir / "run.toml").read_text(encoding="utf-8"))
    assert meta["run"]["type"] == "rule_iter"
    assert not (store.run_dir / "summary.json").exists()

    store.write_event("iteration_started", iteration=0)
    store.finish({"status": "complete", "total_tokens": 0})

    event = json.loads((store.run_dir / "events.jsonl").read_text().splitlines()[0])
    assert event["event"] == "iteration_started"
    assert json.loads((store.run_dir / "summary.json").read_text())["status"] == "complete"
    assert not list(store.run_dir.rglob("*.tmp"))


def test_budget_ledger_charges_exact_limits_then_rejects_next(tmp_path):
    ledger = BudgetLedger(_config(tmp_path).budget)
    ledger.charge_rollout(2)
    ledger.charge_read(episodes=1, decisions=5)
    ledger.charge_usage({"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})
    ledger.charge_api_time(1.25)
    ledger.charge_battle_time(2.5)

    snapshot = ledger.snapshot()
    assert snapshot["rollouts"] == 2
    assert snapshot["episode_reads"] == 1
    assert snapshot["decision_reads"] == 5
    assert snapshot["total_tokens"] == 30
    assert snapshot["api_seconds"] == 1.25
    assert snapshot["battle_seconds"] == 2.5

    with pytest.raises(BudgetExceeded) as raised:
        ledger.charge_rollout()
    assert raised.value.dimension == "rollouts"


def test_budget_rejects_each_bounded_counter(tmp_path):
    config = _config(tmp_path).budget
    cases = [
        (lambda ledger: ledger.charge_read(2, 0), "episode_reads"),
        (lambda ledger: ledger.charge_read(0, 6), "decision_reads"),
        (lambda ledger: ledger.charge_usage({"total_tokens": 31}), "total_tokens"),
    ]
    for charge, dimension in cases:
        ledger = BudgetLedger(config)
        with pytest.raises(BudgetExceeded) as raised:
            charge(ledger)
        assert raised.value.dimension == dimension
