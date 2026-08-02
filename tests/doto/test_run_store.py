import json

import pytest

from agentbench_frame.doto.loop_config import LoopConfig
from agentbench_frame.doto.run_store import BudgetExceeded, BudgetLedger, DotoRunStore
from tests.doto.test_loop_config import write_config


def test_frame_reads_are_cumulative_across_iterations(tmp_path):
    config = LoopConfig.load(write_config(tmp_path, max_frame_reads=3))
    ledger = BudgetLedger(config.budget)
    ledger.charge_read(episodes=1, frames=2)
    with pytest.raises(BudgetExceeded, match="frame_reads"):
        ledger.charge_read(episodes=1, frames=2)


def test_all_budget_and_time_dimensions_are_visible(tmp_path):
    config = LoopConfig.load(write_config(tmp_path))
    ledger = BudgetLedger(config.budget)
    ledger.charge_build(1)
    ledger.charge_rollout(2)
    ledger.charge_usage({"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7})
    ledger.charge_context(10, 100)
    ledger.charge_compile_time(.2)
    ledger.charge_battle_time(.3)
    ledger.charge_api_time(.4)
    assert ledger.snapshot()["builds"] == 1
    assert ledger.snapshot()["total_tokens"] == 7


def test_run_copies_exact_skills_and_never_secret(tmp_path, monkeypatch):
    config = LoopConfig.load(write_config(tmp_path))
    monkeypatch.setenv("DOTO_TEST_KEY", "never-save-me")
    store = DotoRunStore.create(config, tmp_path / "results", run_id="fixed")
    assert (store.run_dir / "skills/doto-harness.SKILL.md").is_file()
    assert (store.run_dir / "skills/doto-replay-reader.SKILL.md").is_file()
    store.write_event("started", api_key_env=config.llm.api_key_env)
    store.finish({"status": "complete", "total_steps": 1, "total_episodes": 2})
    text = "".join(path.read_text(errors="ignore") for path in store.run_dir.rglob("*") if path.is_file())
    assert "never-save-me" not in text
    assert json.loads((store.run_dir / "summary.json").read_text())["game"] == "23_doto"
