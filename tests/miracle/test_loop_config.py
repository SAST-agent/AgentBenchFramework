from pathlib import Path

import pytest

from agentbench_frame.miracle.loop_config import LoopConfig


def _write_config(tmp_path: Path, *, opponent="sample", max_iterations=1) -> Path:
    initial = tmp_path / "initial.py"
    initial.write_text("# initial\n", encoding="utf-8")
    path = tmp_path / "loop.toml"
    path.write_text(
        f'''agent = "tested_llm"
initial_strategy = "initial.py"
opponent = "{opponent}"

[llm]
base_url = "http://127.0.0.1:8123"
api_key_env = "TEST_LLM_KEY"
model = "mock-model"
reasoning_effort = "low"

[evaluation]
seeds = [11]
seats = [0]

[budget]
max_iterations = {max_iterations}
max_rollouts = 4
max_episode_reads = 1
max_decision_reads = 200
max_total_tokens = 10000
max_wall_seconds = 600
''',
        encoding="utf-8",
    )
    return path


def test_loads_minimal_loop_config(tmp_path):
    cfg = LoopConfig.from_toml(_write_config(tmp_path))

    assert cfg.agent == "tested_llm"
    assert cfg.initial_strategy == (tmp_path / "initial.py").resolve()
    assert cfg.evaluation.seeds == (11,)
    assert cfg.evaluation.seats == (0,)
    assert cfg.budget.max_iterations == 1
    assert cfg.llm.temperature == 0.0
    assert cfg.llm.max_tokens == 8192
    assert cfg.llm.timeout_seconds == 120.0
    assert cfg.llm.reasoning_effort == "low"


@pytest.mark.parametrize(
    ("opponent", "max_iterations", "message"),
    [("not_registered", 1, "opponent"), ("sample", 0, "max_iterations")],
)
def test_rejects_unknown_opponent_and_nonpositive_budget(
    tmp_path, opponent, max_iterations, message,
):
    with pytest.raises(ValueError, match=message):
        LoopConfig.from_toml(
            _write_config(tmp_path, opponent=opponent, max_iterations=max_iterations)
        )


def test_public_config_never_resolves_or_contains_api_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "top-secret-value")
    cfg = LoopConfig.from_toml(_write_config(tmp_path))

    public = cfg.public_dict()

    assert public["llm"]["api_key_env"] == "TEST_LLM_KEY"
    assert "top-secret-value" not in repr(public)


def test_rejects_missing_initial_strategy(tmp_path):
    path = _write_config(tmp_path)
    (tmp_path / "initial.py").unlink()

    with pytest.raises(ValueError, match="initial_strategy"):
        LoopConfig.from_toml(path)
