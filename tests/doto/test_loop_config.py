from pathlib import Path

import pytest

from agentbench_frame.doto.loop_config import LoopConfig


def write_config(tmp_path: Path, **budget_overrides) -> Path:
    (tmp_path / "playerAI.cpp").write_text("// candidate\n")
    (tmp_path / "opponent.out").write_text("binary\n")
    budget = {
        "max_iterations": 3, "max_builds": 4, "max_rollouts": 8,
        "max_episode_reads": 6, "max_frame_reads": 100,
        "max_total_tokens": 20000, "max_wall_seconds": 3600,
    }
    budget.update(budget_overrides)
    lines = [
        'agent = "test-model"', 'initial_player_ai = "playerAI.cpp"',
        '', '[[opponents]]', 'name = "baseline"', 'executable = "opponent.out"',
        '', '[llm]', 'base_url = "https://example.invalid"',
        'api_key_env = "DOTO_TEST_KEY"', 'model = "model"',
        '', '[evaluation]', 'seeds = [11]', 'seats = [0, 1]',
        'realtime_scale = 1.0', '', '[budget]',
        *[f"{key} = {value}" for key, value in budget.items()],
    ]
    path = tmp_path / "loop.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_defaults_to_streaming_one_million_context_and_no_output_cap(tmp_path):
    config = LoopConfig.load(write_config(tmp_path))
    assert config.llm.stream is True
    assert config.llm.max_context_tokens == 1_000_000
    assert config.llm.max_tokens is None
    assert config.initial_player_ai == (tmp_path / "playerAI.cpp").resolve()
    assert config.opponents[0].executable == (tmp_path / "opponent.out").resolve()


@pytest.mark.parametrize("field", [
    "max_iterations", "max_builds", "max_rollouts", "max_episode_reads",
    "max_frame_reads", "max_total_tokens", "max_wall_seconds",
])
def test_budget_limits_must_be_positive(tmp_path, field):
    with pytest.raises(ValueError, match=field):
        LoopConfig.load(write_config(tmp_path, **{field: 0}))


def test_rejects_nonofficial_realtime_scale(tmp_path):
    path = write_config(tmp_path)
    path.write_text(path.read_text().replace("realtime_scale = 1.0", "realtime_scale = 0.0"))
    with pytest.raises(ValueError, match="realtime_scale"):
        LoopConfig.load(path)
