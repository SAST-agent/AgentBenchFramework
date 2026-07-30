import sys
from pathlib import Path

import pytest

from agentbench_frame.games.rollman.match import ProcessSpec, run_match
from agentbench_frame.games.rollman.state_tracker import FrozenStateTracker


FIXTURES = Path(__file__).with_name("fixtures")


def _python_fixture(name: str) -> ProcessSpec:
    return ProcessSpec(argv=(sys.executable, str(FIXTURES / name)))


def test_match_routes_frozen_frames_and_records_predecision_context(tmp_path):
    result = run_match(
        logic=_python_fixture("fake_logic.py"),
        rollman=_python_fixture("fake_rollman.py"),
        ghosts=_python_fixture("fake_ghosts.py"),
        seed=1729,
        timeout_s=2,
        replay_path=tmp_path / "replay.jsonl",
        trace_path=tmp_path / "trace.jsonl",
    )

    assert result.status == "complete"
    assert result.seed == 1729
    assert result.rollman_score == 0
    assert result.ghosts_score == 0
    assert result.result == "draw"
    assert result.replay.raw_sha256
    assert len(result.rollman_decisions) == 1
    decision = result.rollman_decisions[0]
    assert decision["action"] == 0
    assert decision["memory_id"] == "constant-stay-v1"
    assert decision["state"]["level"] == 1
    assert decision["state"]["round"] == 0
    assert decision["state_id"]


def test_same_seed_and_processes_produce_identical_normalized_replay(tmp_path):
    kwargs = dict(
        logic=_python_fixture("fake_logic.py"),
        rollman=_python_fixture("fake_rollman.py"),
        ghosts=_python_fixture("fake_ghosts.py"),
        seed=8128,
        timeout_s=2,
    )
    first = run_match(
        **kwargs,
        replay_path=tmp_path / "first.jsonl",
        trace_path=tmp_path / "first.trace.jsonl",
    )
    second = run_match(
        **kwargs,
        replay_path=tmp_path / "second.jsonl",
        trace_path=tmp_path / "second.trace.jsonl",
    )

    assert first.replay.normalized_sha256 == second.replay.normalized_sha256
    assert first.rollman_decisions == second.rollman_decisions


@pytest.mark.integration
def test_frozen_backend_is_deterministic_after_seed_injection(tmp_path):
    agentbench_root = Path("/Users/qingle/Code/SAST/AgentBench")
    logic_root = (
        agentbench_root
        / "backend_sources/corpus/29_rollman/logic/gamecode_logic/PacmanLogic"
    )
    if not logic_root.is_dir():
        pytest.skip("frozen AgentBench backend is unavailable")
    wrapper = (
        Path(__file__).parents[2]
        / "src/agentbench_frame/games/rollman/logic_runner.py"
    )

    def one_run(stem: str):
        return run_match(
            logic=ProcessSpec(
                argv=(
                    sys.executable,
                    str(wrapper),
                    "--logic-root",
                    str(logic_root),
                    "--seed",
                    "424242",
                ),
                cwd=logic_root,
            ),
            rollman=_python_fixture("fake_rollman.py"),
            ghosts=_python_fixture("fake_ghosts.py"),
            seed=424242,
            timeout_s=3,
            replay_path=tmp_path / f"{stem}.jsonl",
            trace_path=tmp_path / f"{stem}.trace.jsonl",
            state_tracker=FrozenStateTracker(logic_root),
        )

    first = one_run("frozen-first")
    second = one_run("frozen-second")

    assert first.replay.normalized_sha256 == second.replay.normalized_sha256
    assert first.rollman_score == second.rollman_score
    assert first.ghosts_score == second.ghosts_score
    assert first.rollman_decisions == second.rollman_decisions
    assert len(first.rollman_decisions) > 1


@pytest.mark.integration
def test_from_scratch_candidate_runner_completes_frozen_game(tmp_path):
    framework_root = Path(__file__).parents[2]
    logic_root = Path(
        "/Users/qingle/Code/SAST/AgentBench/"
        "backend_sources/corpus/29_rollman/logic/gamecode_logic/PacmanLogic"
    )
    sdk_root = Path("/Users/qingle/Code/SAST/PacmanSDK-python")
    if not logic_root.is_dir() or not sdk_root.is_dir():
        pytest.skip("frozen Rollman dependencies are unavailable")
    logic_wrapper = (
        framework_root / "src/agentbench_frame/games/rollman/logic_runner.py"
    )
    candidate_runner = (
        framework_root / "src/agentbench_frame/games/rollman/candidate_runner.py"
    )
    candidate_workspace = (
        framework_root
        / "src/agentbench_frame/games/rollman/assets/candidate-template"
    )

    result = run_match(
        logic=ProcessSpec(
            (
                sys.executable,
                str(logic_wrapper),
                "--logic-root",
                str(logic_root),
                "--seed",
                "99",
            ),
            cwd=logic_root,
        ),
        rollman=ProcessSpec(
            (
                sys.executable,
                str(candidate_runner),
                "--workspace",
                str(candidate_workspace),
                "--sdk-root",
                str(sdk_root),
            )
        ),
        ghosts=_python_fixture("fake_ghosts.py"),
        seed=99,
        timeout_s=3,
        replay_path=tmp_path / "candidate.jsonl",
        trace_path=tmp_path / "candidate.trace.jsonl",
        state_tracker=FrozenStateTracker(logic_root),
    )

    assert result.status == "complete"
    assert result.rollman_decisions[0]["memory_id"] == "from-scratch-v0"
    assert len(result.rollman_decisions) > 1
