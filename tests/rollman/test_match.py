import sys
import json
from pathlib import Path


def test_process_cleanup_does_not_mask_broken_pipe_error():
    from agentbench_frame.games.rollman.match import _stop

    class BrokenStream:
        def close(self):
            raise BrokenPipeError("child already closed stdin")

    class ExitedProcess:
        pid = 123456
        stdin = BrokenStream()
        stdout = None
        stderr = None

        @staticmethod
        def poll():
            return 1

        @staticmethod
        def wait(timeout=None):
            return 1

    _stop(ExitedProcess())


def test_exited_player_stderr_is_bounded_for_fault_diagnostics():
    import io

    from agentbench_frame.games.rollman.match import _stderr_tail

    class ExitedProcess:
        stderr = io.BytesIO(b"prefix\nTraceback: numpy truth-value failure\n")

        @staticmethod
        def poll():
            return 1

    assert _stderr_tail(ExitedProcess(), max_bytes=32) == (
        "back: numpy truth-value failure\n"
    )

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


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox integration")
def test_untrusted_runtime_cannot_spawn_subprocess(tmp_path):
    from agentbench_frame.games.rollman.match import _start, _stop

    marker = tmp_path / "spawned.txt"
    script = tmp_path / "spawn_parent.py"
    script.write_text(
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', "
        f"\"from pathlib import Path; Path({str(marker)!r}).write_text('bad')\""
        "])\n",
        encoding="utf-8",
    )
    process = _start(
        ProcessSpec(
            argv=(sys.executable, str(script)),
            cwd=tmp_path,
            untrusted=True,
            read_roots=(tmp_path,),
        ),
        "spawn-fixture",
    )
    try:
        return_code = process.wait(timeout=3)
        assert return_code != 0
        assert not marker.exists()
    finally:
        _stop(process)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox integration")
def test_untrusted_direct_process_memory_is_limited(tmp_path):
    from agentbench_frame.games.rollman.match import _start, _stop

    script = tmp_path / "memory_process.py"
    script.write_text(
        "import time\n"
        "memory = bytearray(160 * 1024 * 1024)\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    process = _start(
        ProcessSpec(
            argv=(sys.executable, str(script)),
            cwd=tmp_path,
            untrusted=True,
            memory_limit_mb=100,
            read_roots=(tmp_path,),
        ),
        "memory-fixture",
    )
    try:
        return_code = process.wait(timeout=6)
        assert return_code != 0
    finally:
        _stop(process)


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
    trace = [
        json.loads(line)
        for line in first.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    limits = [item for item in trace if item["type"] == "round_config"]
    assert limits[0] == {"type": "round_config", "time": 20.0, "length": 1024}
    assert any(item["time"] == 1.0 for item in limits[1:])


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
            ),
            untrusted=True,
            read_roots=(
                framework_root / "src",
                candidate_workspace,
                sdk_root,
            ),
            denied_paths=(framework_root / ".env",),
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


@pytest.mark.integration
def test_frozen_backend_counts_game_rule_ghost_timeout_as_rollman_win(tmp_path):
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

    result = run_match(
        logic=ProcessSpec(
            argv=(
                sys.executable,
                str(wrapper),
                "--logic-root",
                str(logic_root),
                "--seed",
                "101",
            ),
            cwd=logic_root,
        ),
        rollman=_python_fixture("fake_rollman.py"),
        ghosts=_python_fixture("slow_ghosts.py"),
        seed=101,
        timeout_s=3,
        replay_path=tmp_path / "timeout.jsonl",
        trace_path=tmp_path / "timeout.trace.jsonl",
        state_tracker=FrozenStateTracker(logic_root),
    )

    assert result.status == "complete"
    assert result.result == "win"
    assert result.end_state == ("OK", "TLE")
    assert result.rollman_score > result.ghosts_score
    assert any(
        item.get("type") == "ai_fault"
        and item.get("player") == 1
        and item.get("error") == "TLE"
        for item in (
            json.loads(line)
            for line in result.trace_path.read_text(encoding="utf-8").splitlines()
        )
    )
