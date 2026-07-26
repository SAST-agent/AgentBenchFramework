import json
from pathlib import Path
import sys

import pytest

from agentbench_frame.generals.models import AgentProcessSpec, ProcessLimits
from agentbench_frame.generals.process import (
    DecisionTimeout,
    ManagedAgentProcess,
    PrematureExit,
)


FAKE = Path(__file__).parent / "fixtures" / "fake_agent.py"


def limits(**overrides):
    values = {
        "startup_timeout_s": 1,
        "decision_timeout_s": 0.15,
        "match_timeout_s": 2,
        "max_packet_bytes": 65536,
        "max_artifact_bytes": 1024,
    }
    values.update(overrides)
    return ProcessLimits(**values)


def spec(mode, tmp_path):
    return AgentProcessSpec(
        agent_id="fake",
        argv=(sys.executable, str(FAKE), mode),
        cwd=tmp_path,
        env={},
    )


def test_managed_agent_round_trip_and_artifacts(tmp_path):
    artifacts = tmp_path / "artifacts"
    with ManagedAgentProcess(spec("valid", tmp_path), limits(), artifacts) as agent:
        agent.send_initial({"Player": 0})
        assert agent.request_turn() == ((8,),)
    assert (artifacts / "agent.protocol.bin").exists()
    metadata = json.loads((artifacts / "process.json").read_text())
    assert metadata["pid"] > 0
    assert metadata["returncode"] is not None


def test_decision_timeout_is_typed_and_kills_process_group(tmp_path):
    artifacts = tmp_path / "artifacts"
    with pytest.raises(DecisionTimeout):
        with ManagedAgentProcess(spec("hang", tmp_path), limits(), artifacts) as agent:
            agent.send_initial({"Player": 0})
            agent.request_turn()
    metadata = json.loads((artifacts / "process.json").read_text())
    assert metadata["returncode"] is not None


def test_premature_exit_is_typed(tmp_path):
    with pytest.raises(PrematureExit):
        with ManagedAgentProcess(spec("exit", tmp_path), limits(), tmp_path / "a") as agent:
            agent.send_initial({"Player": 0})
            agent.request_turn()


def test_stderr_is_bounded(tmp_path):
    artifacts = tmp_path / "artifacts"
    with ManagedAgentProcess(spec("stderr", tmp_path), limits(), artifacts) as agent:
        agent.send_initial({"Player": 0})
        assert agent.request_turn() == ((8,),)
    assert (artifacts / "agent.stderr.log").stat().st_size <= 1024
