import json
from pathlib import Path

from agentbench_frame.miracle.host import MatchResult
from agentbench_frame.miracle.llm_client import StrategyProposal
from agentbench_frame.miracle.loop import build_messages, run_loop
from agentbench_frame.miracle.loop_config import LoopConfig


INITIAL = '''
from agentbench_frame.miracle.agent_bridge import MiracleAgent
class CandidateAgent(MiracleAgent):
    def choose_cards(self, camp):
        return {"artifacts":["HolyLight"],"creatures":["Archer","Swordsman","BlackBat"]}
    def act(self, obs):
        return {"operation_type":"endround","operation_parameters":{}}
'''

EVOLVED = INITIAL.replace('"endround"', '"surrender"')


def _config(tmp_path, *, iterations=1):
    (tmp_path / "initial.py").write_text(INITIAL, encoding="utf-8")
    path = tmp_path / "loop.toml"
    path.write_text(f'''
agent = "tested_llm"
initial_strategy = "initial.py"
opponent = "endround"
[llm]
base_url = "http://localhost:1"
api_key_env = "KEY"
model = "mock"
[evaluation]
seeds = [11]
seats = [0]
[budget]
max_iterations = {iterations}
max_rollouts = 4
max_episode_reads = 1
max_decision_reads = 5
max_total_tokens = 100
max_wall_seconds = 60
''', encoding="utf-8")
    return LoopConfig.from_toml(path)


def _trace_row(obs):
    return {"seq": 1, "kind": "from_logic", "payload": {
        "listen": [obs["camp"]], "content": ["000000" + json.dumps(obs)],
    }}


def _fake_match(agent0, agent1, *, replay_dir, seed, tag, **kwargs):
    replay_dir = Path(replay_dir)
    replay_dir.mkdir(parents=True, exist_ok=True)
    obs = {"camp": 0, "round": 1, "map": {"units": [], "barracks": []},
           "players": [[[], 0, 0, [], []], [[], 0, 0, [], []]]}
    action = agent0.act(obs)["operation_type"]
    score = 10 if action == "surrender" else 1
    replay = replay_dir / f"{tag}.mrc"
    trace = replay_dir / f"{tag}.mrc.trace.jsonl"
    replay.write_bytes(b"real-ish-replay")
    trace.write_text(json.dumps(_trace_row(obs)) + "\n", encoding="utf-8")
    return MatchResult(0, (score, 0), 1, str(replay), str(trace), 0.01, "normal", [])


class FakeClient:
    def propose_strategy(self, messages):
        raw = {"choices": [{"message": {"content": "saved"}}]}
        return StrategyProposal(
            "improve", EVOLVED,
            {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            {"model": "mock", "messages": messages}, raw, 0.02, False,
        )


def test_build_messages_contains_contract_skills_source_evidence_and_budget():
    messages = build_messages(
        INITIAL, {"harness": "HARNESS", "replay": "REPLAY"},
        [{"episode_id": "ep", "observations": [{"round": 1}]}],
        {"score": 1}, {"total_tokens": 0},
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    combined = "\n".join(message["content"] for message in messages)
    assert "CandidateAgent" in combined
    assert "HARNESS" in combined and "REPLAY" in combined
    assert "strategy_code" in combined


def test_run_loop_saves_baseline_candidate_curves_and_usage(tmp_path):
    run_dir = run_loop(
        _config(tmp_path), client=FakeClient(), match_runner=_fake_match,
        data_dir=tmp_path / "results", run_id="fixed-run",
    )

    assert (run_dir / "iterations/iteration-0000/strategy.py").exists()
    assert (run_dir / "iterations/iteration-0001/candidate.py").exists()
    assert (run_dir / "iterations/iteration-0001/strategy.py").exists()
    assert (run_dir / "iterations/iteration-0001/llm_request.json").exists()
    score_curve = json.loads((run_dir / "score_curve.json").read_text())
    ig_curve = json.loads((run_dir / "ig_curve.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())

    assert score_curve["points"][0]["evo"] == 1.0
    assert score_curve["points"][1]["evo"] == 10.0
    assert score_curve["points"][1]["gain"] == 9.0
    assert ig_curve["points"][1]["status"] == "measured"
    assert summary["total_tokens"] == 20
    assert summary["final_gain"] == 9.0
    events = (run_dir / "events.jsonl").read_text()
    assert "llm_request_started" in events and "llm_request_finished" in events


class InvalidSourceClient(FakeClient):
    def propose_strategy(self, messages):
        proposal = super().propose_strategy(messages)
        return StrategyProposal(
            proposal.analysis, "class CandidateAgent(:", proposal.usage,
            proposal.request_body, proposal.raw_response, proposal.latency_seconds,
        )


def test_invalid_candidate_is_preserved_without_advancing_strategy(tmp_path):
    run_dir = run_loop(
        _config(tmp_path), client=InvalidSourceClient(), match_runner=_fake_match,
        data_dir=tmp_path / "results", run_id="failed-run",
    )

    failed = json.loads((run_dir / "iterations/iteration-0001/iteration.json").read_text())
    assert failed["status"] == "failed"
    assert failed["failure_stage"] == "compile"
    assert (run_dir / "iterations/iteration-0001/candidate.py").exists()
    assert not (run_dir / "iterations/iteration-0001/strategy.py").exists()
    assert json.loads((run_dir / "summary.json").read_text())["failure_counts"]["compile"] == 1


class FailThenSucceedClient(FakeClient):
    def __init__(self):
        self.calls = 0

    def propose_strategy(self, messages):
        self.calls += 1
        proposal = super().propose_strategy(messages)
        if self.calls == 1:
            return StrategyProposal(
                proposal.analysis, "class CandidateAgent(:", proposal.usage,
                proposal.request_body, proposal.raw_response, proposal.latency_seconds,
            )
        return proposal


def test_later_iteration_can_continue_after_failed_update(tmp_path):
    run_dir = run_loop(
        _config(tmp_path, iterations=2), client=FailThenSucceedClient(),
        match_runner=_fake_match, data_dir=tmp_path / "results", run_id="resume-run",
    )

    second = json.loads((run_dir / "iterations/iteration-0002/iteration.json").read_text())
    assert second["status"] == "accepted"
