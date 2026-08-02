import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agentbench_frame.miracle.cli import main


INITIAL = '''
from agentbench_frame.miracle.agent_bridge import EndRoundAgent
class CandidateAgent(EndRoundAgent):
    pass
'''

EVOLVED = '''
from agentbench_frame.miracle.agent_bridge import SampleAgent
class CandidateAgent(SampleAgent):
    pass
'''


@contextmanager
def _mock_openai():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            json.loads(self.rfile.read(length))
            content = json.dumps({"analysis": "Use the working sample policy.",
                                  "strategy_code": EVOLVED})
            response = json.dumps({
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


def test_cli_completes_real_official_logic_iteration(tmp_path, monkeypatch, capsys):
    initial = tmp_path / "initial.py"
    initial.write_text(INITIAL, encoding="utf-8")
    monkeypatch.setenv("MIRACLE_E2E_KEY", "must-not-be-saved")
    with _mock_openai() as base_url:
        config = tmp_path / "loop.toml"
        config.write_text(f'''
agent = "e2e_llm"
initial_strategy = "initial.py"
opponent = "endround"
[llm]
base_url = "{base_url}"
api_key_env = "MIRACLE_E2E_KEY"
model = "mock"
[evaluation]
seeds = [11]
seats = [0]
[budget]
max_iterations = 1
max_rollouts = 2
max_episode_reads = 1
max_decision_reads = 500
max_total_tokens = 100
max_wall_seconds = 120
''', encoding="utf-8")
        assert main(["loop", "--config", str(config),
                     "--data-dir", str(tmp_path / "results")]) == 0

    output = json.loads(capsys.readouterr().out)
    run_dir = next((tmp_path / "results/runs/24_miracle/e2e_llm").iterdir())
    assert output["run_dir"] == str(run_dir.resolve())
    strategy0 = (run_dir / "iterations/iteration-0000/strategy.py").read_text()
    strategy1 = (run_dir / "iterations/iteration-0001/strategy.py").read_text()
    assert strategy0 != strategy1

    for index in (0, 1):
        episode_dir = run_dir / f"iterations/iteration-{index:04d}/episodes"
        result = json.loads(next(episode_dir.glob("*.json")).read_text())
        assert result["terminated_by"] == "normal"
        assert result["errors"] == []
        assert next(episode_dir.glob("*.mrc")).stat().st_size > 0
        assert next(episode_dir.glob("*.trace.jsonl")).stat().st_size > 0

    score = json.loads((run_dir / "score_curve.json").read_text())
    ig = json.loads((run_dir / "ig_curve.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert score["points"][1]["evo"] > score["points"][0]["evo"]
    assert ig["points"][0]["status"] == "baseline"
    assert ig["points"][1]["status"] == "measured"
    assert summary["final_gain"] > 0
    assert summary["total_tokens"] == 30
    assert "wall_hours" in summary
    assert (run_dir / "run.toml").is_file()

    all_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in run_dir.rglob("*") if path.is_file() and path.suffix != ".mrc"
    )
    assert "must-not-be-saved" not in all_text
    events = [json.loads(line)["event"] for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert {
        "battle_started", "battle_finished", "llm_request_started",
        "llm_request_finished", "iteration_finished", "run_finished",
    } <= set(events)
