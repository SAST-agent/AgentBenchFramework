import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentbench_frame.miracle.llm_client import ChatCompletionsClient, LLMRequestError
from agentbench_frame.miracle.loop_config import LLMConfig


@contextmanager
def _server(response, status=200):
    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            captured.update({
                "path": self.path,
                "headers": dict(self.headers),
                "body": json.loads(self.rfile.read(length)),
            })
            body = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", captured
    finally:
        httpd.shutdown()
        thread.join()


def _response(content):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def _config(base_url):
    return LLMConfig(base_url, "TEST_LLM_KEY", "mock-model", max_tokens=123)


def test_calls_chat_completions_and_parses_complete_source(monkeypatch):
    content = json.dumps({"analysis": "improve", "strategy_code": "class CandidateAgent: pass"})
    monkeypatch.setenv("TEST_LLM_KEY", "secret-token")
    with _server(_response(content)) as (base_url, captured):
        proposal = ChatCompletionsClient(_config(base_url)).propose_strategy(
            [{"role": "user", "content": "go"}]
        )

    assert captured["path"] == "/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["body"]["model"] == "mock-model"
    assert captured["body"]["max_tokens"] == 123
    assert proposal.strategy_code == "class CandidateAgent: pass"
    assert proposal.usage["total_tokens"] == 30
    assert proposal.normalized_fence is False


def test_accepts_one_fenced_json_object():
    content = '```json\n{"analysis":"a","strategy_code":"class CandidateAgent: pass"}\n```'
    with _server(_response(content)) as (base_url, _):
        proposal = ChatCompletionsClient(_config(base_url)).propose_strategy([])

    assert proposal.normalized_fence is True


def test_malformed_proposal_has_stage_and_raw_response():
    response = _response("not-json")
    with _server(response) as (base_url, _):
        with pytest.raises(LLMRequestError) as raised:
            ChatCompletionsClient(_config(base_url)).propose_strategy([])

    assert raised.value.stage == "proposal_json"
    assert raised.value.raw_response == response


def test_http_error_never_exposes_api_key(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "never-print-this")
    with _server({"error": "denied"}, status=401) as (base_url, _):
        with pytest.raises(LLMRequestError) as raised:
            ChatCompletionsClient(_config(base_url)).propose_strategy([])

    assert raised.value.stage == "request"
    assert "never-print-this" not in str(raised.value)
