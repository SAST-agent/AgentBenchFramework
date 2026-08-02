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
    return LLMConfig(
        base_url, "TEST_LLM_KEY", "mock-model",
        max_tokens=123, reasoning_effort="low", stream=False,
    )


@contextmanager
def _sse_server(events):
    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            captured["body"] = json.loads(self.rfile.read(length))
            body = "".join(events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
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


def test_calls_chat_completions_and_parses_complete_source(monkeypatch):
    content = json.dumps({"analysis": "improve", "strategy_code": "class CandidateAgent: pass"})
    monkeypatch.setenv("TEST_LLM_KEY", "secret-token")
    with _server(_response(content)) as (base_url, captured):
        proposal = ChatCompletionsClient(_config(base_url)).propose_strategy(
            [{"role": "user", "content": "go"}]
        )

    assert captured["path"] == "/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["headers"]["User-Agent"] == "AgentBenchFramework/0.1"
    assert captured["body"]["model"] == "mock-model"
    assert captured["body"]["max_tokens"] == 123
    assert captured["body"]["reasoning_effort"] == "low"
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
    assert raised.value.raw_response["choices"] == response["choices"]
    assert raised.value.raw_response["usage"] == response["usage"]
    assert raised.value.raw_response["stream"] is False
    assert raised.value.usage["total_tokens"] == 30
    assert raised.value.latency_seconds >= 0


def test_http_error_never_exposes_api_key(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "never-print-this")
    with _server({"error": "denied"}, status=401) as (base_url, _):
        with pytest.raises(LLMRequestError) as raised:
            ChatCompletionsClient(_config(base_url)).propose_strategy([])

    assert raised.value.stage == "request"
    assert "never-print-this" not in str(raised.value)


def _data(value):
    return "data: " + json.dumps(value) + "\n\n"


def test_stream_reconstructs_reasoning_content_usage_and_telemetry():
    events = [
        ": heartbeat\n\n",
        _data({"id": "chat-1", "model": "mock-model", "choices": [{
            "delta": {"role": "assistant", "reasoning_content": "think "},
            "finish_reason": None,
        }]}),
        _data({"choices": [{"delta": {"reasoning_content": "done"},
                             "finish_reason": None}]}),
        _data({"choices": [{"delta": {"content": '{"analysis":"a",'},
                             "finish_reason": None}]}),
        _data({"choices": [{"delta": {"content": '"strategy_code":"code"}'},
                             "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}),
        "data: [DONE]\n\n",
    ]
    with _sse_server(events) as (base_url, captured):
        config = LLMConfig(base_url, "", "mock-model", stream=True)
        proposal = ChatCompletionsClient(config).propose_strategy([])

    assert captured["body"]["stream"] is True
    assert captured["body"]["stream_options"] == {"include_usage": True}
    assert "max_tokens" not in captured["body"]
    assert proposal.analysis == "a"
    assert proposal.strategy_code == "code"
    assert proposal.usage["total_tokens"] == 7
    assert proposal.raw_response["stream"] is True
    assert proposal.raw_response["chunk_count"] == 4
    assert proposal.raw_response["usage_missing"] is False
    assert proposal.raw_response["choices"][0]["message"]["reasoning_content"] == "think done"


def test_stream_malformed_chunk_preserves_partial_response():
    events = [
        _data({"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]}),
        "data: {bad-json}\n\n",
    ]
    with _sse_server(events) as (base_url, _):
        with pytest.raises(LLMRequestError) as raised:
            ChatCompletionsClient(LLMConfig(base_url, "", "mock", stream=True)).propose_strategy([])

    assert raised.value.stage == "stream_chunk_json"
    assert raised.value.raw_response["choices"][0]["message"]["content"] == "partial"


def test_stream_eof_before_done_is_explicitly_incomplete():
    events = [_data({
        "choices": [{"delta": {"content": "partial"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    })]
    with _sse_server(events) as (base_url, _):
        with pytest.raises(LLMRequestError) as raised:
            ChatCompletionsClient(LLMConfig(base_url, "", "mock", stream=True)).propose_strategy([])

    assert raised.value.stage == "stream_incomplete"
    assert raised.value.usage["total_tokens"] == 5
    assert raised.value.raw_response["choices"][0]["finish_reason"] == "length"
