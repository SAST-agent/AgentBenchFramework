import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentbench_frame.doto.llm_client import ChatCompletionsClient, LLMRequestError
from agentbench_frame.doto.loop_config import LLMConfig


class Handler(BaseHTTPRequestHandler):
    response = None
    request_body = None
    request_headers = None

    def do_POST(self):
        type(self).request_body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).request_headers = self.headers
        status, content_type, payload = type(self).response
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        thread.join()


def config(base_url, *, stream=True):
    return LLMConfig(base_url, "DOTO_TEST_KEY", "mock-model", stream=stream)


def test_stream_accumulates_reasoning_content_usage_and_done(server, monkeypatch):
    monkeypatch.setenv("DOTO_TEST_KEY", "top-secret")
    proposal = json.dumps({"analysis": "improve", "player_ai_cpp": "void playerAI() {}"})
    chunks = [
        ': heartbeat\n\n',
        'data: {"choices":[{"delta":{"reasoning_content":"think "}}]}\n\n',
        f'data: {json.dumps({"choices": [{"delta": {"content": proposal[:20]}}]})}\n\n',
        f'data: {json.dumps({"choices": [{"delta": {"content": proposal[20:]}, "finish_reason": "stop"}]})}\n\n',
        'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}\n\n',
        'data: [DONE]\n\n',
    ]
    Handler.response = (200, "text/event-stream", "".join(chunks).encode())
    result = ChatCompletionsClient(config(server)).propose([{"role": "user", "content": "x"}])
    assert result.analysis == "improve"
    assert result.player_ai_cpp.startswith("void")
    assert result.usage["total_tokens"] == 7
    assert result.raw_response["choices"][0]["message"]["reasoning_content"] == "think "
    assert Handler.request_body["stream_options"] == {"include_usage": True}
    assert "max_tokens" not in Handler.request_body
    assert Handler.request_headers["User-Agent"].startswith("AgentBenchFramework/")


def test_nonstream_json_and_markdown_fence(server):
    content = '```json\n{"analysis":"a","player_ai_cpp":"code"}\n```'
    payload = {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 2}}
    Handler.response = (200, "application/json", json.dumps(payload).encode())
    result = ChatCompletionsClient(config(server, stream=False)).propose([])
    assert result.player_ai_cpp == "code"
    assert result.normalized_fence is True


@pytest.mark.parametrize("payload,stage", [
    (b'data: {bad}\n\n', "stream_chunk_json"),
    (b'data: {"choices":[]}\n\n', "stream_incomplete"),
])
def test_stream_errors_are_explicit_and_do_not_leak_key(server, monkeypatch, payload, stage):
    monkeypatch.setenv("DOTO_TEST_KEY", "never-print-this")
    Handler.response = (200, "text/event-stream", payload)
    with pytest.raises(LLMRequestError) as caught:
        ChatCompletionsClient(config(server)).propose([])
    assert caught.value.stage == stage
    assert "never-print-this" not in str(caught.value)
    assert "never-print-this" not in json.dumps(caught.value.raw_response)


def test_http_error_body_cannot_leak_key(server, monkeypatch):
    monkeypatch.setenv("DOTO_TEST_KEY", "never-print-this")
    Handler.response = (401, "application/json", b'{"echo":"never-print-this"}')
    with pytest.raises(LLMRequestError) as caught:
        ChatCompletionsClient(config(server)).propose([])
    assert caught.value.stage == "request"
    assert "never-print-this" not in str(caught.value)
