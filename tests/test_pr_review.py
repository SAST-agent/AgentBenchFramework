from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tools.pr_review import (
    build_api_request,
    extract_model_text,
    parse_review_document,
    review_should_fail,
)


@contextlib.contextmanager
def serve(handler_class):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/review"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _input_file(tmp_path: Path) -> Path:
    path = tmp_path / "review-input.json"
    path.write_text(json.dumps({"diff": "diff --git a/a.py b/a.py"}), encoding="utf-8")
    return path


def _cli_env(tmp_path: Path, endpoint: str, mode: str = "responses") -> dict[str, str]:
    return {
        **os.environ,
        "PR_REVIEW_API_KEY": "secret-value",
        "PR_REVIEW_ENDPOINT": endpoint,
        "PR_REVIEW_MODEL": "review-model",
        "PR_REVIEW_API_MODE": mode,
        "PR_REVIEW_RETRY_BACKOFF_S": "0",
        "PR_REVIEW_INPUT_JSON": str(_input_file(tmp_path)),
    }


def _run_cli(env: dict[str, str]):
    return subprocess.run(
        [sys.executable, "tools/pr_review.py"],
        env=env,
        text=True,
        capture_output=True,
    )


def test_extracts_responses_output_text_and_validates_review():
    response = {
        "output": [{
            "type": "message",
            "content": [{
                "type": "output_text",
                "text": '{"decision":"fail","summary":"bad","findings":[]}',
            }],
        }],
    }

    review = parse_review_document(extract_model_text("responses", response))

    assert review["decision"] == "fail"
    assert review_should_fail(review) is True


def test_extracts_chat_completion_content():
    response = {"choices": [{"message": {
        "content": '{"decision":"pass","summary":"ok","findings":[]}',
    }}]}

    review = parse_review_document(extract_model_text("chat_completions", response))

    assert review == {"decision": "pass", "summary": "ok", "findings": []}


def test_extracts_fenced_json_from_responses():
    response = {"output_text": "```json\n{\"decision\":\"pass\",\"summary\":\"ok\",\"findings\":[]}\n```"}

    assert extract_model_text("responses", response).startswith("```")
    assert parse_review_document(extract_model_text("responses", response))["decision"] == "pass"


def test_rejects_unknown_severity_and_malformed_json():
    with pytest.raises(ValueError, match="severity"):
        parse_review_document(
            '{"decision":"pass","summary":"x","findings":[{"severity":"P9","message":"x"}]}'
        )
    with pytest.raises(ValueError, match="JSON"):
        parse_review_document("not json")


def test_rejects_missing_required_finding_message():
    with pytest.raises(ValueError, match="message"):
        parse_review_document(
            '{"decision":"pass","summary":"x","findings":[{"severity":"P1"}]}'
        )


def test_p1_finding_fails_even_when_decision_is_pass():
    review = parse_review_document(json.dumps({
        "decision": "pass",
        "summary": "reviewed",
        "findings": [{"severity": "P1", "message": "blocking issue"}],
    }))

    assert review_should_fail(review) is True


def test_p2_finding_does_not_fail_when_decision_is_pass():
    review = parse_review_document(json.dumps({
        "decision": "pass",
        "summary": "reviewed",
        "findings": [{"severity": "P2", "message": "follow-up"}],
    }))

    assert review_should_fail(review) is False


def test_builds_protocol_specific_json_requests():
    responses = build_api_request("responses", "review-model", "system", "diff")
    chat = build_api_request("chat_completions", "review-model", "system", "diff")

    assert responses["model"] == "review-model"
    assert responses["input"] == "diff"
    responses_format = responses["text"]["format"]
    assert responses_format["type"] == "json_schema"
    assert responses_format["name"] == "pr_review"
    assert responses_format["strict"] is True
    message_schema = responses_format["schema"]["properties"]["findings"]["items"]["properties"]["message"]
    assert message_schema["type"] == "string"
    assert "non-empty" in message_schema["description"]
    assert responses["reasoning"] == {"effort": "high"}
    assert chat["model"] == "review-model"
    assert chat["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "diff"},
    ]
    chat_format = chat["response_format"]
    assert chat_format["type"] == "json_schema"
    assert chat_format["json_schema"]["name"] == "pr_review"
    assert chat_format["json_schema"]["strict"] is True
    assert chat_format["json_schema"]["schema"] == responses_format["schema"]
    assert chat["reasoning_effort"] == "high"


def test_rejects_unknown_api_mode():
    with pytest.raises(ValueError, match="api mode"):
        build_api_request("legacy", "m", "system", "payload")


def test_cli_sends_auth_and_passes_without_printing_secret(tmp_path):
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen["authorization"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = {"output_text": json.dumps({
                "decision": "pass", "summary": "ok", "findings": [],
            })}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *_args):
            return

    with serve(Handler) as endpoint:
        result = _run_cli(_cli_env(tmp_path, endpoint))

    assert result.returncode == 0
    assert seen["authorization"] == "Bearer secret-value"
    assert seen["body"]["model"] == "review-model"
    assert "secret-value" not in result.stdout + result.stderr


def test_cli_supports_chat_completions_mode(tmp_path):
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = {"choices": [{"message": {"content": json.dumps({
                "decision": "pass", "summary": "ok", "findings": [],
            })}}]}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *_args):
            return

    with serve(Handler) as endpoint:
        result = _run_cli(_cli_env(tmp_path, endpoint, mode="chat_completions"))

    assert result.returncode == 0
    assert "messages" in seen["body"]


def test_cli_falls_back_to_legacy_json_when_structured_output_is_rejected(tmp_path):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if len(seen) == 1:
                self.send_response(400)
                self.end_headers()
                return
            body = {"output_text": json.dumps({
                "decision": "pass", "summary": "ok", "findings": [],
            })}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *_args):
            return

    with serve(Handler) as endpoint:
        result = _run_cli(_cli_env(tmp_path, endpoint))

    assert result.returncode == 0
    assert len(seen) == 2
    assert seen[0]["text"]["format"]["type"] == "json_schema"
    assert seen[0]["reasoning"] == {"effort": "high"}
    assert seen[1]["text"]["format"] == {"type": "json_object"}
    assert "reasoning" not in seen[1]


def test_cli_falls_back_after_same_request_gateway_retries(tmp_path):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if len(seen) <= 3:
                self.send_response(504)
                self.end_headers()
                return
            body = {"output_text": json.dumps({
                "decision": "pass", "summary": "ok", "findings": [],
            })}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *_args):
            return

    with serve(Handler) as endpoint:
        result = _run_cli(_cli_env(tmp_path, endpoint))

    assert result.returncode == 0
    assert len(seen) == 4
    assert seen[0]["text"]["format"]["type"] == "json_schema"
    assert seen[0]["reasoning"] == {"effort": "high"}
    assert seen[1] == seen[0]
    assert seen[2] == seen[0]
    assert seen[3]["text"]["format"]["type"] == "json_schema"
    assert "reasoning" not in seen[3]


def test_cli_retries_same_request_after_gateway_timeout(tmp_path):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if len(seen) == 1:
                self.send_response(504)
                self.end_headers()
                return
            body = {"output_text": json.dumps({
                "decision": "pass", "summary": "ok", "findings": [],
            })}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *_args):
            return

    with serve(Handler) as endpoint:
        result = _run_cli(_cli_env(tmp_path, endpoint))

    assert result.returncode == 0
    assert len(seen) == 2
    assert seen[0] == seen[1]


def test_cli_does_not_retry_non_gateway_http_error(tmp_path):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(401)
            self.end_headers()

        def log_message(self, *_args):
            return

    with serve(Handler) as endpoint:
        result = _run_cli(_cli_env(tmp_path, endpoint))

    assert result.returncode != 0
    assert len(seen) == 1


@pytest.mark.parametrize("env_name", [
    "PR_REVIEW_API_KEY", "PR_REVIEW_ENDPOINT", "PR_REVIEW_MODEL",
    "PR_REVIEW_INPUT_JSON",
])
def test_cli_fails_when_required_configuration_is_missing(tmp_path, env_name):
    env = _cli_env(tmp_path, "http://127.0.0.1:1/review")
    env.pop(env_name)

    result = _run_cli(env)

    assert result.returncode != 0
    assert "secret-value" not in result.stdout + result.stderr


def test_cli_fails_closed_on_http_error(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.send_response(500)
            self.end_headers()

        def log_message(self, *_args):
            return

    with serve(Handler) as endpoint:
        result = _run_cli(_cli_env(tmp_path, endpoint))

    assert result.returncode != 0


@pytest.mark.parametrize("review_body", [
    {"output_text": "not json"},
    {"output_text": json.dumps({
        "decision": "pass", "summary": "bad", "findings": [
            {"severity": "P1", "message": "blocking"},
        ],
    })},
])
def test_cli_fails_closed_on_invalid_or_blocking_review(tmp_path, review_body):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(review_body).encode())

        def log_message(self, *_args):
            return

    with serve(Handler) as endpoint:
        result = _run_cli(_cli_env(tmp_path, endpoint))

    assert result.returncode != 0


def test_setup_document_names_workflow_configuration():
    document = Path("docs/integration/pr-review-check.md").read_text(encoding="utf-8")

    for name in (
        "PR_REVIEW_API_KEY", "PR_REVIEW_ENDPOINT", "PR_REVIEW_MODEL",
        "PR_REVIEW_API_MODE",
    ):
        assert name in document
    assert "framework-tests" in document
    assert "ai-pr-review" in document
