from __future__ import annotations

import json

import pytest

from tools.pr_review import (
    build_api_request,
    extract_model_text,
    parse_review_document,
    review_should_fail,
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
    assert responses["text"]["format"] == {"type": "json_object"}
    assert chat["model"] == "review-model"
    assert chat["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "diff"},
    ]
    assert chat["response_format"] == {"type": "json_object"}


def test_rejects_unknown_api_mode():
    with pytest.raises(ValueError, match="api mode"):
        build_api_request("legacy", "m", "system", "payload")
