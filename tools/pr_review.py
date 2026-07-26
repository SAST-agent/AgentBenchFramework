"""Protocol and result validation helpers for the blocking PR review check.

The module deliberately uses only the Python standard library.  The GitHub
workflow runs it from a trusted base checkout, while PR-derived data is passed
as JSON data rather than interpolated into shell source.
"""
from __future__ import annotations

import json
import re
from typing import Any


API_MODES = frozenset({"responses", "chat_completions"})
SEVERITIES = frozenset({"P0", "P1", "P2", "P3"})
_FENCED_JSON = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _text_from_parts(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return None
    chunks = []
    for part in value:
        if isinstance(part, str):
            chunks.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            chunks.append(part["text"])
    return "".join(chunks) if chunks else None


def extract_model_text(api_mode: str, response: dict[str, Any]) -> str:
    """Extract the assistant's text from either supported API response shape."""
    if not isinstance(response, dict):
        raise ValueError("model response must be a JSON object")
    if api_mode == "responses":
        direct = response.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct
        output = response.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                text = _text_from_parts(item.get("content"))
                if text and text.strip():
                    return text
    elif api_mode == "chat_completions":
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    text = _text_from_parts(message.get("content"))
                    if text and text.strip():
                        return text
    else:
        raise ValueError(f"unsupported api mode: {api_mode}")
    raise ValueError(f"no assistant text found in {api_mode} response")


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = _FENCED_JSON.match(stripped)
    return match.group(1).strip() if match else stripped


def _validate_optional_finding_field(finding: dict[str, Any], key: str) -> Any:
    value = finding.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"finding {key} must be a string when present")
    return value


def parse_review_document(text: str) -> dict[str, Any]:
    """Parse and normalize the strict review response contract."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("review response text is empty")
    try:
        document = json.loads(_strip_json_fence(text))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"review response is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("review response must be a JSON object")

    decision = document.get("decision")
    summary = document.get("summary")
    findings = document.get("findings")
    if decision not in {"pass", "fail"}:
        raise ValueError("decision must be pass or fail")
    if not isinstance(summary, str):
        raise ValueError("summary must be a string")
    if not isinstance(findings, list):
        raise ValueError("findings must be a list")

    normalized_findings = []
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise ValueError(f"finding {index} must be an object")
        severity = finding.get("severity")
        message = finding.get("message")
        if severity not in SEVERITIES:
            raise ValueError(f"finding {index} severity must be one of P0, P1, P2, P3")
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"finding {index} message must be a non-empty string")
        normalized = {"severity": severity, "message": message}
        for key in ("path", "suggestion"):
            value = _validate_optional_finding_field(finding, key)
            if value is not None:
                normalized[key] = value
        line = finding.get("line")
        if line is not None and (isinstance(line, bool) or not isinstance(line, int) or line < 1):
            raise ValueError(f"finding {index} line must be a positive integer when present")
        if line is not None:
            normalized["line"] = line
        normalized_findings.append(normalized)
    return {"decision": decision, "summary": summary, "findings": normalized_findings}


def review_should_fail(review: dict[str, Any]) -> bool:
    """Return whether a normalized review must fail the required check."""
    return review["decision"] == "fail" or any(
        finding["severity"] in {"P0", "P1"}
        for finding in review["findings"]
    )


def build_api_request(api_mode: str, model: str, instructions: str, payload: str) -> dict[str, Any]:
    """Build a JSON request body for the selected OpenAI-compatible API."""
    if api_mode == "responses":
        return {
            "model": model,
            "instructions": instructions,
            "input": payload,
            "text": {"format": {"type": "json_object"}},
        }
    if api_mode == "chat_completions":
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": payload},
            ],
            "response_format": {"type": "json_object"},
        }
    raise ValueError(f"unsupported api mode: {api_mode}")
