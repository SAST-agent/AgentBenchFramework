"""Protocol and result validation helpers for the blocking PR review check.

The module deliberately uses only the Python standard library.  The GitHub
workflow runs it from a trusted base checkout, while PR-derived data is passed
as JSON data rather than interpolated into shell source.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any


API_MODES = frozenset({"responses", "chat_completions"})
SEVERITIES = frozenset({"P0", "P1", "P2", "P3"})
_FENCED_JSON = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
DEFAULT_API_MODE = "responses"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_TIMEOUT_S = 90.0
MAX_INPUT_BYTES = 400_000
REVIEW_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["pass", "fail"],
        },
        "summary": {
            "type": "string",
            "description": "A concise non-empty review summary.",
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["P0", "P1", "P2", "P3"],
                    },
                    "path": {"type": ["string", "null"]},
                    "line": {"type": ["integer", "null"]},
                    "message": {
                        "type": "string",
                        "description": "A concise non-empty actionable finding message.",
                    },
                    "suggestion": {"type": ["string", "null"]},
                },
                "required": ["severity", "path", "line", "message", "suggestion"],
            },
        },
    },
    "required": ["decision", "summary", "findings"],
}
REVIEW_INSTRUCTIONS = """You are the blocking code reviewer for AgentBenchFramework.
Review the supplied pull request diff for correctness, scientific data integrity,
reproducibility, process and secret safety, and forward compatibility. Only
report actionable findings. P0/P1 findings block merging. The response must
conform to the supplied structured review schema. If there are no actionable
findings, set findings to []. Never emit a placeholder finding: every finding
message must contain a concrete, non-empty explanation. Return ONLY the JSON
object required by the review response schema; do not use Markdown fences.
"""


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


def build_api_request(
    api_mode: str,
    model: str,
    instructions: str,
    payload: str,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    structured_outputs: bool = True,
) -> dict[str, Any]:
    """Build a JSON request body for the selected OpenAI-compatible API."""
    if api_mode == "responses":
        request = {
            "model": model,
            "instructions": instructions,
            "input": payload,
            "text": {
                "format": (
                    {
                        "type": "json_schema",
                        "name": "pr_review",
                        "strict": True,
                        "schema": REVIEW_RESPONSE_SCHEMA,
                    }
                    if structured_outputs
                    else {"type": "json_object"}
                ),
            },
        }
        if reasoning_effort is not None:
            request["reasoning"] = {"effort": reasoning_effort}
        return request
    if api_mode == "chat_completions":
        request = {
            "model": model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": payload},
            ],
            "response_format": (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "pr_review",
                        "strict": True,
                        "schema": REVIEW_RESPONSE_SCHEMA,
                    },
                }
                if structured_outputs
                else {"type": "json_object"}
            ),
        }
        if reasoning_effort is not None:
            request["reasoning_effort"] = reasoning_effort
        return request
    raise ValueError(f"unsupported api mode: {api_mode}")


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"missing required environment variable: {name}")
    return value


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "[REDACTED]") if secret else text


def _escape_annotation(text: str) -> str:
    return (str(text).replace("%", "%25").replace("\r", "%0D")
            .replace("\n", "%0A").replace(":", "%3A").replace(",", "%2C"))


def _write_summary(review: dict[str, Any] | None, error: str | None, secret: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    if error is not None:
        content = f"## Automated PR review\n\n**FAIL:** {_redact(error, secret)}\n"
    else:
        findings = review["findings"] if review else []
        lines = ["## Automated PR review", "", f"**Decision:** `{review['decision']}`", "",
                 _redact(review["summary"], secret), "", "### Findings"]
        if findings:
            lines.extend(
                f"- `{f['severity']}` { _redact(f['message'], secret) }"
                for f in findings
            )
        else:
            lines.append("- None")
        content = "\n".join(lines) + "\n"
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write(content)


def _post_review_request(
    endpoint: str,
    api_key: str,
    request_body: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _emit_review(review: dict[str, Any], secret: str) -> None:
    print(f"PR review decision={review['decision']} findings={len(review['findings'])}")
    print(_redact(review["summary"], secret))
    for finding in review["findings"]:
        message = finding["message"]
        suggestion = finding.get("suggestion")
        if suggestion:
            message = f"{message} Suggestion: {suggestion}"
        escaped = _escape_annotation(_redact(message, secret))
        path = finding.get("path")
        line = finding.get("line")
        if isinstance(path, str) and path and isinstance(line, int):
            print(f"::{ 'error' if finding['severity'] in {'P0', 'P1'} else 'warning' } "
                  f"file={_escape_annotation(path)},line={line}::{escaped}")
        else:
            print(f"{finding['severity']}: {escaped}")


def run_review() -> int:
    """Read workflow inputs, call the configured endpoint, and return a check exit code."""
    secret = os.environ.get("PR_REVIEW_API_KEY", "")
    try:
        api_key = _required_env("PR_REVIEW_API_KEY")
        endpoint = _required_env("PR_REVIEW_ENDPOINT")
        model = _required_env("PR_REVIEW_MODEL")
        input_path = _required_env("PR_REVIEW_INPUT_JSON")
        api_mode = os.environ.get("PR_REVIEW_API_MODE", DEFAULT_API_MODE).strip().lower()
        with open(input_path, "r", encoding="utf-8") as handle:
            input_data = json.load(handle)
        payload = json.dumps(input_data, ensure_ascii=False, separators=(",", ":"))
        payload_bytes = payload.encode("utf-8")
        if len(payload_bytes) > MAX_INPUT_BYTES:
            raise ValueError(f"review input exceeds {MAX_INPUT_BYTES} bytes")
        timeout = float(os.environ.get("PR_REVIEW_TIMEOUT_S", DEFAULT_TIMEOUT_S))
        request_body = build_api_request(api_mode, model, REVIEW_INSTRUCTIONS, payload)
        try:
            response_data = _post_review_request(endpoint, api_key, request_body, timeout)
        except urllib.error.HTTPError as exc:
            if exc.code not in {400, 422}:
                raise
            legacy_request = build_api_request(
                api_mode,
                model,
                REVIEW_INSTRUCTIONS,
                payload,
                reasoning_effort=None,
                structured_outputs=False,
            )
            response_data = _post_review_request(endpoint, api_key, legacy_request, timeout)
        if isinstance(response_data, dict) and {"decision", "summary", "findings"} <= response_data.keys():
            review = parse_review_document(json.dumps(response_data, ensure_ascii=False))
        else:
            review = parse_review_document(extract_model_text(api_mode, response_data))
        _write_summary(review, None, api_key)
        _emit_review(review, api_key)
        return 1 if review_should_fail(review) else 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError, urllib.error.URLError) as exc:
        message = _redact(str(exc), secret)
        print(f"Automated PR review failed closed: {message}", file=sys.stderr)
        try:
            _write_summary(None, message, secret)
        except OSError:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(run_review())
