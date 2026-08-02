"""Programmatic Codex CLI sessions for HL acts."""

from __future__ import annotations

import hashlib
import dataclasses
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Mapping, Optional

from agentbench_frame.hl.config import ProviderConfig
from agentbench_frame.tracking.provider import ProviderInvocation
from agentbench_frame.tracking.providers import parse_codex_jsonl


_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_])(/[^\s'\"<>|;&]+)")
_PARENT_TRAVERSAL = re.compile(r"(^|[\s'\"=])\.\.(?:/|\s|$)")


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


class CodexSessionProvider:
    """Invoke official ``codex exec`` and resume its persisted session."""

    provider_name = "codex"

    def __init__(
        self,
        config: ProviderConfig,
        *,
        run_root: str | Path,
        environ: Optional[Mapping[str, str]] = None,
        timeout_s: Optional[float] = None,
        idle_timeout_s: Optional[float] = None,
    ) -> None:
        self.config = config
        self.run_root = Path(run_root)
        self.codex_home = self.run_root / "codex-home"
        self.codex_home.mkdir(parents=True, exist_ok=True)
        self.config_path = self.codex_home / "config.toml"
        self.environ = dict(os.environ if environ is None else environ)
        self.timeout_s = timeout_s
        self.idle_timeout_s = idle_timeout_s
        self._rate_limit_resume_at = 0.0
        self._write_config()

    def _write_config(self) -> None:
        lines = [
            f"model = {_toml_string(self.config.model)}" if self.config.model else "",
            f"review_model = {_toml_string(self.config.review_model)}"
            if self.config.review_model
            else "",
            f"model_reasoning_effort = {_toml_string(self.config.reasoning_effort)}",
            'sandbox_mode = "workspace-write"',
            f"disable_response_storage = {str(self.config.disable_response_storage).lower()}",
            "",
            "[sandbox_workspace_write]",
            f"network_access = {str(self.config.network_access == 'enabled').lower()}",
            "",
            "[shell_environment_policy]",
            'inherit = "core"',
            (
                "exclude = "
                + json.dumps(["CODEX_API_KEY", self.config.env_key], ensure_ascii=False)
            ),
        ]
        if self.config.base_url:
            lines[0:0] = ['model_provider = "agentbench_proxy"']
            lines.extend(
                [
                    "",
                    "[model_providers.agentbench_proxy]",
                    'name = "AgentBench Proxy"',
                    f"base_url = {_toml_string(self.config.base_url)}",
                    f"wire_api = {_toml_string(self.config.wire_api)}",
                    (
                        "requires_openai_auth = "
                        + str(self.config.requires_openai_auth).lower()
                    ),
                    f"# credential source: {self.config.env_key} -> CODEX_API_KEY",
                ]
            )
        budget = self.config.rollout_budget
        if budget.enabled:
            lines.extend(
                [
                    "",
                    "[features.rollout_budget]",
                    "enabled = true",
                    f"limit_tokens = {budget.limit_tokens}",
                    (
                        "reminder_at_remaining_tokens = "
                        + json.dumps(list(budget.reminder_at_remaining_tokens))
                    ),
                    f"sampling_token_weight = {budget.sampling_token_weight}",
                    f"prefill_token_weight = {budget.prefill_token_weight}",
                ]
            )
        self.config_path.write_text(
            "\n".join(line for line in lines if line is not None) + "\n",
            encoding="utf-8",
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.config_path.read_bytes()).hexdigest()

    def build_command(
        self,
        prompt: str,
        workspace: str | Path,
        *,
        session_id: Optional[str],
    ) -> list[str]:
        if not prompt:
            raise ValueError("prompt is required")
        executable = self.config.executable
        model_args = ["-m", self.config.model] if self.config.model else []
        if session_id and self.config.context_mode == "resumable":
            return [
                executable,
                "exec",
                "resume",
                "--json",
                *model_args,
                session_id,
                prompt,
            ]
        return [
            executable,
            "exec",
            "--json",
            "--sandbox",
            "workspace-write",
            "-C",
            str(workspace),
            *model_args,
            prompt,
        ]

    def build_environment(self) -> dict[str, str]:
        key = self.environ.get(self.config.env_key)
        if not key:
            raise RuntimeError(
                f"missing provider credential environment variable: {self.config.env_key}"
            )
        child = {
            name: value
            for name, value in self.environ.items()
            if name in {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SHELL"}
        }
        child["CODEX_HOME"] = str(self.codex_home)
        child["CODEX_API_KEY"] = key
        return child

    def redacted_environment(self) -> dict[str, str]:
        return {
            key: ("<redacted>" if key == "CODEX_API_KEY" else value)
            for key, value in self.build_environment().items()
        }

    def preflight(self) -> dict[str, object]:
        """Verify the frozen Codex runtime without issuing a model request."""

        environment = self.build_environment()
        version = subprocess.run(
            [self.config.executable, "--version"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if version.returncode != 0:
            raise RuntimeError(
                "Codex preflight could not read CLI version: "
                + (version.stderr.strip() or f"exit {version.returncode}")
            )
        actual_version = version.stdout.strip()
        expected_version = self.config.expected_cli_version
        if expected_version is not None and actual_version != expected_version:
            raise RuntimeError(
                "Codex CLI version mismatch: "
                f"expected {expected_version!r}, got {actual_version!r}"
            )

        budget_enabled = False
        if self.config.rollout_budget.enabled:
            features = subprocess.run(
                [self.config.executable, "features", "list"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if features.returncode != 0:
                raise RuntimeError(
                    "Codex preflight could not inspect features: "
                    + (features.stderr.strip() or f"exit {features.returncode}")
                )
            for line in features.stdout.splitlines():
                fields = line.split()
                if fields and fields[0] == "rollout_budget":
                    budget_enabled = fields[-1].lower() == "true"
                    break
            if not budget_enabled:
                raise RuntimeError(
                    "Codex preflight found rollout_budget disabled in generated config"
                )

        facts: dict[str, object] = {
            "schema_version": "1.0",
            "cli_version": actual_version,
            "expected_cli_version": expected_version,
            "provider_fingerprint": self.fingerprint,
            "rollout_budget_enabled": budget_enabled,
            "rollout_budget": json.loads(
                json.dumps(dataclasses.asdict(self.config.rollout_budget))
            ),
        }
        target = self.run_root / "provider-preflight.json"
        target.write_text(
            json.dumps(facts, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return facts

    def _annotate_budget(
        self,
        result: ProviderInvocation,
        *,
        raw_output: str,
        stderr: str,
    ) -> None:
        budget = self.config.rollout_budget
        if not budget.enabled:
            return
        diagnostic = " ".join(
            (str(result.error or ""), stderr, raw_output)
        ).lower()
        exhausted = any(
            marker in diagnostic
            for marker in (
                "sessionbudgetexceeded",
                "session budget exceeded",
                "rollout budget exceeded",
                "shared rollout token budget exhausted",
            )
        )
        result.metadata["rollout_budget_exhausted"] = exhausted
        result.metadata["rollout_budget_limit_tokens"] = budget.limit_tokens
        if exhausted:
            result.metadata["termination_reason"] = "rollout_budget_exhausted"
        usage = result.usage
        if usage.prompt_tokens is None or usage.completion_tokens is None:
            result.metadata["weighted_tokens"] = None
            return
        cached = usage.cached_input_tokens or 0
        non_cached = max(0, usage.prompt_tokens - cached)
        result.metadata["weighted_tokens"] = (
            non_cached * budget.prefill_token_weight
            + usage.completion_tokens * budget.sampling_token_weight
        )

    def invoke(
        self,
        *,
        prompt: str,
        workspace: str | Path,
        raw_output_path: str | Path,
        session_id: Optional[str] = None,
    ) -> ProviderInvocation:
        raw_path = Path(raw_output_path)
        retry_outputs: list[str] = []
        max_attempts = self.config.transport_retry_attempts + 1
        for attempt_index in range(max_attempts):
            self._respect_rate_limit_cooldown()
            result = self._invoke_once(
                prompt=prompt,
                workspace=workspace,
                raw_output_path=raw_path,
                session_id=session_id,
            )
            rate_limited = self._is_rate_limit_failure(result, raw_path)
            if rate_limited:
                result.metadata["rate_limited"] = True
                self._rate_limit_resume_at = max(
                    self._rate_limit_resume_at,
                    time.monotonic() + self.config.rate_limit_cooldown_seconds,
                )
            should_retry = (
                attempt_index + 1 < max_attempts
                and self._is_retryable_transport_failure(result, raw_path)
            )
            if not should_retry:
                result.metadata["transport_retry_count"] = len(retry_outputs)
                result.metadata["transport_retry_outputs"] = retry_outputs
                return result

            attempt_path = raw_path.with_name(
                f"{raw_path.stem}.attempt-{attempt_index + 1}{raw_path.suffix}"
            )
            attempt_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.replace(attempt_path)
            retry_outputs.append(str(attempt_path))
            backoff = (
                0.0
                if rate_limited
                else self.config.transport_retry_backoff_seconds
                * (2**attempt_index)
            )
            if backoff:
                time.sleep(backoff)
        raise AssertionError("provider retry loop must return")

    def _respect_rate_limit_cooldown(self) -> None:
        remaining = self._rate_limit_resume_at - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _is_rate_limit_failure(
        result: ProviderInvocation,
        raw_output_path: str | Path,
    ) -> bool:
        if result.status != "failed":
            return False
        raw_path = Path(raw_output_path)
        raw_output = (
            raw_path.read_text(encoding="utf-8", errors="replace")
            if raw_path.is_file()
            else ""
        )
        diagnostic = " ".join(
            (
                str(result.error or ""),
                str(result.metadata.get("stderr") or ""),
                raw_output,
            )
        ).lower()
        return "429 too many requests" in diagnostic

    def _invoke_once(
        self,
        *,
        prompt: str,
        workspace: str | Path,
        raw_output_path: str | Path,
        session_id: Optional[str] = None,
    ) -> ProviderInvocation:
        command = self.build_command(prompt, workspace, session_id=session_id)
        started = time.monotonic()
        try:
            completed = self._run_command(
                command=command,
                workspace=workspace,
                raw_output_path=raw_output_path,
            )
        except subprocess.TimeoutExpired as exc:
            partial = exc.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            raw_path = Path(raw_output_path)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(partial, encoding="utf-8")
            result = parse_codex_jsonl(partial)
            result.status = "timeout"
            result.elapsed_time_s = time.monotonic() - started
            timeout_kind = getattr(exc, "timeout_kind", "hard")
            if timeout_kind == "idle":
                result.error = (
                    "provider produced no stream progress for "
                    f"{self.idle_timeout_s}s"
                )
            else:
                result.error = f"provider timed out after {self.timeout_s}s"
            result.raw_output_ref = str(raw_path)
            violations = self._access_policy_violations(
                partial,
                workspace=workspace,
            )
            result.metadata.update(
                {
                    "command": command,
                    "provider_fingerprint": self.fingerprint,
                    "partial_output_persisted": True,
                    "timeout_kind": timeout_kind,
                    "access_policy_violations": violations,
                }
            )
            self._annotate_budget(
                result,
                raw_output=partial,
                stderr=str(exc.stderr or ""),
            )
            return result
        raw_path = Path(raw_output_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(completed.stdout or "", encoding="utf-8")
        result = parse_codex_jsonl(completed.stdout or "")
        violations = self._access_policy_violations(
            completed.stdout or "",
            workspace=workspace,
        )
        result.metadata["access_policy_violations"] = violations
        if violations:
            result.status = "failed"
            result.error = (
                "provider access policy violation: coding agent read outside "
                "the isolated candidate context"
            )
        result.elapsed_time_s = time.monotonic() - started
        result.raw_output_ref = str(raw_path)
        result.metadata.update(
            {
                "command": command,
                "return_code": completed.returncode,
                "stderr": completed.stderr or "",
                "provider_fingerprint": self.fingerprint,
                "resumed_session_id": session_id,
            }
        )
        if session_id and not result.metadata.get("thread_id"):
            result.metadata["thread_id"] = session_id
        if completed.returncode != 0:
            result.status = "failed"
            result.error = result.error or (
                completed.stderr.strip() or f"provider exited {completed.returncode}"
            )
        self._annotate_budget(
            result,
            raw_output=completed.stdout or "",
            stderr=completed.stderr or "",
        )
        return result

    def _run_command(
        self,
        *,
        command: list[str],
        workspace: str | Path,
        raw_output_path: str | Path,
    ) -> subprocess.CompletedProcess[str]:
        if self.idle_timeout_s is None:
            return subprocess.run(
                command,
                cwd=str(workspace),
                env=self.build_environment(),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )

        process = subprocess.Popen(
            command,
            cwd=str(workspace),
            env=self.build_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        raw_path = Path(raw_output_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        last_progress = started
        observed_sizes = (0, 0)
        partial_stdout = ""
        partial_stderr = ""

        def decoded(value: str | bytes | None) -> str:
            if value is None:
                return ""
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return value

        while True:
            now = time.monotonic()
            hard_remaining = (
                float("inf")
                if self.timeout_s is None
                else self.timeout_s - (now - started)
            )
            idle_remaining = self.idle_timeout_s - (now - last_progress)
            if hard_remaining <= 0 or idle_remaining <= 0:
                timeout_kind = "hard" if hard_remaining <= 0 else "idle"
                process.kill()
                stdout, stderr = process.communicate()
                partial_stdout = decoded(stdout) or partial_stdout
                partial_stderr = decoded(stderr) or partial_stderr
                raw_path.write_text(partial_stdout, encoding="utf-8")
                error = subprocess.TimeoutExpired(
                    cmd=command,
                    timeout=(
                        self.timeout_s
                        if timeout_kind == "hard"
                        else self.idle_timeout_s
                    ),
                    output=partial_stdout,
                    stderr=partial_stderr,
                )
                error.timeout_kind = timeout_kind
                raise error
            try:
                stdout, stderr = process.communicate(
                    timeout=max(0.01, min(0.1, hard_remaining, idle_remaining))
                )
            except subprocess.TimeoutExpired as exc:
                stdout = decoded(exc.stdout)
                stderr = decoded(exc.stderr)
                sizes = (len(stdout), len(stderr))
                if sizes != observed_sizes:
                    observed_sizes = sizes
                    last_progress = time.monotonic()
                    partial_stdout = stdout
                    partial_stderr = stderr
                    raw_path.write_text(partial_stdout, encoding="utf-8")
                continue
            return subprocess.CompletedProcess(
                command,
                process.returncode,
                decoded(stdout),
                decoded(stderr),
            )

    @staticmethod
    def _is_retryable_transport_failure(
        result: ProviderInvocation,
        raw_output_path: str | Path,
    ) -> bool:
        """Retry only failures that cannot have performed or billed useful work."""

        if result.status != "failed":
            return False
        if result.usage.total_tokens is not None or result.tool_call_count:
            return False
        raw_path = Path(raw_output_path)
        raw_output = (
            raw_path.read_text(encoding="utf-8", errors="replace")
            if raw_path.is_file()
            else ""
        )
        diagnostic = " ".join(
            (
                str(result.error or ""),
                str(result.metadata.get("stderr") or ""),
                raw_output,
            )
        ).lower()
        transport_markers = (
            "stream disconnected",
            "error sending request",
            "connection reset",
            "connection closed",
            "could not resolve host",
            "dns error",
            "temporarily unavailable",
            "http status 502",
            "http status 503",
            "http status 504",
            "429 too many requests",
        )
        return any(marker in diagnostic for marker in transport_markers)

    def recover_completed_output(
        self,
        *,
        raw_output_path: str | Path,
        workspace: str | Path,
    ) -> ProviderInvocation:
        """Revalidate a persisted successful Codex stream without another API call."""

        raw_path = Path(raw_output_path)
        raw_output = raw_path.read_text(encoding="utf-8")
        result = parse_codex_jsonl(raw_output)
        violations = self._access_policy_violations(
            raw_output,
            workspace=workspace,
        )
        result.metadata["access_policy_violations"] = violations
        result.metadata["provider_fingerprint"] = self.fingerprint
        result.metadata["recovered_from_persisted_output"] = True
        result.raw_output_ref = str(raw_path)
        if violations:
            result.status = "failed"
            result.error = (
                "provider access policy violation: coding agent read outside "
                "the isolated candidate context"
            )
        return result

    def _access_policy_violations(
        self,
        raw_output: str,
        *,
        workspace: str | Path,
    ) -> list[str]:
        allowed_roots = tuple(
            path.resolve()
            for path in (
                Path(workspace),
                self.run_root / "context",
                self.run_root / "experience",
                self.run_root / "matches",
                self.run_root / "measurement",
                self.run_root / "proposals",
                self.run_root / "distillation",
                self.run_root / "research_state.json",
            )
        )
        home = Path(self.environ.get("HOME", str(Path.home()))).resolve()
        commands: set[str] = set()
        for line in raw_output.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = record.get("item")
            if (
                isinstance(item, Mapping)
                and item.get("type") == "command_execution"
                and isinstance(item.get("command"), str)
            ):
                commands.add(str(item["command"]))

        violations: set[str] = set()
        for command in commands:
            if _PARENT_TRAVERSAL.search(command):
                violations.add("<relative parent traversal>")
            for raw_path in _ABSOLUTE_PATH.findall(command):
                candidate = Path(raw_path.rstrip(",)]}")).resolve()
                try:
                    candidate.relative_to(home)
                except ValueError:
                    continue
                if any(
                    candidate == root or candidate.is_relative_to(root)
                    for root in allowed_roots
                ):
                    continue
                violations.add(str(candidate))
        return sorted(violations)
