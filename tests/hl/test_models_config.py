from __future__ import annotations
import logging
from pathlib import Path
import textwrap
import pytest

from agentbench_frame.hl.models_config import ModelEntry, load_models


def _write_env(tmp_path: Path, body: str) -> Path:
    p = tmp_path / ".env"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_load_models_basic(tmp_path):
    env = _write_env(tmp_path, """\
        MODELS=glm,deepseek
        GLM_PROVIDER=openai
        GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
        GLM_MODEL=glm-4.6
        GLM_API_KEY=sk-glm
        DEEPSEEK_MODEL=deepseek-chat
        DEEPSEEK_API_KEY=sk-ds
    """)
    models = load_models(env)
    assert set(models) == {"glm", "deepseek"}
    assert models["glm"] == ModelEntry(
        label="glm", provider="openai", model="glm-4.6",
        api_key="sk-glm", base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    # provider defaults to openai; base_url defaults to None
    assert models["deepseek"].provider == "openai"
    assert models["deepseek"].base_url is None


def test_anthropic_provider(tmp_path):
    env = _write_env(tmp_path, """\
        MODELS=claude
        CLAUDE_PROVIDER=anthropic
        CLAUDE_MODEL=claude-sonnet-5
        CLAUDE_API_KEY=sk-ant
    """)
    models = load_models(env)
    assert models["claude"].provider == "anthropic"
    assert models["claude"].base_url is None


def test_run_set_is_exactly_MODELS(tmp_path):
    # key configured but not in MODELS -> not loaded
    env = _write_env(tmp_path, """\
        MODELS=glm
        GLM_MODEL=glm-4.6
        GLM_API_KEY=sk-glm
        EXTRA_MODEL=x
        EXTRA_API_KEY=sk-x
    """)
    assert set(load_models(env)) == {"glm"}


def test_missing_field_skipped_not_fatal(tmp_path, caplog):
    env = _write_env(tmp_path, """\
        MODELS=glm,broken,deepseek
        GLM_MODEL=glm-4.6
        GLM_API_KEY=sk-glm
        BROKEN_API_KEY=sk-broken
        DEEPSEEK_MODEL=deepseek-chat
        DEEPSEEK_API_KEY=sk-ds
    """)
    with caplog.at_level(logging.WARNING, logger="agentbench_frame.hl.models_config"):
        models = load_models(env)
    assert set(models) == {"glm", "deepseek"}          # broken skipped
    assert any("broken" in r.message for r in caplog.records)


def test_unknown_provider_skipped(tmp_path, caplog):
    env = _write_env(tmp_path, """\
        MODELS=weird
        WEIRD_PROVIDER=madeup
        WEIRD_MODEL=m
        WEIRD_API_KEY=k
    """)
    with caplog.at_level(logging.WARNING, logger="agentbench_frame.hl.models_config"):
        models = load_models(env)
    assert models == {}
    assert any("weird" in r.message for r in caplog.records)


def test_never_logs_api_key(tmp_path, caplog):
    env = _write_env(tmp_path, """\
        MODELS=glm
        GLM_MODEL=glm-4.6
        GLM_API_KEY=SECRET-VALUE-12345
    """)
    with caplog.at_level(logging.DEBUG, logger="agentbench_frame.hl.models_config"):
        load_models(env)
    dumped = "\n".join(r.message for r in caplog.records)
    assert "SECRET-VALUE-12345" not in dumped


def test_empty_MODELS(tmp_path):
    env = _write_env(tmp_path, "MODELS=\n")
    assert load_models(env) == {}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_models(tmp_path / "nope.env")
