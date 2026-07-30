"""
Auto-discovered multi-model config for the HL raw-API runner.

One gitignored ``.env`` describes every model: a ``MODELS=`` run-set (the keys
to run; the key is also the curve label and per-model round dir name) plus, per
key, ``<KEY>_PROVIDER`` (default ``openai``), ``<KEY>_BASE_URL`` (optional),
``<KEY>_MODEL`` (required), ``<KEY>_API_KEY`` (required). Adding a model is a
config edit only — no code change.

Honesty rules:
- An entry missing ``_MODEL`` or ``_API_KEY``, or with an unknown provider, is
  SKIPPED with a logged warning naming the key — it never aborts the batch.
- A key configured in ``.env`` but not listed in ``MODELS=`` is not loaded
  (the run-set is exactly ``MODELS=``).
- API key values are NEVER logged.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

log = logging.getLogger(__name__)

_KNOWN_PROVIDERS = ("openai", "anthropic")


@dataclass(frozen=True)
class ModelEntry:
    label: str
    provider: str            # "openai" | "anthropic"
    model: str
    api_key: str
    base_url: Optional[str] = None


def _default_env_path() -> Path:
    # src/agentbench_frame/hl/models_config.py -> AgentBenchFramework/.env
    return Path(__file__).resolve().parents[3] / ".env"


def load_models(path: Optional[Union[str, os.PathLike]] = None) -> Dict[str, ModelEntry]:
    """Load the model run-set from ``.env``.

    ``path`` overrides; else ``$MODELS_CONFIG``; else the framework-root
    ``.env``. Raises ``FileNotFoundError`` if the file is absent.
    """
    env_path = Path(path) if path else Path(
        os.environ.get("MODELS_CONFIG", _default_env_path()))
    if not env_path.exists():
        raise FileNotFoundError(f"models config not found: {env_path}")

    from dotenv import dotenv_values  # imported lazily (opt-in extra)
    cfg = dotenv_values(env_path)

    raw = cfg.get("MODELS", "") or ""
    keys = [k.strip() for k in raw.split(",") if k.strip()]

    out: Dict[str, ModelEntry] = {}
    for key in keys:
        key_upper = key.upper()
        provider = (cfg.get(f"{key_upper}_PROVIDER") or "openai").strip().lower()
        model = (cfg.get(f"{key_upper}_MODEL") or "").strip()
        api_key = (cfg.get(f"{key_upper}_API_KEY") or "").strip()
        base_url = (cfg.get(f"{key_upper}_BASE_URL") or "").strip() or None

        if not model or not api_key:
            log.warning("skipping model %r: missing _MODEL or _API_KEY", key)
            continue
        if provider not in _KNOWN_PROVIDERS:
            log.warning("skipping model %r: unknown provider %r", key, provider)
            continue
        out[key] = ModelEntry(label=key, provider=provider, model=model,
                              api_key=api_key, base_url=base_url)
        log.info("loaded model %r (provider=%s model=%s base_url=%s)",
                 key, provider, model, base_url)
    return out
