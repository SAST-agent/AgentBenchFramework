"""Explicitly gated checks against preserved submissions and a real Codex CLI."""

from pathlib import Path
import os
import shutil
import sys

import pytest

from agentbench_frame.generals.assets import (
    load_pilot_config,
    prepare_opponents,
    resolve_assets,
)
from agentbench_frame.generals.engine import OfficialGeneralsEngine
from agentbench_frame.generals.pipeline import GeneralsHLPipeline
from agentbench_frame.generals.process import ManagedAgentProcess
from agentbench_frame.tracking.providers import CodexProvider


LIVE = os.environ.get("AGENTBENCH_RUN_LIVE_GENERALS") == "1"
pytestmark = pytest.mark.skipif(
    not LIVE, reason="set AGENTBENCH_RUN_LIVE_GENERALS=1"
)


@pytest.fixture
def real_assets():
    root_value = os.environ.get("AGENTBENCH_ASSET_ROOT")
    if not root_value:
        pytest.skip("AGENTBENCH_ASSET_ROOT is not set")
    root = Path(root_value)
    manifest = root / "backend_sources/corpus/28_generals/benchmark/pilot-v1.toml"
    config = load_pilot_config(manifest)
    return config, resolve_assets(config, root), manifest


def test_all_frozen_opponents_start_and_complete_one_decision(real_assets, tmp_path):
    config, layout, _ = real_assets
    prepared = prepare_opponents(layout, tmp_path / "prepared", Path(sys.executable))
    for opponent in prepared:
        engine = OfficialGeneralsEngine(layout.engine_root, seed=280101)
        with ManagedAgentProcess(
            opponent, config.limits, tmp_path / "artifacts" / opponent.agent_id
        ) as player:
            player.send_initial(engine.initial_observation(0))
            commands = player.request_turn()
        assert commands[-1] == (8,), opponent.agent_id


def test_real_codex_one_act_produces_finalized_run(real_assets, tmp_path):
    _, layout, manifest = real_assets
    pipeline = GeneralsHLPipeline.from_paths(
        agentbench_root=layout.root,
        manifest_path=manifest,
        data_dir=tmp_path / "data",
        provider=CodexProvider(
            executable=shutil.which("codex") or "codex", timeout_s=1800
        ),
    )
    result = pipeline.run()
    assert result.act_count == 1
    assert (result.run_dir / "provider" / "codex.raw.jsonl").stat().st_size > 0
    assert (result.run_dir / "quality.json").exists()
