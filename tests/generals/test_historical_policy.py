from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from agentbench_frame.generals.engine import OfficialGeneralsEngine
from agentbench_frame.generals.models import HistoricalPolicyConfig
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


@pytest.fixture
def engine_root():
    value = os.environ.get("AGENTBENCH_ASSET_ROOT")
    if not value:
        pytest.skip("AGENTBENCH_ASSET_ROOT is not set")
    return (
        Path(value)
        / "backend_sources/corpus/28_generals/logic/gamecode_logic"
    )


def make_history(data_dir, version, run_id, source_files):
    version_root = (
        data_dir
        / "runs/28_generals/generals-hl"
        / run_id
        / "versions"
        / version
    )
    source = version_root / "source"
    source.mkdir(parents=True)
    for relative, text in source_files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    snapshotter = LocalWorkspaceSnapshotter()
    manifest = snapshotter.capture(source)
    snapshotter.write_manifest(manifest, version_root / "manifest.json")
    return HistoricalPolicyConfig(
        version=version,
        run_id=run_id,
        content_hash=manifest.content_hash,
    )


def test_history_resolver_requires_exact_manifest_hash(tmp_path):
    from agentbench_frame.generals.historical_policy import (
        HistoricalPolicyError,
        resolve_historical_policies,
    )

    config = make_history(
        tmp_path,
        "v0",
        "run-0",
        {"main.py": "def agent(round_number, my_seat, state): return []\n"},
    )
    bad = replace(config, content_hash="0" * 64)

    with pytest.raises(HistoricalPolicyError, match="content hash"):
        resolve_historical_policies(
            (bad,),
            tmp_path,
            tmp_path / "measurement",
        )


def test_history_resolver_copies_and_reverifies_immutable_source(tmp_path):
    from agentbench_frame.generals.historical_policy import (
        resolve_historical_policies,
    )

    config = make_history(
        tmp_path,
        "v0",
        "run-0",
        {
            "main.py": "VALUE = 1\n",
            "nested/strategy.py": "CHOICE = 2\n",
        },
    )

    (resolved,) = resolve_historical_policies(
        (config,),
        tmp_path,
        tmp_path / "measurement",
    )
    original = (
        tmp_path
        / "runs/28_generals/generals-hl/run-0/versions/v0/source/main.py"
    )
    original.write_text("VALUE = 99\n", encoding="utf-8")

    assert (resolved.source / "main.py").read_text(encoding="utf-8") == (
        "VALUE = 1\n"
    )
    assert resolved.content_hash == config.content_hash


def test_probe_runs_twice_and_appends_one_sdk_end_marker(
    engine_root,
    tmp_path,
):
    from agentbench_frame.generals.historical_policy import (
        probe_historical_policy,
        resolve_historical_policies,
    )

    config = make_history(
        tmp_path,
        "v0",
        "run-0",
        {
            "main.py": (
                "def agent(round_number, my_seat, state):\n"
                "    assert state.rest_move_step == [1, 2]\n"
                "    return [[5, 1]]\n"
            ),
        },
    )
    (policy,) = resolve_historical_policies(
        (config,),
        tmp_path,
        tmp_path / "measurement",
    )
    engine = OfficialGeneralsEngine(
        engine_root,
        289101,
        tmp_path / "official.jsonl",
    )
    engine.state.rest_move_step = [1, 2]

    result = probe_historical_policy(
        policy,
        engine.measurement_state(0),
        engine_root=engine_root,
        sdk_root=tmp_path,
        repeats=2,
        timeout_s=5.0,
    )

    assert result.status == "complete"
    assert result.deterministic is True
    assert result.raw_actions[0] == result.raw_actions[1]
    assert result.raw_actions[0] == ((5, 1), (8,))


def test_probe_reports_nondeterminism_across_fresh_processes(
    engine_root,
    tmp_path,
):
    from agentbench_frame.generals.historical_policy import (
        probe_historical_policy,
        resolve_historical_policies,
    )

    config = make_history(
        tmp_path,
        "v0",
        "run-0",
        {
            "main.py": (
                "import os\n"
                "def agent(round_number, my_seat, state):\n"
                "    return [[1, 0, 0, 1, os.getpid()]]\n"
            ),
        },
    )
    (policy,) = resolve_historical_policies(
        (config,),
        tmp_path,
        tmp_path / "measurement",
    )
    engine = OfficialGeneralsEngine(engine_root, 289101, tmp_path / "a.jsonl")

    result = probe_historical_policy(
        policy,
        engine.measurement_state(0),
        engine_root=engine_root,
        sdk_root=tmp_path,
        repeats=2,
        timeout_s=5.0,
    )

    assert result.status == "nondeterministic"
    assert result.deterministic is False


def test_probe_captures_timeout_and_stderr(engine_root, tmp_path):
    from agentbench_frame.generals.historical_policy import (
        probe_historical_policy,
        resolve_historical_policies,
    )

    config = make_history(
        tmp_path,
        "v0",
        "run-0",
        {
            "main.py": (
                "import sys, time\n"
                "def agent(round_number, my_seat, state):\n"
                "    print('before sleep', file=sys.stderr)\n"
                "    time.sleep(2)\n"
                "    return []\n"
            ),
        },
    )
    (policy,) = resolve_historical_policies(
        (config,),
        tmp_path,
        tmp_path / "measurement",
    )
    engine = OfficialGeneralsEngine(engine_root, 289101, tmp_path / "a.jsonl")

    result = probe_historical_policy(
        policy,
        engine.measurement_state(0),
        engine_root=engine_root,
        sdk_root=tmp_path,
        repeats=2,
        timeout_s=0.5,
    )

    assert result.status == "timeout"
    assert "before sleep" in result.stderr


@pytest.mark.parametrize(
    ("main_source", "expected_status"),
    [
        (
            "print('noise')\n"
            "def agent(round_number, my_seat, state): return []\n",
            "malformed_output",
        ),
        ("raise RuntimeError('cannot import')\n", "worker_error"),
    ],
)
def test_probe_reports_worker_protocol_failures(
    engine_root,
    tmp_path,
    main_source,
    expected_status,
):
    from agentbench_frame.generals.historical_policy import (
        probe_historical_policy,
        resolve_historical_policies,
    )

    config = make_history(
        tmp_path,
        "v0",
        "run-0",
        {"main.py": main_source},
    )
    (policy,) = resolve_historical_policies(
        (config,),
        tmp_path,
        tmp_path / "measurement",
    )
    engine = OfficialGeneralsEngine(engine_root, 289101, tmp_path / "a.jsonl")

    result = probe_historical_policy(
        policy,
        engine.measurement_state(0),
        engine_root=engine_root,
        sdk_root=tmp_path,
        repeats=2,
        timeout_s=5.0,
    )

    assert result.status == expected_status
    assert result.deterministic is False


def test_real_v6_probe_uses_its_original_state_view(
    engine_root,
    tmp_path,
):
    from agentbench_frame.generals.historical_policy import (
        probe_historical_policy,
        resolve_historical_policies,
    )

    data_dir = (
        Path(__file__).resolve().parents[2]
        / "agentbench_data"
    )
    version_root = (
        data_dir
        / "runs/28_generals/generals-hl"
        / "20260729_1653_af8eda26/versions/v6"
    )
    if not version_root.is_dir():
        pytest.skip("real v6 history is unavailable")
    config = HistoricalPolicyConfig(
        version="v6",
        run_id="20260729_1653_af8eda26",
        content_hash=(
            "974050ee1a3d4b4c4f96e61f5af39b4"
            "e52e2cfbc50f146f9e8b4ac96c4ac798b"
        ),
    )
    (policy,) = resolve_historical_policies(
        (config,),
        data_dir,
        tmp_path / "measurement",
    )
    engine = OfficialGeneralsEngine(engine_root, 289101, tmp_path / "a.jsonl")
    engine.state.rest_move_step = [0, 2]
    asset_root = Path(os.environ["AGENTBENCH_ASSET_ROOT"])
    sdk_root = (
        asset_root
        / "top_algorithms/corpus/28_generals_popular_final/extracted"
        / "rank16__Xiaoaojianghu__痔取松弛肛__v1"
    )

    result = probe_historical_policy(
        policy,
        engine.measurement_state(0),
        engine_root=engine_root,
        sdk_root=sdk_root,
        repeats=2,
        timeout_s=5.0,
    )

    assert result.status == "complete"
    assert result.raw_actions[0][-1] == (8,)
