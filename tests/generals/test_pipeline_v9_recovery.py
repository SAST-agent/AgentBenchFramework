import json

import pytest

from tests.generals.test_pipeline_v9 import (
    V9Evaluator,
    V9Provider,
    _pipeline,
)


class RaisingProvider:
    provider_name = "codex"

    def __init__(self):
        self.calls = 0

    def invoke(self, context):
        self.calls += 1
        raise AssertionError("recovery must not invoke provider")


def test_v9_recovery_uses_frozen_candidate_without_provider(tmp_path):
    first_provider = V9Provider()
    failed = _pipeline(
        tmp_path / "failed",
        provider=first_provider,
        evaluator=V9Evaluator(raise_phase="formal9"),
    ).run()
    assert first_provider.calls == 1
    assert failed.runnable is True

    recovery_provider = RaisingProvider()
    recovered = _pipeline(
        tmp_path / "recovered",
        provider=recovery_provider,
        evaluator=V9Evaluator(),
    ).recover(failed.run_dir)

    assert recovery_provider.calls == 0
    assert recovered.runnable is True
    assert recovered.formal_attempted is True
    assert recovered.champion_claim is True
    summary = json.loads((recovered.run_dir / "summary.json").read_text())
    assert summary["recovery_mode"] == "frozen_candidate"
    assert summary["recovered_from_run_id"] == failed.run_dir.name
    assert summary["round_act_count"] == 1


@pytest.mark.parametrize("target", ["prompt", "candidate", "events"])
def test_v9_recovery_rejects_changed_authority(tmp_path, target):
    failed = _pipeline(
        tmp_path / target,
        provider=V9Provider(),
        evaluator=V9Evaluator(raise_phase="formal9"),
    ).run()
    if target == "prompt":
        (failed.run_dir / "provider/codex-act-v9.prompt.md").write_text("changed\n")
    elif target == "candidate":
        (failed.run_dir / "versions/v9/source/strategy.py").write_text("changed\n")
    else:
        with (failed.run_dir / "events.jsonl").open("a") as stream:
            stream.write('{"event_type":"changed"}\n')

    with pytest.raises(ValueError, match="changed|digest|hash|manifest"):
        _pipeline(
            tmp_path / f"recover-{target}",
            provider=RaisingProvider(),
            evaluator=V9Evaluator(),
        ).recover(failed.run_dir)
