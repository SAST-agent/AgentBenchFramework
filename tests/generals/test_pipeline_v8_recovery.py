import json

import pytest

from tests.generals.test_pipeline_v8 import (
    ChampionEvaluator,
    PhaseExceptionEvaluator,
    Round8Provider,
    _pipeline,
)


def test_clean_room_v8_recovery_never_invokes_provider_again(tmp_path):
    provider = Round8Provider()
    evaluator = PhaseExceptionEvaluator("formal")
    pipeline, _ = _pipeline(tmp_path, provider, evaluator)
    failed = pipeline.run()
    assert provider.calls == 1
    assert failed.runnable is True

    recovery_provider = Round8Provider()
    recovered_pipeline, _ = _pipeline(
        tmp_path / "recovery",
        recovery_provider,
        ChampionEvaluator(),
    )
    recovered_pipeline.parent_run_dir = pipeline.parent_run_dir
    recovered_pipeline.expected_parent_hash = pipeline.expected_parent_hash
    recovered = recovered_pipeline.recover(failed.run_dir)

    assert recovery_provider.calls == 0
    assert recovered.formal_attempted is True
    assert recovered.runnable is True


def _summary(run_dir):
    return json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )


