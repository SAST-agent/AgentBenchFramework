import json

import pytest

from tests.generals.test_pipeline_v7 import (
    ChampionEvaluator,
    PhaseExceptionEvaluator,
    Round7Provider,
    _pipeline,
)


def _summary(run_dir):
    return json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )


def test_v7_recovery_reuses_exact_frozen_candidate_without_new_act(tmp_path):
    provider = Round7Provider()
    pipeline, _ = _pipeline(
        tmp_path,
        provider,
        PhaseExceptionEvaluator("validation"),
    )
    failed = pipeline.run()
    failed_summary_bytes = (failed.run_dir / "summary.json").read_bytes()
    recovery_evaluator = ChampionEvaluator()
    pipeline.evaluator = recovery_evaluator

    recovered = pipeline.recover(failed.run_dir)

    assert recovered.run_dir != failed.run_dir
    assert provider.calls == 1
    assert recovered.round_act_count == 0
    assert recovered.global_act_count == 8
    assert recovered.status == "complete"
    assert recovered.champion_claim is True
    assert recovery_evaluator.calls == [
        ("v7", "validation", 12),
        ("v7", "formal", 18),
        ("v7", "sealed", 20),
    ]
    summary = _summary(recovered.run_dir)
    assert summary["failed_run_id"] == failed.run_dir.name
    assert summary["recovery_mode"] == "frozen_v7"
    assert summary["source_learning_budget"] == (
        _summary(failed.run_dir)["local_learning_budget"]
    )
    assert (failed.run_dir / "summary.json").read_bytes() == (
        failed_summary_bytes
    )


def test_v7_frozen_recovery_retains_candidate_test_failure(
    tmp_path,
    monkeypatch,
):
    pipeline, _ = _pipeline(
        tmp_path,
        Round7Provider(),
        PhaseExceptionEvaluator("validation"),
    )
    failed = pipeline.run()
    monkeypatch.setattr(
        pipeline,
        "_run_candidate_tests",
        lambda workspace: (False, "candidate regression\n"),
    )

    recovered = pipeline.recover(failed.run_dir)

    assert recovered.status == "invalid_version"
    assert recovered.runnable is False
    assert recovered.round_act_count == 0
    summary = _summary(recovered.run_dir)
    assert summary["status"] == "invalid_version"
    assert summary["runnable"] is False
    assert summary["champion_validation"]["status"] == "not_run"
    assert summary["champion_sealed"]["status"] == "not_opened"
    assert (
        recovered.run_dir / "versions/v7/tests.log"
    ).read_text(encoding="utf-8") == "candidate regression\n"


def test_v7_recovery_after_provider_failure_uses_new_visible_act(tmp_path):
    provider = Round7Provider(fail=True)
    pipeline, evaluator = _pipeline(tmp_path, provider)
    failed = pipeline.run()
    provider.fail = False
    evaluator.calls.clear()

    recovered = pipeline.recover(failed.run_dir)

    assert recovered.run_dir != failed.run_dir
    assert provider.calls == 2
    assert recovered.round_act_count == 1
    assert recovered.global_act_count == 8
    assert evaluator.calls == [
        ("v6", "learning", 6),
        ("v7", "validation", 12),
        ("v7", "formal", 18),
        ("v7", "sealed", 20),
    ]
    summary = _summary(recovered.run_dir)
    assert summary["failed_run_id"] == failed.run_dir.name
    assert summary["recovery_mode"] == "fresh_retry"


def test_v7_recovery_rejects_changed_frozen_source(tmp_path):
    pipeline, _ = _pipeline(
        tmp_path,
        Round7Provider(),
        PhaseExceptionEvaluator("validation"),
    )
    failed = pipeline.run()
    (failed.run_dir / "versions/v7/source/strategy.py").write_text(
        "tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="v7 source"):
        pipeline.recover(failed.run_dir)


def test_v7_recovery_rejects_changed_prompt_receipt(tmp_path):
    pipeline, _ = _pipeline(
        tmp_path,
        Round7Provider(),
        PhaseExceptionEvaluator("validation"),
    )
    failed = pipeline.run()
    prompt = failed.run_dir / "provider/codex-act-v7.prompt.md"
    prompt.write_text(
        prompt.read_text(encoding="utf-8") + "\ntampered\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prompt"):
        pipeline.recover(failed.run_dir)
