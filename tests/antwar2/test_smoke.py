from pathlib import Path


def test_missing_candidate_source_is_failed_smoke(tmp_path):
    from agentbench_frame.games.antwar2.smoke import verify_candidate_smoke

    result = verify_candidate_smoke(tmp_path)

    assert result.status == "failed"
    assert "ai.py" in result.error


def test_historical_v22_passes_public_state_smoke_when_fixture_is_available(tmp_path):
    from agentbench_frame.games.antwar2.runtime import assemble_candidate
    from agentbench_frame.games.antwar2.smoke import verify_candidate_smoke

    handoff = Path(
        "/Users/qingle/Library/Containers/com.tencent.xinWeChat/Data/Documents/"
        "xwechat_files/wxid_j335ee7y0vgw12_7288/msg/file/2026-08/"
        "handoff_next_agent_2026-08-01"
    )
    if not handoff.is_dir():
        return
    candidate = tmp_path / "v22"
    assemble_candidate(
        destination=candidate,
        policy_root=handoff / "04_versions/ifelse_v22",
        sdk_root=handoff / "04_versions/canonical_SDK",
    )

    result = verify_candidate_smoke(candidate)

    assert result.status == "complete"
    assert result.artifacts["roles"] == ["P0", "P1"]
    assert len(result.artifacts["policy_sha256"]) == 64


def test_smoke_cli_serializes_immutable_artifacts(tmp_path, monkeypatch):
    from agentbench_frame.games.antwar2 import smoke
    from agentbench_frame.hl.game_profile import SmokeResult

    monkeypatch.setattr(
        smoke,
        "verify_candidate_smoke",
        lambda workspace: SmokeResult(
            status="complete",
            error=None,
            artifacts={"roles": ["P0", "P1"], "counts": {"P0": 2, "P1": 2}},
        ),
    )
    output = tmp_path / "result.json"

    assert smoke.main(["--workspace", str(tmp_path), "--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8").startswith('{\n  "artifacts"')
