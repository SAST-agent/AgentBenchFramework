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
