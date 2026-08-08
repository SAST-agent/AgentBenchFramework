import argparse
import hashlib
import json

import agentbench_frame.generals.cli as cli
from agentbench_frame.generals.cli import register_parser
from agentbench_frame.generals.leaderboard_qualification import (
    QUALIFICATION_BUDGETS,
    QUALIFICATION_ID,
    QUALIFICATION_RESULT_SCHEMA,
    LeaderboardQualificationResult,
    QualificationCheckpoint,
)


def _parser():
    parser = argparse.ArgumentParser()
    register_parser(parser.add_subparsers(dest="command"))
    return parser


def _result(provider, wins):
    checkpoints = []
    for replicate_index in range(1, 4):
        for acts in QUALIFICATION_BUDGETS:
            checkpoints.append(
                QualificationCheckpoint(
                    replicate_id=f"replicate-{replicate_index}",
                    coding_agent_acts=acts,
                    status="complete",
                    passed=True,
                    policy_hash="p" * 64,
                    valid_games=40,
                    wins=wins,
                    losses=40 - wins,
                    draws=0,
                    score=wins / 40,
                    win_rate=wins / 40,
                    win_rate_lower_bound=0.5,
                    per_seat_wins={0: wins // 2, 1: wins // 2},
                    reasons=(),
                    budget={
                        "coding_agent_acts": acts,
                        "total_tokens": acts * 100,
                        "wall_time_s": acts * 2.0,
                        "learning_episodes": acts * 6,
                        "cost_usd": acts * 0.25,
                    },
                )
            )
    return LeaderboardQualificationResult(
        qualification_id=QUALIFICATION_ID,
        status="qualified",
        qualified=True,
        provider=provider,
        model="frontier-model",
        model_revision="2026-08-09",
        harness_hash="h" * 64,
        qualifying_replicates=("replicate-1", "replicate-2", "replicate-3"),
        minimum_qualifying_replicates=2,
        qualification_budget_acts=8,
        checkpoints=tuple(checkpoints),
        reasons=(),
    )


def test_registers_leaderboard_aggregation_without_data_directory(tmp_path):
    args = _parser().parse_args(
        [
            "generals",
            "build-leaderboard",
            "--agentbench-root", "/assets",
            "--manifest", "/assets/pilot.toml",
            "--qualification-manifest", "/assets/qualification.toml",
            "--qualification-result", str(tmp_path / "provider-a.json"),
            "--qualification-result", str(tmp_path / "provider-b.json"),
            "--output", str(tmp_path / "leaderboard.json"),
        ]
    )

    assert args.generals_command == "build-leaderboard"
    assert len(args.qualification_result) == 2
    assert not hasattr(args, "data_dir")


def test_routes_hashed_qualified_results_into_ranked_leaderboard(
    tmp_path, monkeypatch, capsys
):
    qualification_manifest = tmp_path / "qualification.toml"
    qualification_manifest.write_text("qualification_id = 'test'\n", encoding="utf-8")
    manifest_hash = hashlib.sha256(qualification_manifest.read_bytes()).hexdigest()
    result_paths = []
    for provider, wins in (("api-a", 30), ("api-b", 26)):
        path = tmp_path / f"{provider}.json"
        payload = {
            "schema": QUALIFICATION_RESULT_SCHEMA,
            "manifest_sha256": manifest_hash,
            "receipt_sha256": provider * 8,
            **_result(provider, wins).to_dict(),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        result_paths.append(path)
    output = tmp_path / "out" / "leaderboard.json"
    args = _parser().parse_args(
        [
            "generals",
            "build-leaderboard",
            "--agentbench-root", "/assets",
            "--manifest", "/assets/pilot.toml",
            "--qualification-manifest", str(qualification_manifest),
            "--qualification-result", str(result_paths[0]),
            "--qualification-result", str(result_paths[1]),
            "--output", str(output),
        ]
    )
    monkeypatch.setattr(cli, "_assets", lambda _args: (object(), object()))

    assert cli.handle(args) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == json.loads(capsys.readouterr().out)
    assert payload["qualification_manifest_sha256"] == manifest_hash
    assert len(payload["qualification_result_sha256"]) == 2
    entries = payload["checkpoints"][0]["entries"]
    assert [entry["provider"] for entry in entries] == ["api-a", "api-b"]
    assert [entry["rank"] for entry in entries] == [1, 2]
