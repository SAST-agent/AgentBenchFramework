import json
from pathlib import Path

from agentbench_frame.research.agentbench_catalog import (
    AGENTBENCH_GAME_SPECS,
    SOURCE_MANIFEST_SHA256,
)

REPORT_PATH = Path("docs/research/agentbench-ludi-k-v1.json")
SOURCE_COMMIT = "b581bca3ba3d2d7d58a2f8c6bbddd060fc7fdc87"


def test_frozen_agentbench_report_covers_every_game_and_source_file():
    report = json.loads(REPORT_PATH.read_text())

    assert report["schema_version"] == "agentbench.ludi-k.v1"
    assert report["reference_machine"]["family"] == "AB-LUDI/1"
    assert report["reference_machine"]["id"].startswith("AB-LUDI/1+zlib-")
    assert report["reference_machine"]["decoder_constant_included"] is False
    assert report["source"]["commit"] == SOURCE_COMMIT
    assert report["source"]["manifest_sha256"] == SOURCE_MANIFEST_SHA256

    games = report["games"]
    expected_ids = {spec.game_id for spec in AGENTBENCH_GAME_SPECS}
    assert len(games) == 10
    assert {game["game_id"] for game in games} == expected_ids
    assert [game["k_upper_bits"] for game in games] == sorted(
        game["k_upper_bits"] for game in games
    )

    for game in games:
        spec = next(
            item for item in AGENTBENCH_GAME_SPECS if item.game_id == game["game_id"]
        )
        assert game["module_count"] == len(game["files"]) > 0
        assert [
            (file["path"], file["sha256"]) for file in game["files"]
        ] == [(file.path, file.sha256) for file in spec.files]
        assert game["source_bytes"] == sum(file["bytes"] for file in game["files"])
        assert game["source_bits"] == game["source_bytes"] * 8
        assert game["canonical_bits"] == game["canonical_bytes"] * 8
        assert game["k_upper_bits"] == game["compressed_bytes"] * 8
        assert game["source_bits"] > 0
        assert game["canonical_bits"] > 0
        assert game["k_upper_bits"] > 0
        assert len(game["source_sha256"]) == 64
        assert len(game["description_sha256"]) == 64
        assert all(len(file["sha256"]) == 64 for file in game["files"])
