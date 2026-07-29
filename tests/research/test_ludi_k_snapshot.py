import json
from pathlib import Path

from agentbench_frame.research.agentbench_catalog import (
    AGENTBENCH_GAME_SPECS,
    SOURCE_MANIFEST_SHA256,
)
from agentbench_frame.research.ludi_k import (
    EXPECTED_ZLIB_BEHAVIOR_FINGERPRINT,
    REFERENCE_MACHINE_ID,
)

REPORT_PATH = Path("docs/research/agentbench-ludi-k-v1.json")
SOURCE_COMMIT = "b581bca3ba3d2d7d58a2f8c6bbddd060fc7fdc87"
EXPECTED_K_UPPER_BITS = {
    "23_doto": 113384,
    "24_miracle": 144200,
    "25_aquawar": 178632,
    "25_lostspace": 179032,
    "26_snakego": 79624,
    "27_antwar": 170528,
    "28_generals": 126888,
    "29_rollman": 101920,
    "30_antwar2": 299528,
    "30_deepclue": 189496,
}


def test_frozen_agentbench_report_covers_every_game_and_source_file():
    report = json.loads(REPORT_PATH.read_text())

    assert report["schema_version"] == "agentbench.ludi-k.v1"
    assert report["reference_machine"]["family"] == "AB-LUDI/1"
    assert report["reference_machine"]["id"] == REFERENCE_MACHINE_ID
    assert report["reference_machine"]["compressor"] == {
        "behavior_fingerprint": EXPECTED_ZLIB_BEHAVIOR_FINGERPRINT,
        "format": "zlib",
        "level": 9,
        "mem_level": 9,
        "method": "DEFLATED",
        "strategy": "Z_DEFAULT_STRATEGY",
        "wbits": 15,
    }
    assert report["reference_machine"]["decoder_constant_included"] is False
    assert report["source"]["commit"] == SOURCE_COMMIT
    assert report["source"]["manifest_sha256"] == SOURCE_MANIFEST_SHA256

    games = report["games"]
    expected_ids = {spec.game_id for spec in AGENTBENCH_GAME_SPECS}
    assert len(games) == 10
    assert {game["game_id"] for game in games} == expected_ids
    assert {
        game["game_id"]: game["k_upper_bits"] for game in games
    } == EXPECTED_K_UPPER_BITS
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
