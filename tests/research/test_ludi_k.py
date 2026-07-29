import hashlib
import json

import pytest

from agentbench_frame.research.agentbench_catalog import (
    GameSourceSpec,
    SourceModule,
)
from agentbench_frame.research.ludi_k import (
    REFERENCE_MACHINE_ID,
    ZLIB_BEHAVIOR_FINGERPRINT,
    decode_ludi_description,
    encode_ludi_description,
    measure_game,
    write_json_report,
    write_markdown_report,
)


def test_reference_machine_identity_includes_compressor_behavior():
    assert REFERENCE_MACHINE_ID.startswith("AB-LUDI/1+zlib-")
    assert len(ZLIB_BEHAVIOR_FINGERPRINT) == 64
    assert ZLIB_BEHAVIOR_FINGERPRINT[:16] in REFERENCE_MACHINE_ID


def test_ludi_description_round_trips_arbitrary_source_bytes():
    modules = (
        SourceModule("a/(rule).py", b"left\x00right\n"),
        SourceModule("z.json", b'\xff{"x": 1}\x00'),
    )

    encoded = encode_ludi_description("game-\N{SNOWMAN}", modules)
    decoded_game_id, decoded_modules = decode_ludi_description(encoded)

    assert encoded.startswith(b"AB-LUDI/1\x00")
    assert decoded_game_id == "game-\N{SNOWMAN}"
    assert decoded_modules == modules


def test_ludi_description_is_independent_of_input_module_order():
    forward = (
        SourceModule("a.py", b"A"),
        SourceModule("b.py", b"B"),
    )

    assert encode_ludi_description("g", forward) == encode_ludi_description(
        "g", tuple(reversed(forward))
    )


@pytest.mark.parametrize(
    "modules, message",
    [
        (
            (SourceModule("same.py", b"A"), SourceModule("same.py", b"B")),
            "duplicate module path",
        ),
        ((SourceModule("../escape.py", b"A"),), "unsafe module path"),
        ((SourceModule("/absolute.py", b"A"),), "unsafe module path"),
    ],
)
def test_ludi_description_rejects_ambiguous_or_unsafe_paths(modules, message):
    with pytest.raises(ValueError, match=message):
        encode_ludi_description("g", modules)


def test_ludi_decoder_rejects_truncated_or_trailing_data():
    encoded = encode_ludi_description(
        "g",
        (SourceModule("rules.py", b"RULE = 1\n"),),
    )

    with pytest.raises(ValueError, match="truncated"):
        decode_ludi_description(encoded[:-1])
    with pytest.raises(ValueError, match="trailing"):
        decode_ludi_description(encoded + b"x")


def test_measure_game_reports_auditable_upper_bound_and_file_manifest():
    spec = GameSourceSpec("test_game", "Test Game", "logic/test_game")
    modules = (
        SourceModule("a.py", b"RULE = 1\n"),
        SourceModule("config.json", b'{"turns": 4}\n'),
    )

    result = measure_game(spec, modules)
    canonical = encode_ludi_description(spec.game_id, modules)

    assert result["game_id"] == "test_game"
    assert result["title"] == "Test Game"
    assert result["source_root"] == "logic/test_game"
    assert result["module_count"] == 2
    assert result["source_bytes"] == 22
    assert result["source_bits"] == 176
    assert result["canonical_bytes"] == len(canonical)
    assert result["canonical_bits"] == len(canonical) * 8
    assert result["compressed_bytes"] > 0
    assert result["k_upper_bits"] == result["compressed_bytes"] * 8
    assert result["description_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert 0 < result["compression_ratio"] < 2
    assert result["files"] == [
        {
            "path": "a.py",
            "bytes": 9,
            "sha256": hashlib.sha256(b"RULE = 1\n").hexdigest(),
        },
        {
            "path": "config.json",
            "bytes": 13,
            "sha256": hashlib.sha256(b'{"turns": 4}\n').hexdigest(),
        },
    ]

    changed = measure_game(
        spec,
        (
            SourceModule("a.py", b"RULE = 2\n"),
            SourceModule("config.json", b'{"turns": 4}\n'),
        ),
    )
    assert changed["source_sha256"] != result["source_sha256"]
    assert changed["description_sha256"] != result["description_sha256"]


def test_report_writers_are_stable_and_machine_readable(tmp_path):
    game = measure_game(
        GameSourceSpec("g", "Game", "logic/g"),
        (SourceModule("rules.py", b"RULE = 1\n"),),
    )
    report = {
        "schema_version": "agentbench.ludi-k.v1",
        "reference_machine": {
            "id": REFERENCE_MACHINE_ID,
            "metric": "conditional_k_upper_bits",
        },
        "source": {
            "repository": "https://github.com/Aoraku/AgentBench",
            "commit": "a" * 40,
        },
        "games": [game],
    }
    json_path = tmp_path / "nested/report.json"
    markdown_path = tmp_path / "nested/report.md"

    write_json_report(report, json_path)
    first_json = json_path.read_bytes()
    write_json_report(report, json_path)
    write_markdown_report(report, markdown_path)
    first_markdown = markdown_path.read_bytes()
    write_markdown_report(report, markdown_path)

    assert json_path.read_bytes() == first_json
    assert markdown_path.read_bytes() == first_markdown
    assert json.loads(first_json)["games"][0]["game_id"] == "g"
    markdown = first_markdown.decode()
    assert "AB-Ludi/1" in markdown
    assert "not exact Kolmogorov complexity" in markdown
    assert "| 1 | `g` | Game |" in markdown
