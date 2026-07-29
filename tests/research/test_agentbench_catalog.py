from pathlib import Path

import pytest

from agentbench_frame.research.agentbench_catalog import (
    AGENTBENCH_GAME_SPECS,
    collect_game_sources,
    validate_agentbench_corpus,
)

EXPECTED_GAME_IDS = {
    "23_doto",
    "24_miracle",
    "25_aquawar",
    "25_lostspace",
    "26_snakego",
    "27_antwar",
    "28_generals",
    "29_rollman",
    "30_antwar2",
    "30_deepclue",
}


def _spec(game_id: str):
    return next(spec for spec in AGENTBENCH_GAME_SPECS if spec.game_id == game_id)


def _source_root(repo: Path, game_id: str) -> Path:
    spec = _spec(game_id)
    root = repo / spec.source_root
    root.mkdir(parents=True)
    return root


def _materialize_all_games(repo: Path) -> None:
    for spec in AGENTBENCH_GAME_SPECS:
        root = repo / spec.source_root
        root.mkdir(parents=True)
        (root / "rules.py").write_text(f"GAME = {spec.game_id!r}\n")


def test_catalog_names_every_public_game_once():
    game_ids = [spec.game_id for spec in AGENTBENCH_GAME_SPECS]

    assert len(game_ids) == 10
    assert len(set(game_ids)) == 10
    assert set(game_ids) == EXPECTED_GAME_IDS


def test_collect_game_sources_keeps_rule_code_and_configuration(tmp_path):
    root = _source_root(tmp_path, "25_lostspace")
    expected = {
        "engine.py": b"print('engine')\n",
        "native/rule.cpp": b"int rule = 1;\n",
        "native/rule.h": b"#define RULE 1\n",
        "native/rule.hpp": b"struct Rule {};\n",
        "native/rule.c": b"int c_rule = 1;\n",
        "rules/config.json": b'{"turns": 8}\n',
        "rules/map.g4": b"grammar Map;\n",
        "rules/arena.map": b"floor\n",
    }
    for relative_path, content in expected.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    modules = collect_game_sources(tmp_path, _spec("25_lostspace"))

    assert [module.path for module in modules] == sorted(expected)
    assert {module.path: module.content for module in modules} == expected


def test_collect_game_sources_excludes_non_authoritative_material(tmp_path):
    root = _source_root(tmp_path, "30_deepclue")
    (root / "engine.py").write_text("ENGINE = True\n")

    excluded = {
        "data/0/story.json": "{}",
        "jsoncpp/json.hpp": "// vendor",
        "lib/helper.cpp": "// vendor",
        "output/main.cpp": "// generated output",
        "test_config/case.json": "{}",
        "tests/test_engine.py": "assert True",
        "judge_dev_sample_ai/bot.py": "pass",
        "bak/old.py": "pass",
        "docs/rules.py": "pass",
        "MapLexer.py": "# generated",
        "MapParser.py": "# generated",
        "MapListener.py": "# generated",
        "main_test.py": "pass",
        "main_with_debug.py": "pass",
        "old_main.py": "pass",
        "ai_demo.py": "pass",
        "upload.py": "pass",
        "binary.png": "not source",
        "README.md": "not executable logic",
        "Makefile": "build only",
    }
    for relative_path, content in excluded.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    modules = collect_game_sources(tmp_path, _spec("30_deepclue"))

    assert [module.path for module in modules] == ["engine.py"]


def test_collect_game_sources_rejects_missing_or_empty_root(tmp_path):
    spec = _spec("24_miracle")

    with pytest.raises(ValueError, match="missing authoritative source root"):
        collect_game_sources(tmp_path, spec)

    (tmp_path / spec.source_root).mkdir(parents=True)
    with pytest.raises(ValueError, match="no selected logic files"):
        collect_game_sources(tmp_path, spec)


def test_collect_game_sources_rejects_symlinks_outside_authoritative_root(tmp_path):
    root = _source_root(tmp_path, "26_snakego")
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = True\n")
    (root / "rules.py").symlink_to(outside)

    with pytest.raises(ValueError, match="symbolic link"):
        collect_game_sources(tmp_path, _spec("26_snakego"))


def test_validate_agentbench_corpus_fails_closed_on_missing_or_extra_game(tmp_path):
    _materialize_all_games(tmp_path)
    validate_agentbench_corpus(tmp_path)

    missing = tmp_path / "backend_sources/corpus/24_miracle"
    for path in sorted(missing.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    missing.rmdir()
    with pytest.raises(ValueError, match="game set mismatch.*missing=24_miracle"):
        validate_agentbench_corpus(tmp_path)

    _source_root(tmp_path, "24_miracle").joinpath("rules.py").write_text("pass\n")
    extra = tmp_path / "backend_sources/corpus/31_unknown/logic/gamecode_logic"
    extra.mkdir(parents=True)
    (extra / "rules.py").write_text("pass\n")
    with pytest.raises(ValueError, match="extra=31_unknown"):
        validate_agentbench_corpus(tmp_path)
