"""Frozen source boundary for public AgentBench game logic.

The catalog intentionally selects one authoritative backend implementation for
each public game. It excludes duplicate judge-dev copies, sample agents,
generated/build artifacts, tests, vendored libraries, and non-engine content.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GameSourceSpec:
    """Authoritative public source root for one AgentBench game."""

    game_id: str
    title: str
    source_root: str
    excluded_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceModule:
    """One exact source blob with a canonical path relative to its game root."""

    path: str
    content: bytes


AGENTBENCH_GAME_SPECS: tuple[GameSourceSpec, ...] = (
    GameSourceSpec(
        "23_doto",
        "DOTO",
        (
            "backend_sources/corpus/23_doto/logic/public_Arena-Doto-AI/"
            "Arena-Doto-AI-master/server"
        ),
        ("GamePlayerInUnity.py", "server2.py", "server3.py"),
    ),
    GameSourceSpec(
        "24_miracle",
        "Miracle",
        "backend_sources/corpus/24_miracle/logic/gamecode_logic",
    ),
    GameSourceSpec(
        "25_aquawar",
        "AquaWar",
        "backend_sources/corpus/25_aquawar/logic/gamecode_logic",
    ),
    GameSourceSpec(
        "25_lostspace",
        "LostSpace",
        "backend_sources/corpus/25_lostspace/logic/gamecode_logic",
        (
            "src/MapGen/MapLexer.py",
            "src/MapGen/MapListener.py",
            "src/MapGen/MapParser.py",
        ),
    ),
    GameSourceSpec(
        "26_snakego",
        "SnakeGo",
        "backend_sources/corpus/26_snakego/logic/gamecode_logic",
    ),
    GameSourceSpec(
        "27_antwar",
        "AntWar",
        (
            "backend_sources/corpus/27_antwar/logic/gamecode_logic/"
            "ant_game - deploy"
        ),
    ),
    GameSourceSpec(
        "28_generals",
        "Generals",
        "backend_sources/corpus/28_generals/logic/gamecode_logic",
        ("main_for_player_test.py",),
    ),
    GameSourceSpec(
        "29_rollman",
        "Rollman",
        (
            "backend_sources/corpus/29_rollman/logic/gamecode_logic/"
            "PacmanLogic"
        ),
    ),
    GameSourceSpec(
        "30_antwar2",
        "AntWar2",
        (
            "backend_sources/corpus/30_antwar2/logic/gamecode_logic/game"
        ),
    ),
    GameSourceSpec(
        "30_deepclue",
        "DeepClue",
        "backend_sources/corpus/30_deepclue/logic/gamecode_logic",
    ),
)


_SOURCE_SUFFIXES = frozenset(
    {".py", ".c", ".cpp", ".h", ".hpp", ".json", ".g4", ".map"}
)
_EXCLUDED_DIRECTORIES = frozenset(
    {
        "__pycache__",
        "bak",
        "backup",
        "data",
        "doc",
        "docs",
        "judge_dev_sample_ai",
        "jsoncpp",
        "lib",
        "output",
        "sample_ai",
        "slide",
        "test",
        "test_config",
        "tests",
    }
)
_EXCLUDED_FILENAMES = frozenset(
    {
        "MapLexer.py",
        "MapListener.py",
        "MapParser.py",
        "ai_demo.py",
        "json.hpp",
        "main_test.py",
        "main_with_debug.py",
        "old_main.py",
        "test.py",
        "upload.py",
    }
)


def _is_selected(relative_path: Path, spec: GameSourceSpec) -> bool:
    path_string = relative_path.as_posix()
    if path_string in spec.excluded_paths:
        return False
    if relative_path.name in _EXCLUDED_FILENAMES:
        return False
    if relative_path.suffix.lower() not in _SOURCE_SUFFIXES:
        return False
    return not any(
        part.casefold() in _EXCLUDED_DIRECTORIES
        for part in relative_path.parts[:-1]
    )


def collect_game_sources(
    repo_root: str | Path,
    spec: GameSourceSpec,
) -> tuple[SourceModule, ...]:
    """Collect exact source modules for one game in canonical path order."""

    root = Path(repo_root) / spec.source_root
    if not root.is_dir():
        raise ValueError(
            f"{spec.game_id}: missing authoritative source root: {spec.source_root}"
        )

    paths = sorted(
        root.rglob("*"),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    modules = []
    for path in paths:
        relative_path = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(
                f"{spec.game_id}: symbolic link is not allowed: "
                f"{relative_path.as_posix()}"
            )
        if path.is_file() and _is_selected(relative_path, spec):
            modules.append(
                SourceModule(
                    path=relative_path.as_posix(),
                    content=path.read_bytes(),
                )
            )
    result = tuple(modules)
    if not result:
        raise ValueError(f"{spec.game_id}: no selected logic files in {spec.source_root}")
    return result


def validate_agentbench_corpus(repo_root: str | Path) -> None:
    """Fail closed unless the checkout has exactly the frozen public game set."""

    corpus_root = Path(repo_root) / "backend_sources/corpus"
    if not corpus_root.is_dir():
        raise ValueError(f"missing AgentBench corpus directory: {corpus_root}")

    actual = {path.name for path in corpus_root.iterdir() if path.is_dir()}
    expected = {spec.game_id for spec in AGENTBENCH_GAME_SPECS}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={','.join(missing)}")
        if extra:
            details.append(f"extra={','.join(extra)}")
        raise ValueError(f"AgentBench game set mismatch: {'; '.join(details)}")

    for spec in AGENTBENCH_GAME_SPECS:
        collect_game_sources(repo_root, spec)
