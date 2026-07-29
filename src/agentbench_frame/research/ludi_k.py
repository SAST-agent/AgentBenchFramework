"""AB-Ludi/1 conditional Kolmogorov-complexity upper bounds.

AB-Ludi/1 represents a game as a recursive Ludi-style tree whose leaves are
lossless source modules. The reported program is a fixed zlib-9 encoding of
that canonical tree. It is a reproducible upper bound under this reference
machine, not exact or machine-independent Kolmogorov complexity.
"""

import json
import struct
import zlib
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from .agentbench_catalog import (
    AGENTBENCH_GAME_SPECS,
    AGENTBENCH_SOURCE_COMMIT,
    SOURCE_MANIFEST_SHA256,
    GameSourceSpec,
    SourceModule,
    collect_game_sources,
    git_head_commit,
    validate_agentbench_corpus,
)

SCHEMA_VERSION = "agentbench.ludi-k.v1"
_MAGIC = b"AB-LUDI/1\x00"
_UINT64 = struct.Struct(">Q")
_COMPRESSOR_TEST_VECTOR = (
    b"AB-Ludi/1 compressor behavior test vector\n" * 32
)


def _compress_with_profile(description: bytes) -> bytes:
    compressor = zlib.compressobj(
        level=9,
        method=zlib.DEFLATED,
        wbits=zlib.MAX_WBITS,
        memLevel=9,
        strategy=zlib.Z_DEFAULT_STRATEGY,
    )
    return compressor.compress(description) + compressor.flush()


ZLIB_BEHAVIOR_FINGERPRINT = sha256(
    _compress_with_profile(_COMPRESSOR_TEST_VECTOR)
).hexdigest()
REFERENCE_MACHINE_FAMILY = "AB-LUDI/1"
REFERENCE_MACHINE_ID = (
    f"{REFERENCE_MACHINE_FAMILY}+zlib-{zlib.ZLIB_RUNTIME_VERSION}"
    f"+tv-{ZLIB_BEHAVIOR_FINGERPRINT[:16]}"
)


def _frame(blob: bytes) -> bytes:
    return _UINT64.pack(len(blob)) + blob


def _validate_module_path(path: str) -> None:
    parsed = PurePosixPath(path)
    if (
        not path
        or "\\" in path
        or parsed.is_absolute()
        or ".." in parsed.parts
        or parsed.as_posix() != path
    ):
        raise ValueError(f"unsafe module path: {path!r}")


def _canonical_modules(
    modules: Iterable[SourceModule],
) -> tuple[SourceModule, ...]:
    ordered = tuple(sorted(modules, key=lambda module: module.path))
    seen: set[str] = set()
    for module in ordered:
        _validate_module_path(module.path)
        if module.path in seen:
            raise ValueError(f"duplicate module path: {module.path}")
        seen.add(module.path)
    return ordered


def _encode_module_sequence(modules: tuple[SourceModule, ...]) -> bytes:
    chunks = [_UINT64.pack(len(modules))]
    for module in modules:
        chunks.append(_frame(module.path.encode("utf-8")))
        chunks.append(_frame(module.content))
    return b"".join(chunks)


def encode_ludi_description(
    game_id: str,
    modules: Iterable[SourceModule],
) -> bytes:
    """Encode a lossless, injective AB-Ludi/1 game-logic description."""

    if not game_id:
        raise ValueError("game_id must be non-empty")
    ordered = _canonical_modules(modules)
    return _MAGIC + _frame(game_id.encode("utf-8")) + _encode_module_sequence(ordered)


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def take(self, size: int) -> bytes:
        end = self.offset + size
        if end > len(self.data):
            raise ValueError("truncated AB-Ludi/1 description")
        value = self.data[self.offset:end]
        self.offset = end
        return value

    def uint64(self) -> int:
        return _UINT64.unpack(self.take(_UINT64.size))[0]

    def blob(self) -> bytes:
        return self.take(self.uint64())


def decode_ludi_description(
    description: bytes,
) -> tuple[str, tuple[SourceModule, ...]]:
    """Decode an AB-Ludi/1 description and reject non-canonical framing."""

    reader = _Reader(description)
    if reader.take(len(_MAGIC)) != _MAGIC:
        raise ValueError("invalid AB-Ludi/1 magic")
    try:
        game_id = reader.blob().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("invalid UTF-8 game identifier") from exc

    modules = []
    for _ in range(reader.uint64()):
        try:
            path = reader.blob().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-8 module path") from exc
        modules.append(SourceModule(path, reader.blob()))

    if reader.offset != len(description):
        raise ValueError("trailing data in AB-Ludi/1 description")
    ordered = _canonical_modules(modules)
    if tuple(modules) != ordered:
        raise ValueError("non-canonical module order")
    if not game_id:
        raise ValueError("game_id must be non-empty")
    return game_id, ordered


def _compress_description(description: bytes) -> bytes:
    return _compress_with_profile(description)


def measure_game(
    spec: GameSourceSpec,
    modules: Iterable[SourceModule],
) -> dict[str, Any]:
    """Measure one selected game under the fixed AB-Ludi/1 reference machine."""

    ordered = _canonical_modules(modules)
    if not ordered:
        raise ValueError(f"{spec.game_id}: cannot measure an empty source set")

    description = encode_ludi_description(spec.game_id, ordered)
    compressed = _compress_description(description)
    module_stream = _encode_module_sequence(ordered)
    source_bytes = sum(len(module.content) for module in ordered)
    canonical_bytes = len(description)

    return {
        "game_id": spec.game_id,
        "title": spec.title,
        "source_root": spec.source_root,
        "module_count": len(ordered),
        "source_bytes": source_bytes,
        "source_bits": source_bytes * 8,
        "canonical_bytes": canonical_bytes,
        "canonical_bits": canonical_bytes * 8,
        "compressed_bytes": len(compressed),
        "k_upper_bits": len(compressed) * 8,
        "compression_ratio": len(compressed) / canonical_bytes,
        "source_sha256": sha256(module_stream).hexdigest(),
        "description_sha256": sha256(description).hexdigest(),
        "files": [
            {
                "path": module.path,
                "bytes": len(module.content),
                "sha256": sha256(module.content).hexdigest(),
            }
            for module in ordered
        ],
    }


def measure_agentbench_repository(
    repo_root: str | Path,
    *,
    repository_url: str = "https://github.com/Aoraku/AgentBench",
) -> dict[str, Any]:
    """Calculate all frozen public AgentBench game-logic bounds."""

    root = Path(repo_root).resolve()
    source_commit = git_head_commit(root)
    if source_commit != AGENTBENCH_SOURCE_COMMIT:
        raise ValueError(
            "AB-Ludi/1 requires AgentBench commit "
            f"{AGENTBENCH_SOURCE_COMMIT}, got {source_commit}"
        )
    validate_agentbench_corpus(root, source_commit)
    games = [
        measure_game(spec, collect_game_sources(root, spec, source_commit))
        for spec in AGENTBENCH_GAME_SPECS
    ]
    games.sort(key=lambda game: (game["k_upper_bits"], game["game_id"]))

    return {
        "schema_version": SCHEMA_VERSION,
        "reference_machine": {
            "family": REFERENCE_MACHINE_FAMILY,
            "id": REFERENCE_MACHINE_ID,
            "metric": "conditional_k_upper_bits",
            "encoding": "uint64-be-length-prefixed source-module ludeme tree",
            "compressor": {
                "format": "zlib",
                "level": 9,
                "method": "DEFLATED",
                "wbits": zlib.MAX_WBITS,
                "mem_level": 9,
                "strategy": "Z_DEFAULT_STRATEGY",
                "compile_version": zlib.ZLIB_VERSION,
                "runtime_version": zlib.ZLIB_RUNTIME_VERSION,
                "behavior_fingerprint": ZLIB_BEHAVIOR_FINGERPRINT,
            },
            "decoder_constant_included": False,
            "conditioned_on": [
                "AB-Ludi/1 decoder",
                "language runtimes and toolchains",
            ],
            "literature": [
                {
                    "title": "Evolutionary Game Design",
                    "url": (
                        "https://cambolbro.com/cv/publications/"
                        "ciaig-browne-maire-19.pdf"
                    ),
                },
                {
                    "title": "Measuring Intelligence through Games",
                    "url": "https://arxiv.org/abs/1109.1314",
                },
            ],
        },
        "source": {
            "repository": repository_url,
            "commit": source_commit,
            "manifest_schema": "agentbench.ludi-source-manifest.v1",
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "catalog_game_count": len(AGENTBENCH_GAME_SPECS),
        },
        "games": games,
    }


def write_json_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a stable, UTF-8, machine-readable report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def _format_ratio(value: float) -> str:
    return f"{value:.4f}"


def write_markdown_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a stable human-readable report from the JSON-compatible result."""

    source = report["source"]
    games = sorted(
        report["games"],
        key=lambda game: (game["k_upper_bits"], game["game_id"]),
    )
    lines = [
        "# AgentBench AB-Ludi/1 K-Complexity Upper Bounds",
        "",
        (
            f"Source: [{source['repository']}]({source['repository']}) at "
            f"`{source['commit']}`."
        ),
        "",
        (
            "These values are **not exact Kolmogorov complexity**. They are "
            "conditional executable-description upper bounds under the fixed "
            "AB-Ludi/1 source-module reference machine."
        ),
        "",
        (
            "`k_upper_bits` is eight times the byte length of the canonical "
            "ludeme tree compressed with the report's fixed zlib-9 profile. "
            "The reference-machine ID binds the zlib runtime and fixed-vector "
            "behavior fingerprint. The shared decoder and language runtimes "
            "are conditioned out."
        ),
        "",
        "$$",
        (
            "K(G \\mid U_{\\mathrm{AB\\text{-}Ludi/1}}, R)"
            " \\leq 8\\,\\left|\\operatorname{zlib9}("
            "\\operatorname{encode}_{\\mathrm{AB\\text{-}Ludi/1}}(G))"
            "\\right| + \\mathcal{O}(1)"
        ),
        "$$",
        "",
        "| Rank | Game ID | Game | Modules | Source bits | K upper bits | Ratio |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for rank, game in enumerate(games, start=1):
        lines.append(
            f"| {rank} | `{game['game_id']}` | {game['title']} | "
            f"{game['module_count']} | {game['source_bits']} | "
            f"{game['k_upper_bits']} | "
            f"{_format_ratio(game['compression_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "## Boundary and interpretation",
            "",
            (
                "The measurement reads the exact Git blobs at the pinned commit "
                "for the path/SHA-256 manifest identified in the JSON artifact. "
                "Working-tree changes cannot affect it. Duplicate judge-dev copies, "
                "sample agents, backups, generated/build artifacts, vendored "
                "libraries, tests, and DeepClue story data are excluded."
            ),
            "",
            (
                "This is implementation-description complexity. It is not "
                "game-tree size, strategic depth, learning difficulty, or "
                "information gain."
            ),
            "",
            "## Method references",
            "",
            (
                "- Cameron Browne and Frederic Maire, "
                "[Evolutionary Game Design]"
                "(https://cambolbro.com/cv/publications/"
                "ciaig-browne-maire-19.pdf)."
            ),
            (
                "- Tom Schaul, Julian Togelius, and Jürgen Schmidhuber, "
                "[Measuring Intelligence through Games]"
                "(https://arxiv.org/abs/1109.1314)."
            ),
            "",
        ]
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
