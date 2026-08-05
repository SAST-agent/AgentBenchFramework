"""Audit and isolate the historical 23_doto human-policy population."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from agentbench_frame.doto.population import (
    audit_population,
    build_sealed_test_bundle,
    load_population,
    materialize_training_sources,
)


DEFAULT_MANIFEST = (
    Path(__file__).parents[1] / "src/agentbench_frame/doto/population.toml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maintain isolated DOTO human-policy pools")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "materialize-training", "build-sealed"):
        command = subcommands.add_parser(name)
        command.add_argument("--corpus-root", type=Path, required=True)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        if name != "audit":
            command.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_population(args.manifest)
    if args.command == "audit":
        report = audit_population(args.corpus_root, manifest)
        success = not report["missing"] and not report["hash_mismatches"]
    elif args.command == "materialize-training":
        report = materialize_training_sources(args.corpus_root, manifest, args.destination)
        success = len(report["copied_policy_ids"]) == sum(
            policy.split == "train" for policy in manifest.policies
        )
    else:
        report = build_sealed_test_bundle(args.corpus_root, manifest, args.destination)
        success = report["ready"] == report["declared_test_policies"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
