"""Validated historical DOTO population manifest and isolated builder."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PopulationPolicy:
    name: str
    source: Path
    origin: str
    split: str
    expected_sha256: str
    leaderboard: str
    enabled: bool = True


def load_population(path: Path) -> list[PopulationPolicy]:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    rows = raw.get("policies")
    if not isinstance(rows, list) or not rows:
        raise ValueError("population policies must be nonempty")
    policies, names = [], set()
    for row in rows:
        policy = PopulationPolicy(str(row.get("name", "")).strip(), Path(str(row.get("source", ""))),
                                  str(row.get("origin", "")).strip(), str(row.get("split", "")).strip(),
                                  str(row.get("expected_sha256", "")).lower(),
                                  str(row.get("leaderboard", "")).strip(), row.get("enabled", True))
        if not policy.name or policy.name in names:
            raise ValueError(f"invalid or duplicate population name: {policy.name!r}")
        if policy.source.is_absolute() or ".." in policy.source.parts:
            raise ValueError(f"population source must be relative: {policy.source}")
        if policy.split not in ("train", "validation", "test"):
            raise ValueError(f"invalid population split: {policy.split}")
        if len(policy.expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in policy.expected_sha256):
            raise ValueError(f"invalid expected_sha256 for {policy.name}")
        if not isinstance(policy.enabled, bool):
            raise ValueError(f"enabled must be boolean for {policy.name}")
        names.add(policy.name)
        policies.append(policy)
    return policies


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(item for item in Path(path).rglob("*") if item.is_file()):
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(file.read_bytes() + b"\0")
    return digest.hexdigest()


def build_population(policies: list[PopulationPolicy], corpus_root: Path, output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy in policies:
        row = {**asdict(policy), "source": str(policy.source), "status": "disabled"}
        source = Path(corpus_root) / policy.source
        actual = source_hash(source) if source.is_dir() else None
        row["actual_sha256"] = actual
        target = output_dir / policy.name
        if policy.enabled and actual != policy.expected_sha256:
            row.update(status="hash_mismatch", stderr="source hash does not match manifest")
        elif policy.enabled:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            completed = subprocess.run(["make"], cwd=target, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            executable = target / "main.out"
            status = "ready" if completed.returncode == 0 and executable.is_file() else "build_failed"
            row.update(status=status, exit_code=completed.returncode, stdout=completed.stdout,
                       stderr=completed.stderr, executable=str(executable) if status == "ready" else None)
        rows.append(row)
    report = {"game": "23_doto", "policies": rows}
    temporary = output_dir / ".population-build.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, output_dir / "population-build.json")
    return report
