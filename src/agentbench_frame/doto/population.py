"""Validated historical DOTO population manifest and isolated builder."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal

from .assets import official_server_dir
from .decision_space import canonicalize_action
from .process import ManagedProcess
from .protocol import read_ai_frame, write_ai_observation


@dataclass(frozen=True)
class PopulationPolicy:
    name: str
    source: Path
    origin: str
    split: Literal["train", "test"]
    expected_sha256: str
    leaderboard: bool
    enabled: bool = True
    policy_id: str = ""
    origin_split: str = ""

    @property
    def source_sha256(self) -> str:
        """Canonical schema-v2 name for the complete-source-tree digest."""

        return self.expected_sha256


@dataclass(frozen=True)
class PopulationManifest:
    schema_version: int
    benchmark_version: str
    game: str
    seed: int
    seats: tuple[int, ...]
    policies: tuple[PopulationPolicy, ...]

    def __iter__(self) -> Iterator[PopulationPolicy]:
        """Keep legacy callers that consumed a policy list working."""

        return iter(self.policies)

    def __len__(self) -> int:
        return len(self.policies)

    def __getitem__(self, index: int) -> PopulationPolicy:
        return self.policies[index]


@dataclass(frozen=True)
class BuiltPolicy:
    policy_id: str
    name: str
    executable: Path
    executable_sha256: str


@dataclass(frozen=True)
class PopulationBundle:
    benchmark_version: str
    split: Literal["train", "test"]
    seed: int
    seats: tuple[int, ...]
    policies: tuple[BuiltPolicy, ...]
    bundle_sha256: str | None = None


@dataclass(frozen=True)
class PopulationBuildRow:
    policy_id: str
    name: str
    status: str
    policy_sha256: str
    executable_sha256: str | None
    error: str | None = None


@dataclass(frozen=True)
class PopulationBuildResult:
    benchmark_version: str
    split: Literal["train", "test"]
    output_dir: Path
    policies: tuple[PopulationBuildRow, ...]

    def to_json(self) -> dict:
        return {
            "benchmark_version": self.benchmark_version,
            "split": self.split,
            "policies": [asdict(policy) for policy in self.policies],
        }


def load_population(path: Path) -> PopulationManifest:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    schema_version = raw.get("schema_version")
    benchmark_version = raw.get("benchmark_version")
    game = raw.get("game")
    seed = raw.get("seed")
    seats = tuple(raw.get("seats", ()))
    if schema_version != 2:
        raise ValueError("population schema_version must be 2")
    if not isinstance(benchmark_version, str) or not benchmark_version.strip():
        raise ValueError("population benchmark_version must be nonempty")
    if game != "23_doto" or seed != 11 or seats != (0, 1):
        raise ValueError("population must declare game 23_doto, seed 11, and seats [0, 1]")
    rows = raw.get("policies")
    if not isinstance(rows, list) or not rows:
        raise ValueError("population policies must be nonempty")
    policies: list[PopulationPolicy] = []
    names: set[str] = set()
    policy_ids: set[str] = set()
    hashes: set[str] = set()
    for index, row in enumerate(rows):
        name = str(row.get("name", "")).strip()
        policy_id = str(row.get("policy_id", f"doto-{index:02d}-{name.lower()}"))
        origin_split = str(row.get("origin_split", row.get("split", ""))).strip()
        split = "train" if origin_split == "train" else "test"
        expected_sha256 = str(row.get("source_sha256", row.get("sha256", ""))).lower()
        policy = PopulationPolicy(
            name=name,
            source=Path(str(row.get("source", ""))),
            origin=str(row.get("origin", "")).strip(),
            split=split,
            expected_sha256=expected_sha256,
            leaderboard=row.get("leaderboard"),
            enabled=row.get("enabled", True),
            policy_id=policy_id,
            origin_split=origin_split,
        )
        if not policy.name or policy.name in names or not policy.policy_id or policy.policy_id in policy_ids:
            raise ValueError(f"invalid or duplicate population identity: {policy.name!r}/{policy.policy_id!r}")
        if policy.source.is_absolute() or ".." in policy.source.parts:
            raise ValueError(f"population source must be relative: {policy.source}")
        if policy.origin_split not in ("train", "validation", "test"):
            raise ValueError(f"invalid population origin split: {policy.origin_split}")
        if len(policy.expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in policy.expected_sha256):
            raise ValueError(f"invalid expected_sha256 for {policy.name}")
        if policy.expected_sha256 in hashes:
            raise ValueError(f"duplicate complete-source hash for {policy.name}")
        if not isinstance(policy.enabled, bool) or not isinstance(policy.leaderboard, bool):
            raise ValueError(f"enabled and leaderboard must be boolean for {policy.name}")
        names.add(policy.name)
        policy_ids.add(policy.policy_id)
        hashes.add(policy.expected_sha256)
        policies.append(policy)
    if len(policies) != 43:
        raise ValueError("population must contain exactly 43 policies")
    if sum(policy.split == "train" for policy in policies) != 15:
        raise ValueError("population must contain exactly 15 training policies")
    if sum(policy.split == "test" for policy in policies) != 28:
        raise ValueError("population must contain exactly 28 hidden test policies")
    return PopulationManifest(
        schema_version=schema_version,
        benchmark_version=benchmark_version,
        game=game,
        seed=seed,
        seats=seats,
        policies=tuple(policies),
    )


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(item for item in Path(path).rglob("*") if item.is_file()):
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(file.read_bytes() + b"\0")
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_population(corpus_root: Path, manifest: PopulationManifest) -> dict:
    """Verify all declared complete snapshots without exposing their contents."""

    root = Path(corpus_root)
    mismatches: list[str] = []
    missing: list[str] = []
    verified = 0
    for policy in manifest.policies:
        directory = root / policy.source
        if not directory.is_dir():
            missing.append(policy.policy_id)
        elif source_hash(directory) != policy.source_sha256:
            mismatches.append(policy.policy_id)
        else:
            verified += 1
    return {
        "benchmark_version": manifest.benchmark_version,
        "declared": len(manifest.policies),
        "verified": verified,
        "train": sum(policy.split == "train" for policy in manifest.policies),
        "test": sum(policy.split == "test" for policy in manifest.policies),
        "missing": missing,
        "hash_mismatches": mismatches,
    }


def materialize_training_sources(corpus_root: Path, manifest: PopulationManifest,
                                 destination: Path) -> dict:
    """Copy only verified training snapshots into a public Framework directory."""

    root = Path(corpus_root)
    destination = Path(destination)
    training = tuple(policy for policy in manifest.policies if policy.split == "train")
    for policy in training:
        directory = root / policy.source
        if not directory.is_dir() or source_hash(directory) != policy.source_sha256:
            raise ValueError(f"training policy failed identity verification: {policy.policy_id}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for policy in training:
            shutil.copytree(root / policy.source, temporary / policy.policy_id)
        metadata = {
            "schema_version": 1,
            "benchmark_version": manifest.benchmark_version,
            "policy_ids": [policy.policy_id for policy in training],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "benchmark_version": manifest.benchmark_version,
        "copied_policy_ids": [policy.policy_id for policy in training],
    }


def build_sealed_test_bundle(corpus_root: Path, manifest: PopulationManifest,
                             destination: Path) -> dict:
    """Build hidden policies into an evaluator-owned runtime-only bundle."""

    root = Path(corpus_root)
    destination = Path(destination)
    hidden = tuple(policy for policy in manifest.policies if policy.split == "test")
    for policy in hidden:
        directory = root / policy.source
        if not directory.is_dir() or source_hash(directory) != policy.source_sha256:
            raise ValueError(f"hidden policy failed identity verification: {policy.policy_id}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    build_root = temporary / ".build"
    runtime_root = temporary / "policies"
    runtime_rows: list[dict] = []
    public_rows: list[dict] = []
    try:
        for policy in hidden:
            build_dir = build_root / policy.policy_id
            shutil.copytree(root / policy.source, build_dir)
            shutil.copytree(official_server_dir() / "Maps", build_dir / "Maps")
            completed = subprocess.run(
                ["make"], cwd=build_dir, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            executable = build_dir / "main.out"
            smoke_ok, _ = _protocol_smoke(executable) if completed.returncode == 0 and executable.is_file() else (False, "")
            status = "ready" if smoke_ok else "protocol_smoke_failed" if executable.is_file() else "build_failed"
            public_rows.append({"policy_id": policy.policy_id, "status": status})
            if status == "ready":
                runtime_dir = runtime_root / policy.policy_id
                runtime_dir.mkdir(parents=True)
                runtime_executable = runtime_dir / "main.out"
                shutil.copy2(executable, runtime_executable)
                shutil.copytree(build_dir / "Maps", runtime_dir / "Maps")
                runtime_rows.append({
                    "policy_id": policy.policy_id,
                    "name": policy.name,
                    "path": f"policies/{policy.policy_id}/main.out",
                    "sha256": file_hash(runtime_executable),
                })
        shutil.rmtree(build_root, ignore_errors=True)
        sealed_manifest = {
            "schema_version": 1,
            "benchmark_version": manifest.benchmark_version,
            "split": "test",
            "seed": manifest.seed,
            "seats": list(manifest.seats),
            "policies": runtime_rows,
        }
        (temporary / "sealed-manifest.json").write_text(
            json.dumps(sealed_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "benchmark_version": manifest.benchmark_version,
        "declared_test_policies": len(hidden),
        "ready": sum(row["status"] == "ready" for row in public_rows),
        "policies": public_rows,
    }


def _load_runtime_bundle(path: Path, manifest_name: str, expected_split: str,
                         benchmark_version: str | None = None) -> PopulationBundle:
    root = Path(path)
    raw = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    if benchmark_version is not None and raw.get("benchmark_version") != benchmark_version:
        raise ValueError("population bundle benchmark version mismatch")
    if raw.get("split") != expected_split:
        raise ValueError(f"expected {expected_split} population bundle")
    policies: list[BuiltPolicy] = []
    for row in raw.get("policies", []):
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("sealed bundle path must remain relative")
        executable = root / relative
        expected = str(row.get("sha256", ""))
        if not executable.is_file() or file_hash(executable) != expected:
            raise ValueError(f"sealed policy integrity failure: {row.get('policy_id')}")
        policies.append(BuiltPolicy(str(row["policy_id"]), str(row["name"]), executable, expected))
    return PopulationBundle(
        benchmark_version=str(raw["benchmark_version"]),
        split=expected_split,
        seed=int(raw["seed"]),
        seats=tuple(int(seat) for seat in raw["seats"]),
        policies=tuple(policies),
        bundle_sha256=source_hash(root),
    )


def load_sealed_bundle(path: Path, benchmark_version: str | None = None) -> PopulationBundle:
    return _load_runtime_bundle(path, "sealed-manifest.json", "test", benchmark_version)


def load_training_bundle(path: Path, benchmark_version: str | None = None) -> PopulationBundle:
    return _load_runtime_bundle(path, "bundle-manifest.json", "train", benchmark_version)


def build_training_bundle(manifest: PopulationManifest, source_root: Path,
                          output_dir: Path) -> PopulationBuildResult:
    """Build every public training snapshot into an isolated runtime bundle."""

    source_root = Path(source_root)
    if source_root.is_file():
        temporary_sources = Path(tempfile.mkdtemp(prefix="doto-training-sources-"))
        try:
            with tarfile.open(source_root, "r:gz") as archive:
                for member in archive.getmembers():
                    target = (temporary_sources / member.name).resolve()
                    if (temporary_sources.resolve() not in target.parents
                            and target != temporary_sources.resolve()):
                        raise ValueError("training source archive contains an unsafe path")
                    if not (member.isfile() or member.isdir()):
                        raise ValueError(
                            "training source archive must contain only files and directories"
                        )
                archive.extractall(temporary_sources)
            return build_training_bundle(manifest, temporary_sources, output_dir)
        except tarfile.TarError as exc:
            raise ValueError("invalid training source archive") from exc
        finally:
            shutil.rmtree(temporary_sources, ignore_errors=True)
    output_dir = Path(output_dir)
    training = tuple(policy for policy in manifest.policies if policy.split == "train")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    build_root = temporary / ".build"
    runtime_root = temporary / "policies"
    rows: list[PopulationBuildRow] = []
    bundle_rows: list[dict] = []
    try:
        for policy in training:
            public_source = source_root / policy.policy_id
            if not public_source.is_dir():
                public_source = source_root / policy.source
            actual_hash = source_hash(public_source) if public_source.is_dir() else ""
            if actual_hash != policy.source_sha256:
                rows.append(PopulationBuildRow(
                    policy.policy_id, policy.name, "hash_mismatch",
                    actual_hash, None, "complete-source hash mismatch",
                ))
                continue
            build_dir = build_root / policy.policy_id
            shutil.copytree(public_source, build_dir)
            shutil.copytree(official_server_dir() / "Maps", build_dir / "Maps")
            completed = subprocess.run(
                ["make"], cwd=build_dir, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            executable = build_dir / "main.out"
            smoke_ok, smoke_error = (
                _protocol_smoke(executable)
                if completed.returncode == 0 and executable.is_file()
                else (False, "")
            )
            status = "ready" if smoke_ok else "protocol_smoke_failed" if executable.is_file() else "build_failed"
            executable_sha256 = None
            if status == "ready":
                runtime_dir = runtime_root / policy.policy_id
                runtime_dir.mkdir(parents=True)
                runtime_executable = runtime_dir / "main.out"
                shutil.copy2(executable, runtime_executable)
                shutil.copytree(build_dir / "Maps", runtime_dir / "Maps")
                executable_sha256 = file_hash(runtime_executable)
                bundle_rows.append({
                    "policy_id": policy.policy_id,
                    "name": policy.name,
                    "path": f"policies/{policy.policy_id}/main.out",
                    "sha256": executable_sha256,
                })
            rows.append(PopulationBuildRow(
                policy.policy_id, policy.name, status, actual_hash,
                executable_sha256, smoke_error or None,
            ))
        shutil.rmtree(build_root, ignore_errors=True)
        bundle_manifest = {
            "schema_version": 1,
            "benchmark_version": manifest.benchmark_version,
            "split": "train",
            "seed": manifest.seed,
            "seats": list(manifest.seats),
            "policies": bundle_rows,
        }
        (temporary / "bundle-manifest.json").write_text(
            json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return PopulationBuildResult(
        manifest.benchmark_version, "train", output_dir.resolve(), tuple(rows)
    )


def _protocol_smoke(executable: Path, timeout: float = 2.0) -> tuple[bool, str]:
    process = ManagedProcess.start([str(executable)], cwd=executable.parent, label="population-smoke")
    humans = [[human_id, 10.0, 175.0, 100, 1, 0, 1, 0, 0, -1, 0]
              for human_id in range(10)]
    try:
        init = {"frame": 0, "map": 0, "faction": 0}
        frame = {"frame": 1, "humans": humans, "fireballs": [], "meteors": [],
                 "balls": [[20, 20, -1, 0], [30, 30, -1, 1]],
                 "scores": [0, 0], "bonus": [0, 0]}
        write_ai_observation(process.stdin, json.dumps(init, separators=(",", ":")).encode())
        write_ai_observation(process.stdin, json.dumps(frame, separators=(",", ":")).encode())
        canonicalize_action(json.loads(read_ai_frame(process.stdout, timeout, "population-smoke")))
        return True, ""
    except Exception as exc:
        return False, str(exc)
    finally:
        process.terminate()


def build_population(policies: list[PopulationPolicy] | PopulationManifest,
                     corpus_root: Path, output_dir: Path) -> dict:
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
            compiled = completed.returncode == 0 and executable.is_file()
            smoke_ok, smoke_error = _protocol_smoke(executable) if compiled else (False, "")
            status = "ready" if smoke_ok else "protocol_smoke_failed" if compiled else "build_failed"
            row.update(status=status, exit_code=completed.returncode, stdout=completed.stdout,
                       stderr=completed.stderr, protocol_smoke_error=smoke_error or None,
                       executable=str(executable) if status == "ready" else None)
        rows.append(row)
    report = {"game": "23_doto", "policies": rows}
    temporary = output_dir / ".population-build.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, output_dir / "population-build.json")
    return report
