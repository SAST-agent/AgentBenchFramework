import json
from pathlib import Path

import pytest

from agentbench_frame.doto.population import (
    PopulationManifest,
    PopulationPolicy,
    audit_population,
    build_population,
    build_sealed_test_bundle,
    build_training_bundle,
    load_population,
    load_sealed_bundle,
    load_training_bundle,
    materialize_training_sources,
    source_hash,
)


MANIFEST = Path(__file__).parents[2] / "src/agentbench_frame/doto/population.toml"


def test_population_preserves_declared_splits_and_hashes():
    policies = load_population(MANIFEST)
    assert {policy.split for policy in policies} == {"train", "test"}
    assert {policy.origin_split for policy in policies} >= {"train", "validation", "test"}
    assert all(len(policy.source_sha256) == 64 for policy in policies)


def test_manifest_freezes_43_distinct_complete_snapshots():
    manifest = load_population(MANIFEST)
    assert manifest.schema_version == 2
    assert manifest.game == "23_doto"
    assert manifest.seed == 11
    assert manifest.seats == (0, 1)
    assert len(manifest.policies) == 43
    assert len({policy.source_sha256 for policy in manifest.policies}) == 43
    assert sum(policy.split == "train" for policy in manifest.policies) == 15
    assert sum(policy.split == "test" for policy in manifest.policies) == 28


def test_auxiliary_source_changes_policy_identity(tmp_path):
    (tmp_path / "playerAI.cpp").write_text("same")
    (tmp_path / "Attack.cpp").write_text("a")
    before = source_hash(tmp_path)
    (tmp_path / "Attack.cpp").write_text("b")
    assert source_hash(tmp_path) != before


def test_failed_policy_remains_in_build_report(tmp_path):
    source = tmp_path / "corpus/broken"
    source.mkdir(parents=True)
    (source / "makefile").write_text("all:\n\tfalse\n")
    policy = PopulationPolicy("broken", Path("broken"), "fixture", "train",
                              source_hash(source), "fixture", True)
    report = build_population([policy], tmp_path / "corpus", tmp_path / "output")
    assert report["policies"][0]["status"] == "build_failed"
    assert report["policies"][0]["executable"] is None


def _tiny_manifest(policy: PopulationPolicy) -> PopulationManifest:
    return PopulationManifest(2, "fixture-v1", "23_doto", 11, (0, 1), (policy,))


def test_audit_reports_hash_mismatch_without_hiding_policy(tmp_path):
    root = tmp_path / "corpus"
    source = root / "policy"
    source.mkdir(parents=True)
    (source / "playerAI.cpp").write_text("original")
    policy = PopulationPolicy("fixture", Path("policy"), "fixture", "train",
                              "0" * 64, False, policy_id="fixture", origin_split="train")
    report = audit_population(root, _tiny_manifest(policy))
    assert report["verified"] == 0
    assert report["hash_mismatches"] == ["fixture"]


def test_materialize_training_never_copies_test_source(tmp_path):
    root = tmp_path / "corpus"
    train = root / "train"
    test = root / "test"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    (train / "playerAI.cpp").write_text("train")
    (test / "playerAI.cpp").write_text("secret")
    policies = (
        PopulationPolicy("train", Path("train"), "fixture", "train", source_hash(train),
                         False, policy_id="train", origin_split="train"),
        PopulationPolicy("test", Path("test"), "fixture", "test", source_hash(test),
                         False, policy_id="test", origin_split="test"),
    )
    manifest = PopulationManifest(2, "fixture-v1", "23_doto", 11, (0, 1), policies)
    report = materialize_training_sources(root, manifest, tmp_path / "public")
    assert report["copied_policy_ids"] == ["train"]
    assert (tmp_path / "public/train/playerAI.cpp").read_text() == "train"
    assert not (tmp_path / "public/test").exists()


def test_sealed_build_report_has_no_paths_or_source(tmp_path):
    root = tmp_path / "corpus"
    policy_dir = root / "hidden"
    policy_dir.mkdir(parents=True)
    fake_ai = Path(__file__).parent / "fixtures/fake_ai.py"
    (policy_dir / "fake_ai.py").write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "if not Path('Maps/0.json').is_file():\n"
        "    raise SystemExit('official map missing')\n"
        + fake_ai.read_text()
    )
    (policy_dir / "makefile").write_text(
        "all:\n\tcp fake_ai.py main.out\n\tchmod +x main.out\n"
    )
    policy = PopulationPolicy("hidden", Path("hidden"), "fixture", "test",
                              source_hash(policy_dir), False,
                              policy_id="hidden", origin_split="test")
    report = build_sealed_test_bundle(root, _tiny_manifest(policy), tmp_path / "sealed")
    serialized = json.dumps(report)
    assert "source" not in serialized
    assert "executable" not in serialized
    assert str(tmp_path) not in serialized
    assert report["policies"] == [{"policy_id": "hidden", "status": "ready"}]


def test_importer_audit_command_reports_complete_real_corpus(capsys):
    from scripts.import_doto_population import main

    corpus = (Path(__file__).parents[2]
              / "../AgentBench/backend_sources/corpus/23_doto").resolve()
    if not corpus.is_dir():
        pytest.skip("external AgentBench DOTO corpus is not available")
    assert main(["audit", "--corpus-root", str(corpus), "--manifest", str(MANIFEST)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verified"] == 43
    assert report["hash_mismatches"] == []


def test_training_bundle_builds_and_hashes_every_policy(tmp_path):
    root = tmp_path / "public"
    policy_dir = root / "train"
    policy_dir.mkdir(parents=True)
    fake_ai = Path(__file__).parent / "fixtures/fake_ai.py"
    (policy_dir / "fake_ai.py").write_text("#!/usr/bin/env python3\n" + fake_ai.read_text())
    (policy_dir / "makefile").write_text(
        "all:\n\tcp fake_ai.py main.out\n\tchmod +x main.out\n"
    )
    policy = PopulationPolicy("train", Path("train"), "fixture", "train",
                              source_hash(policy_dir), False,
                              policy_id="train", origin_split="train")
    result = build_training_bundle(_tiny_manifest(policy), root, tmp_path / "bundle")
    assert [row.status for row in result.policies] == ["ready"]
    assert all(row.executable_sha256 for row in result.policies)
    assert (tmp_path / "bundle/policies/train/Maps/0.json").is_file()
    loaded = load_training_bundle(tmp_path / "bundle", "fixture-v1")
    assert [policy.policy_id for policy in loaded.policies] == ["train"]


def test_sealed_bundle_rejects_wrong_benchmark_version(tmp_path):
    root = tmp_path / "corpus"
    policy_dir = root / "hidden"
    policy_dir.mkdir(parents=True)
    fake_ai = Path(__file__).parent / "fixtures/fake_ai.py"
    (policy_dir / "fake_ai.py").write_text("#!/usr/bin/env python3\n" + fake_ai.read_text())
    (policy_dir / "makefile").write_text(
        "all:\n\tcp fake_ai.py main.out\n\tchmod +x main.out\n"
    )
    policy = PopulationPolicy("hidden", Path("hidden"), "fixture", "test",
                              source_hash(policy_dir), False,
                              policy_id="hidden", origin_split="test")
    build_sealed_test_bundle(root, _tiny_manifest(policy), tmp_path / "sealed")
    try:
        load_sealed_bundle(tmp_path / "sealed", "wrong")
    except ValueError as error:
        assert "benchmark version" in str(error)
    else:
        raise AssertionError("wrong benchmark version was accepted")
