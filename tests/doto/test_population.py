from pathlib import Path

from agentbench_frame.doto.population import PopulationPolicy, build_population, load_population, source_hash


MANIFEST = Path(__file__).parents[2] / "src/agentbench_frame/doto/population.toml"


def test_population_preserves_declared_splits_and_hashes():
    policies = load_population(MANIFEST)
    assert {policy.split for policy in policies} >= {"train", "validation"}
    assert all(len(policy.expected_sha256) == 64 for policy in policies)


def test_failed_policy_remains_in_build_report(tmp_path):
    source = tmp_path / "corpus/broken"
    source.mkdir(parents=True)
    (source / "makefile").write_text("all:\n\tfalse\n")
    policy = PopulationPolicy("broken", Path("broken"), "fixture", "train",
                              source_hash(source), "fixture", True)
    report = build_population([policy], tmp_path / "corpus", tmp_path / "output")
    assert report["policies"][0]["status"] == "build_failed"
    assert report["policies"][0]["executable"] is None
