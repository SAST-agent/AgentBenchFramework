import json
from pathlib import Path


def test_layout_derives_frozen_resources_from_one_agentbench_root(tmp_path):
    from agentbench_frame.games.antwar2.runtime import AntWarLayout

    agentbench = tmp_path / "AgentBench"
    control = tmp_path / "handoff"
    layout = AntWarLayout.from_roots(
        agentbench_root=agentbench,
        build_root=tmp_path / "build",
        positive_control_root=control,
    )

    assert layout.backend_archive == (
        agentbench
        / "backend_sources/corpus/30_antwar2/archives/gamecode_logic__141.zip"
    )
    assert layout.human_manifest == (
        agentbench / "top_algorithms/corpus/30_antwar2_ladder/MANIFEST.tsv"
    )
    assert layout.sdk_root == control / "04_versions/canonical_SDK"
    assert layout.historical_versions_root == control / "04_versions"


def test_historical_candidate_assembly_is_self_contained(tmp_path):
    from agentbench_frame.games.antwar2.runtime import assemble_candidate

    policy = tmp_path / "versions" / "ifelse_v22"
    sdk = tmp_path / "canonical_SDK"
    destination = tmp_path / "candidate"
    policy.mkdir(parents=True)
    sdk.mkdir()
    for name in ("ai.py", "common.py", "main.py", "protocol.py"):
        (policy / name).write_text(f"# {name}\n", encoding="utf-8")
    (sdk / "__init__.py").write_text("", encoding="utf-8")

    manifest = assemble_candidate(
        destination=destination,
        policy_root=policy,
        sdk_root=sdk,
    )

    assert (destination / "SDK/__init__.py").is_file()
    assert (destination / "ai.py").read_text() == "# ai.py\n"
    assert manifest["policy_files"] == ["ai.py", "common.py", "main.py", "protocol.py"]
    assert len(manifest["tree_sha256"]) == 64
    assert json.loads((destination / ".agentbench-package.json").read_text())[
        "tree_sha256"
    ] == manifest["tree_sha256"]


def test_dependency_closure_is_placed_where_historical_delegate_expects(tmp_path):
    from agentbench_frame.games.antwar2.runtime import assemble_candidate

    versions = tmp_path / "versions"
    policy = versions / "ifelse_v239"
    dependency = versions / "ifelse_v107"
    sdk = versions / "canonical_SDK"
    destination = tmp_path / "oracles" / "ifelse_v239"
    for root in (policy, dependency):
        root.mkdir(parents=True)
        for name in ("ai.py", "common.py", "main.py", "protocol.py"):
            (root / name).write_text(f"# {root.name}/{name}\n", encoding="utf-8")
    sdk.mkdir()
    (sdk / "__init__.py").write_text("", encoding="utf-8")

    assemble_candidate(
        destination=destination,
        policy_root=policy,
        sdk_root=sdk,
        dependencies={"ifelse_v107": dependency},
    )

    assert (destination.parent / "ifelse_v107/ai.py").is_file()
    assert not (destination.parent / "ifelse_v107/common.py").exists()


def test_bootstrap_assembly_uses_scaffold_not_historical_policy(tmp_path):
    from agentbench_frame.games.antwar2.runtime import assemble_bootstrap_candidate

    support = tmp_path / "support"
    sdk = tmp_path / "SDK"
    template = tmp_path / "template.py"
    destination = tmp_path / "candidate"
    support.mkdir()
    sdk.mkdir()
    for name in ("common.py", "main.py", "protocol.py"):
        (support / name).write_text(f"# support {name}\n", encoding="utf-8")
    (support / "ai.py").write_text("HISTORICAL = True\n", encoding="utf-8")
    (sdk / "__init__.py").write_text("", encoding="utf-8")
    template.write_text("BOOTSTRAP = True\n", encoding="utf-8")

    assemble_bootstrap_candidate(
        destination=destination,
        support_root=support,
        sdk_root=sdk,
        policy_template=template,
    )

    assert (destination / "ai.py").read_text() == "BOOTSTRAP = True\n"
    assert "HISTORICAL" not in (destination / "ai.py").read_text()


def test_materialize_declared_dependency_next_to_version_object(tmp_path):
    import hashlib

    from agentbench_frame.games.antwar2.runtime import materialize_dependencies

    candidate = tmp_path / "versions" / "objects" / "hash"
    candidate.mkdir(parents=True)
    source = tmp_path / "historical" / "ifelse_v107"
    source.mkdir(parents=True)
    (source / "ai.py").write_text("VALUE = 107\n", encoding="utf-8")
    expected = hashlib.sha256((source / "ai.py").read_bytes()).hexdigest()
    (candidate / ".agentbench-package.json").write_text(
        json.dumps({"dependencies": {"ifelse_v107": expected}}),
        encoding="utf-8",
    )

    materialize_dependencies(
        candidate_root=candidate,
        historical_versions_root=tmp_path / "historical",
    )

    assert (candidate.parent / "ifelse_v107" / "ai.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 107\n"
