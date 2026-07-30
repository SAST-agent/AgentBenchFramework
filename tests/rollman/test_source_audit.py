import json
from pathlib import Path


AGENTBENCH_ROOT = Path("/Users/qingle/Code/SAST/AgentBench")
OFFICIAL_LOGIC_ROOT = Path("/Users/qingle/Code/SAST/PacmanLogic")


def test_frozen_core_matches_pinned_official_logic_core():
    from agentbench_frame.games.rollman.contract import audit_sources

    result = audit_sources(AGENTBENCH_ROOT, OFFICIAL_LOGIC_ROOT)

    assert result["all_core_files_match"] is True
    assert result["frozen_backend_commit"] == "b581bca3ba3d2d7d58a2f8c6bbddd060fc7fdc87"
    assert result["official_logic_commit"] == "81d0468d177089cefe1f08ed1cffe78beb0d27e9"
    assert result["official_core_commit"] == "b293c04746fc3bf9b67a00130a2ca15fc38691bc"
    assert result["random_seed_is_applied"] is False
    assert result["adapter_seed_injection_required"] is True


def test_source_manifest_hashes_match_frozen_backend():
    from agentbench_frame.games.rollman.contract import asset_path, sha256_file

    manifest = json.loads(asset_path("source_manifest.json").read_text(encoding="utf-8"))
    backend = (
        AGENTBENCH_ROOT
        / "backend_sources/corpus/29_rollman/logic/gamecode_logic/PacmanLogic"
    )

    for relative, expected in manifest["frozen_backend_sha256"].items():
        assert sha256_file(backend / relative) == expected
