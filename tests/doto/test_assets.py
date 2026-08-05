import tomllib
from pathlib import Path

from agentbench_frame.doto.assets import official_server_dir, sdk_dir, verify_assets


def test_vendored_assets_match_recorded_hashes():
    hashes = verify_assets()

    assert "official_server/main.py" in hashes
    assert "official_server/Maps/0.json" in hashes
    assert "sdk/main.cpp" in hashes
    assert "sdk/playerAI.cpp" not in hashes
    assert official_server_dir().is_dir()
    assert sdk_dir().is_dir()


def test_doto_extra_installs_official_server_runtime_dependencies():
    root = Path(__file__).parents[2]
    with (root / "pyproject.toml").open("rb") as stream:
        dependencies = tomllib.load(stream)["project"]["optional-dependencies"]["doto"]
    assert any(item.split("=", 1)[0].split(">", 1)[0] == "numpy" for item in dependencies)
