from agentbench_frame.doto.assets import official_server_dir, sdk_dir, verify_assets


def test_vendored_assets_match_recorded_hashes():
    hashes = verify_assets()

    assert "official_server/main.py" in hashes
    assert "official_server/Maps/0.json" in hashes
    assert "sdk/main.cpp" in hashes
    assert "sdk/playerAI.cpp" not in hashes
    assert official_server_dir().is_dir()
    assert sdk_dir().is_dir()
