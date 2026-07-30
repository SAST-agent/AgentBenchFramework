import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_built_wheel_installs_with_ludi_manifest(tmp_path):
    wheel_dir = tmp_path / "wheel"
    install_dir = tmp_path / "installed"
    if importlib.util.find_spec("pip") is not None:
        build_command = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
        ]
    else:
        uv = shutil.which("uv")
        assert uv is not None
        build_command = [
            uv,
            "build",
            "--wheel",
            "--out-dir",
            str(wheel_dir),
        ]
    subprocess.run(
        build_command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheel = next(wheel_dir.glob("agentbench_frame-*.whl"))
    if importlib.util.find_spec("pip") is not None:
        install_command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_dir),
            str(wheel),
        ]
    else:
        install_command = [
            uv,
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_dir),
            str(wheel),
        ]
    subprocess.run(
        install_command,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    code = f"""
import sys
sys.path.insert(0, {str(install_dir)!r})
from agentbench_frame.research.agentbench_catalog import AGENTBENCH_GAME_SPECS
assert len(AGENTBENCH_GAME_SPECS) == 10
assert sum(len(spec.files) for spec in AGENTBENCH_GAME_SPECS) == 149
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
