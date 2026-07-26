import pytest

from agentbench_frame.cli import main


def test_generals_iterate_requires_explicit_roots():
    with pytest.raises(SystemExit) as exc:
        main(["generals", "iterate"])
    assert exc.value.code == 2


def test_generals_help_is_registered(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["generals", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "prepare" in output
    assert "iterate" in output
    assert "calibrate-dev" in output
    assert "iterate-v2" in output
    assert "recover-v2" in output


def test_calibrate_dev_help_exposes_frozen_selection_inputs(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["generals", "calibrate-dev", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--calibration-manifest" in output
    assert "--selection-output" in output
    assert "--parent-run" in output
    assert "--expected-parent-hash" in output


def test_iterate_v2_help_requires_parent_and_calibration_selection(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["generals", "iterate-v2", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--calibration-selection" in output
    assert "--parent-run" in output
    assert "--expected-parent-hash" in output
    assert "--codex-executable" in output


def test_recover_v2_help_requires_failed_run(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["generals", "recover-v2", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--failed-run" in output
    assert "--calibration-selection" in output
    assert "--parent-run" in output
