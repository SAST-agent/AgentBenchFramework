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
