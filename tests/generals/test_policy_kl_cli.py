import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agentbench_frame.generals.cli as cli
from agentbench_frame.generals.cli import register_parser


def parser():
    root = argparse.ArgumentParser()
    register_parser(root.add_subparsers(dest="command"))
    return root


def arguments(command):
    values = [
        "generals",
        command,
        "--agentbench-root",
        "/assets",
        "--manifest",
        "/benchmark/pilot.toml",
        "--reference-manifest",
        "/benchmark/policy-kl-reference.toml",
        "--data-dir",
        "/data",
        "--count-wall-time",
        "90",
        "--count-max-states",
        "1234",
    ]
    if command == "recover-policy-kl":
        values.extend(["--failed-run", "/data/runs/measurement"])
    return values


def extension_arguments(command):
    values = [
        "generals",
        command,
        "--agentbench-root",
        "/assets",
        "--manifest",
        "/benchmark/pilot.toml",
        "--reference-manifest",
        "/benchmark/policy-kl-reference-v2.toml",
        "--source-run",
        "/data/runs/v1",
        "--data-dir",
        "/data",
    ]
    if command == "recover-policy-kl-v7":
        values.extend(["--failed-run", "/data/runs/v2-incomplete"])
    return values


def test_measure_policy_kl_cli_exposes_required_inputs():
    args = parser().parse_args(arguments("measure-policy-kl"))

    assert args.agentbench_root == Path("/assets")
    assert args.reference_manifest == Path(
        "/benchmark/policy-kl-reference.toml"
    )
    assert args.data_dir == Path("/data")
    assert args.count_wall_time == 90
    assert args.count_max_states == 1234


def test_recover_policy_kl_requires_failed_run():
    values = arguments("recover-policy-kl")
    index = values.index("--failed-run")
    del values[index:index + 2]

    with pytest.raises(SystemExit) as stopped:
        parser().parse_args(values)

    assert stopped.value.code == 2


@pytest.mark.parametrize("command", ("measure-policy-kl", "recover-policy-kl"))
def test_policy_kl_cli_routes_without_constructing_a_codex_provider(
    monkeypatch,
    capsys,
    command,
):
    args = parser().parse_args(arguments(command))
    captured = {}

    class FakePipeline:
        @classmethod
        def from_paths(cls, **kwargs):
            captured.update(kwargs)
            return cls()

        def run(self):
            captured["method"] = "run"
            return SimpleNamespace(
                status="complete",
                run_dir=Path("/data/runs/measurement"),
                summary={"controlled_reference_policy_kl": {}},
            )

        def recover(self, failed_run):
            captured["method"] = "recover"
            captured["failed_run"] = failed_run
            return self.run()

    monkeypatch.setattr(cli, "_assets", lambda args: (object(), object()))
    monkeypatch.setattr(cli, "GeneralsPolicyKLPipeline", FakePipeline)
    monkeypatch.setattr(
        cli,
        "CodexProvider",
        lambda **kwargs: pytest.fail(
            "measurement must not construct a coding provider"
        ),
    )

    assert cli.handle(args) == 0
    assert captured["reference_manifest_path"] == Path(
        "/benchmark/policy-kl-reference.toml"
    )
    assert captured["count_wall_time_s"] == 90
    assert captured["count_max_states"] == 1234
    if command == "recover-policy-kl":
        assert captured["failed_run"] == Path("/data/runs/measurement")
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "complete"


@pytest.mark.parametrize(
    "command",
    ("extend-policy-kl-v7", "recover-policy-kl-v7"),
)
def test_policy_kl_v7_extension_cli_requires_explicit_source(command):
    args = parser().parse_args(extension_arguments(command))

    assert args.reference_manifest == Path(
        "/benchmark/policy-kl-reference-v2.toml"
    )
    assert args.source_run == Path("/data/runs/v1")
    assert args.data_dir == Path("/data")


def test_recover_policy_kl_v7_requires_failed_run():
    values = extension_arguments("recover-policy-kl-v7")
    index = values.index("--failed-run")
    del values[index:index + 2]

    with pytest.raises(SystemExit) as stopped:
        parser().parse_args(values)

    assert stopped.value.code == 2


@pytest.mark.parametrize(
    "command",
    ("extend-policy-kl-v7", "recover-policy-kl-v7"),
)
def test_policy_kl_v7_extension_routes_without_a_coding_provider(
    monkeypatch,
    capsys,
    command,
):
    args = parser().parse_args(extension_arguments(command))
    captured = {}

    class FakeExtensionPipeline:
        @classmethod
        def from_paths(cls, **kwargs):
            captured.update(kwargs)
            return cls()

        def run(self):
            captured["method"] = "run"
            return SimpleNamespace(
                status="complete",
                run_dir=Path("/data/runs/v2"),
                summary={
                    "source_run_id": "source-v1",
                    "controlled_reference_policy_kl": {},
                },
            )

        def recover(self, failed_run):
            captured["method"] = "recover"
            captured["failed_run"] = failed_run
            return self.run()

    monkeypatch.setattr(cli, "_assets", lambda args: (object(), object()))
    monkeypatch.setattr(
        cli,
        "GeneralsPolicyKLExtensionPipeline",
        FakeExtensionPipeline,
    )
    monkeypatch.setattr(
        cli,
        "CodexProvider",
        lambda **kwargs: pytest.fail(
            "policy KL extension must not construct a coding provider"
        ),
    )

    assert cli.handle(args) == 0
    assert captured["reference_manifest_path"] == Path(
        "/benchmark/policy-kl-reference-v2.toml"
    )
    assert captured["source_run_dir"] == Path("/data/runs/v1")
    if command == "recover-policy-kl-v7":
        assert captured["failed_run"] == Path("/data/runs/v2-incomplete")
    else:
        assert captured["method"] == "run"
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "controlled_reference_policy_kl": {},
        "run_dir": "/data/runs/v2",
        "source_run_id": "source-v1",
        "status": "complete",
    }
