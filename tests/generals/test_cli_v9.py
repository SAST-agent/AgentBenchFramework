import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import agentbench_frame.generals.cli as cli
from agentbench_frame.generals.cli import register_parser


def parser():
    root = argparse.ArgumentParser()
    register_parser(root.add_subparsers(dest="command"))
    return root


def common(command):
    return [
        "generals", command,
        "--agentbench-root", "/assets",
        "--manifest", "/benchmark/pilot.toml",
        "--data-dir", "/data",
    ]


def v9_args(command):
    values = common(command) + [
        "--challenge-manifest", "/benchmark/v9.toml",
        "--replay-skill", "skills/replay/SKILL.md",
        "--parent-run", "/runs/v7",
        "--predecessor-run", "/runs/v8",
        "--attribution-run", "/runs/attribution",
        "--expected-parent-hash", "7" * 64,
        "--expected-predecessor-hash", "8" * 64,
    ]
    if command == "iterate-v9":
        values += ["--codex-executable", "codex", "--provider-timeout", "1800"]
    else:
        values += ["--failed-run", "/runs/failed-v9"]
    return values


def expanded_args(command):
    values = common(command) + [
        "--reference-manifest", "/benchmark/expanded.toml",
        "--source-run", "/runs/legacy-kl",
        "--target-run", "/runs/v9",
        "--expected-target-hash", "9" * 64,
        "--count-wall-time", "90",
        "--count-max-states", "1234",
    ]
    if command.startswith("recover"):
        values += ["--failed-run", "/runs/incomplete-expanded"]
    return values


def test_registers_all_v9_scientific_commands():
    attribution = parser().parse_args(common("attribute-v8-regression") + [
        "--attribution-manifest", "/benchmark/v9.toml",
        "--v7-run", "/runs/v7", "--v8-run", "/runs/v8",
    ])
    iterate = parser().parse_args(v9_args("iterate-v9"))
    recover = parser().parse_args(v9_args("recover-v9"))
    expanded = parser().parse_args(expanded_args("measure-policy-kl-expanded"))
    recover_expanded = parser().parse_args(expanded_args("recover-policy-kl-expanded"))
    plot = parser().parse_args([
        "generals", "plot-v9-paper",
        "--attribution-run", "/runs/attribution",
        "--v9-run", "/runs/v9",
        "--legacy-kl-run", "/runs/legacy-kl",
        "--expanded-kl-run", "/runs/expanded-kl",
        "--output-dir", "/figures",
    ])

    assert attribution.v7_run == Path("/runs/v7")
    assert iterate.expected_predecessor_hash == "8" * 64
    assert recover.failed_run == Path("/runs/failed-v9")
    assert expanded.count_max_states == 1234
    assert recover_expanded.failed_run == Path("/runs/incomplete-expanded")
    assert plot.output_dir == Path("/figures")


def test_routes_attribution_and_returns_one_for_incomplete(monkeypatch, capsys):
    args = parser().parse_args(common("attribute-v8-regression") + [
        "--attribution-manifest", "/benchmark/v9.toml",
        "--v7-run", "/runs/v7", "--v8-run", "/runs/v8",
    ])
    monkeypatch.setattr(cli, "_assets", lambda _args: (object(), object()))
    monkeypatch.setattr(
        cli,
        "_build_attribution_pipeline",
        lambda *_args: SimpleNamespace(run=lambda: SimpleNamespace(
            status="incomplete",
            run_dir=Path("/runs/attribution"),
            case_count_per_policy=12,
            diagnostic_state_count=0,
        )),
    )

    assert cli.handle(args) == 1
    assert json.loads(capsys.readouterr().out)["run_dir"] == "/runs/attribution"


def test_routes_v9_recovery_without_constructing_codex(monkeypatch, capsys):
    args = parser().parse_args(v9_args("recover-v9"))
    captured = {}

    class Pipeline:
        def recover(self, failed):
            captured["failed"] = failed
            return SimpleNamespace(
                status="complete", run_dir=Path("/runs/v9"),
                evo_score_9=0.5, validation_passed=True,
                formal_attempted=True, runnable=True,
            )

    monkeypatch.setattr(cli, "_assets", lambda _args: (object(), object()))
    monkeypatch.setattr(
        cli,
        "CodexProvider",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("provider forbidden")),
    )
    monkeypatch.setattr(
        cli,
        "_build_v9_pipeline",
        lambda _args, _config, _layout, provider: (
            captured.update(provider=provider) or Pipeline()
        ),
    )

    assert cli.handle(args) == 0
    assert captured["failed"] == Path("/runs/failed-v9")
    assert isinstance(captured["provider"], cli._RecoveryOnlyProvider)
    assert json.loads(capsys.readouterr().out)["evo_score_9"] == 0.5


def test_routes_expanded_recovery_and_plot(monkeypatch, capsys):
    expanded = parser().parse_args(expanded_args("recover-policy-kl-expanded"))
    captured = {}

    class Pipeline:
        def recover(self, failed):
            captured["failed"] = failed
            return SimpleNamespace(
                status="resumable_incomplete",
                run_dir=Path("/runs/expanded"),
                summary={"domains": {}},
            )

    monkeypatch.setattr(cli, "_assets", lambda _args: (object(), object()))
    monkeypatch.setattr(cli, "_build_expanded_pipeline", lambda *_args: Pipeline())
    assert cli.handle(expanded) == 1
    assert captured["failed"] == Path("/runs/incomplete-expanded")
    capsys.readouterr()

    plot = parser().parse_args([
        "generals", "plot-v9-paper",
        "--attribution-run", "/runs/a", "--v9-run", "/runs/v9",
        "--legacy-kl-run", "/runs/l", "--expanded-kl-run", "/runs/e",
        "--output-dir", "/figures",
    ])
    monkeypatch.setattr(cli, "load_v9_figure_data", lambda *args: captured.update(figure_args=args) or object())
    monkeypatch.setattr(cli, "render_v9_paper_figures", lambda _data, _output: (Path("/figures/a.png"),))
    assert cli.handle(plot) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "complete"
    assert captured["figure_args"][-1] == Path("/runs/e")
