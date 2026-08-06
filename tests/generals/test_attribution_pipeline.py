import json
from pathlib import Path

import pytest

from agentbench_frame.eval.benchmark import GameResult
from agentbench_frame.generals.assets import load_pilot_config
from agentbench_frame.generals.evaluator import GeneralsEvaluation
from agentbench_frame.generals.historical_policy import (
    HistoricalPolicySource,
    PolicyProbeResult,
)
from agentbench_frame.generals.models import MatchResult, TurnRecord
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


FIXTURE = Path(__file__).parent / "fixtures" / "pilot-v1.toml"
CHALLENGE = Path(__file__).parent / "fixtures" / "v9-scientific-attribution-v1.toml"

_STRATEGY = '''RECRUIT_RESERVE = 30

def _neighbors(view, seat, position):
    return ()

def _hostile(cell, seat):
    return False

def _position(key):
    return tuple(map(int, key.split(",")))

def _main(view, seat):
    return None

def _effect_fields_known(view):
    return True

def _main_threatened(view: dict, seat: int) -> bool:
    main = _main(view, seat)
    if main is None:
        return False
    position = tuple(main["position"])
    return any(_hostile(cell, seat) for _, _, cell in _neighbors(view, seat, position))


def _urgent_main_defense(view: dict, seat: int):
    return None


def _best_valuable_move(view: dict, seat: int, used_sources: set):
    amount = 30
    required = 0
    route_cost = 2
    first_cell = {"army": 1, "player": seat}
    source = (1, 1)
    first = (1, 2)
    main_position = (0, 0)
    objective = (2, 2)
    DIRECTION = {(0, 1): 4}
    def _target_class(*args): return 0
    def _move(*args): return [8]
    sent = amount if required == 0 else required
    # Objective value, route cost, defender cost, attacking surplus,
    # main exposure, row/column, direction.
    score = (
        _target_class(view, seat, objective),
        route_cost,
        int(first_cell["army"]) if int(first_cell["player"]) != seat else 0,
        -(amount - required),
        source == main_position,
        source,
        DIRECTION[(first[0] - source[0], first[1] - source[1])],
    )
    return score, _move(source, first, sent)


def _upgrade_candidate(view: dict, seat: int, round_number: int):
    if _main_threatened(view, seat) or not _effect_fields_known(view):
        return None
    return [8]


def choose_actions(*args):
    return [[8]]
'''


def _historical(root: Path, version: str, strategy: str) -> HistoricalPolicySource:
    source = root / version / "source"
    (source / "tests").mkdir(parents=True)
    (source / "main.py").write_text("from strategy import choose_actions\n")
    (source / "strategy.py").write_text(strategy)
    (source / "state_view.py").write_text("def normalize(x): return x\n")
    (source / "STRATEGY.md").write_text(f"{version}\n")
    (source / "EXPERIENCE.md").write_text(f"{version}\n")
    (source / "tests/test_strategy.py").write_text("def test_ok(): assert True\n")
    manifest = LocalWorkspaceSnapshotter().capture(source)
    return HistoricalPolicySource(
        version=version,
        run_id=f"run-{version}",
        content_hash=manifest.content_hash,
        source=source,
        manifest=manifest,
    )


def _state(seat: int, marker: int) -> dict:
    return {
        "round": marker,
        "my_seat": seat,
        "cells": {
            "7,4": {"type": 0, "player": seat, "army": 20, "general_id": 1},
            "7,10": {"type": 0, "player": 1 - seat, "army": 20, "general_id": 2},
            "5,5": {"type": 0, "player": seat, "army": 30, "general_id": None},
            "5,6": {"type": 0, "player": 1 - seat, "army": 2, "general_id": None},
        },
        "generals": {
            "1": {"id": 1, "type": "main", "player": seat, "position": [7, 4]},
            "2": {"id": 2, "type": "main", "player": 1 - seat, "position": [7, 10]},
        },
        "coins": [40, 40],
    }


class PairedEvaluator:
    def __init__(self, *, invalid_cell=None, mismatched_pair=False):
        self.invalid_cell = invalid_cell
        self.mismatched_pair = mismatched_pair
        self.calls = []

    def evaluate(self, workspace, version, phase, run, cases):
        self.calls.append((version, tuple(case.case_id for case in cases)))
        results = []
        matches = []
        for index, case in enumerate(cases):
            valid = not (version == self.invalid_cell and index == 0)
            seed = case.seed + (1 if self.mismatched_pair and version == "D" and index == 0 else 0)
            before = _state(case.first_player, index + 1)
            after = _state(case.first_player, index + 2)
            action = ((8,),) if version in {"A", "C"} else ((1, 5, 5, 4, 1), (8,))
            turn = TurnRecord(
                step=0,
                round_number=index + 1,
                player=case.first_player,
                state_id_before=f"shared-{case.seed}-{case.first_player}",
                state_before=before,
                commands=action,
                state_id_after=f"after-{case.seed}-{case.first_player}",
                state_after=after,
            )
            artifact_dir = Path(run.run_dir) / "matches" / version / case.case_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            snapshot = {
                "schema": "generals-measurement-state-v1",
                "actor": case.first_player,
                "state": {"round": index + 1, "marker": f"{version}-{case.case_id}"},
            }
            from agentbench_frame.generals.measurement_state import measurement_state_id
            (artifact_dir / "measurement-state.jsonl").write_text(json.dumps({
                "global_step": 0,
                "player": case.first_player,
                "measurement_state_id": measurement_state_id(snapshot),
                "snapshot": snapshot,
            }) + "\n")
            match = MatchResult(
                case_id=case.case_id,
                valid=valid,
                winner=case.first_player if version in {"B", "D"} else 1 - case.first_player,
                termination_type="normal" if valid else "process_error",
                seed=seed,
                evaluated_seat=case.first_player,
                turns=(turn,),
                elapsed_time_s=0.01,
                engine_hash="engine",
                error=None if valid else "boom",
            )
            matches.append(match)
            outcome = "win" if match.winner == case.first_player else "loss"
            results.append(GameResult(case.case_id, outcome, valid, match.error, {
                "seed": seed,
                "evaluated_seat": case.first_player,
                "artifact_dir": str(artifact_dir),
            }))
            run.write("dense_trajectory", case_id=case.case_id, version=version, phase=phase)
            run.write("dense_episode_summary", case_id=case.case_id, version=version, phase=phase)
            run.log_game_result(phase, version, {
                "case_id": case.case_id,
                "outcome": outcome,
                "valid": valid,
                "seed": seed,
                "evaluated_seat": case.first_player,
            })
        return GeneralsEvaluation(
            version=version,
            status="complete" if all(item.valid for item in results) else "incomplete",
            score=sum(item.outcome == "win" for item in results) / len(results) if all(item.valid for item in results) else None,
            wins=sum(item.outcome == "win" for item in results),
            losses=sum(item.outcome == "loss" for item in results),
            draws=0,
            per_tier={"high": None, "medium": None, "low": None},
            seat_gap=0.0,
            results=tuple(results),
            matches=tuple(matches),
        )


def _probe(policy, measurement_state, **kwargs):
    del measurement_state, kwargs
    action = ((8,),) if policy.version.endswith(("A", "C")) else ((1, 5, 5, 4, 1), (8,))
    return PolicyProbeResult(
        version=policy.version,
        status="complete",
        deterministic=True,
        raw_actions=(action, action),
        elapsed_time_s=0.001,
        stdout="",
        stderr="",
    )


def _pipeline(tmp_path, evaluator=None, probe=_probe):
    from agentbench_frame.generals.attribution_pipeline import GeneralsAttributionPipeline
    from agentbench_frame.generals.challenge_v9 import load_round9_challenge_config

    config = load_pilot_config(FIXTURE)
    challenge = load_round9_challenge_config(
        CHALLENGE,
        config,
        engine_hash="4ca898365a6fe9a0cca39c3c8be0e49f449334814b106882c0b78b3aabdb1d97",
        replay_skill_sha256="0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48",
    )
    v7 = _historical(tmp_path / "history", "v7", _STRATEGY)
    v8 = _historical(tmp_path / "history", "v8", _STRATEGY + "\nV8 = True\n")
    return GeneralsAttributionPipeline(
        config=config,
        challenge=challenge,
        v7=v7,
        v8=v8,
        data_dir=tmp_path / "data",
        evaluator=evaluator or PairedEvaluator(),
        engine_root=tmp_path / "engine",
        sdk_root=tmp_path / "sdk",
        probe_policy=probe,
        canonicalize_action=lambda _, action: tuple(tuple(item) for item in action),
        bootstrap_replicates=100,
    )


def test_attribution_pipeline_runs_four_cells_on_identical_cases(tmp_path):
    evaluator = PairedEvaluator()
    result = _pipeline(tmp_path, evaluator=evaluator).run()

    assert result.status == "complete"
    assert result.policy_order == ("A", "B", "C", "D")
    assert result.case_count_per_policy == 12
    assert result.valid_case_count == 48
    assert result.diagnostic_state_count <= 48
    assert [item[0] for item in evaluator.calls] == ["A", "B", "C", "D"]
    assert len({item[1] for item in evaluator.calls}) == 1
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["coding_agent_act_count"] == 0
    assert summary["formal_benchmark_opened"] is False
    assert summary["attribution"]["estimands"] == ["B-A", "C-A", "D-B-C+A"]
    assert summary["event_quality"]["unknown_event_types"] == 0
    assert (result.run_dir / "diagnosis/report.json").is_file()
    assert (result.run_dir / "diagnosis/report.md").is_file()
    assert (result.run_dir / "diagnosis/evidence.json").is_file()
    events = [
        json.loads(line)
        for line in (result.run_dir / "events.jsonl").read_text().splitlines()
    ]
    counts = {
        event_type: sum(item["event_type"] == event_type for item in events)
        for event_type in {item["event_type"] for item in events}
    }
    assert counts["attribution_policy_materialized"] == 4
    assert counts["attribution_game_result"] == 48
    assert counts["dense_episode_summary"] == 48
    assert counts["diagnostic_policy_probe"] == result.diagnostic_state_count * 4
    assert counts.get("coding_agent_act", 0) == 0
    assert not any(item.get("phase") == "formal" for item in events)


def test_attribution_pipeline_rejects_an_invalid_game(tmp_path):
    result = _pipeline(tmp_path, evaluator=PairedEvaluator(invalid_cell="C")).run()
    assert result.status == "incomplete"
    assert result.report_hash is None


def test_attribution_pipeline_rejects_mismatched_pair_coordinates(tmp_path):
    result = _pipeline(tmp_path, evaluator=PairedEvaluator(mismatched_pair=True)).run()
    assert result.status == "invalid_attribution_pairs"
    assert result.report_hash is None


def test_attribution_pipeline_rejects_source_hash_drift(tmp_path):
    pipeline = _pipeline(tmp_path)
    (pipeline.v7.source / "strategy.py").write_text(_STRATEGY + "\nDRIFT = True\n")
    result = pipeline.run()
    assert result.status == "source_hash_drift"


def test_attribution_pipeline_rejects_nondeterministic_probe(tmp_path):
    def nondeterministic(policy, measurement_state, **kwargs):
        result = _probe(policy, measurement_state, **kwargs)
        return PolicyProbeResult(
            version=result.version,
            status="complete",
            deterministic=False,
            raw_actions=(((8,),), ((1, 1, 1, 4, 1), (8,))),
            elapsed_time_s=result.elapsed_time_s,
            stdout="",
            stderr="",
        )

    result = _pipeline(tmp_path, probe=nondeterministic).run()
    assert result.status == "invalid_diagnostic_probe"
    assert result.report_hash is None


def test_attribution_pipeline_rejects_unknown_or_malformed_events(tmp_path):
    class DefectiveEvaluator(PairedEvaluator):
        def evaluate(self, workspace, version, phase, run, cases):
            result = super().evaluate(workspace, version, phase, run, cases)
            if version == "A":
                run.write("not_a_known_attribution_event")
                run.writer.flush()
                with (Path(run.run_dir) / "events.jsonl").open("a") as stream:
                    stream.write("not-json\n")
            return result

    result = _pipeline(tmp_path, evaluator=DefectiveEvaluator()).run()

    assert result.status == "invalid_event_quality"
    assert result.report_hash is None
