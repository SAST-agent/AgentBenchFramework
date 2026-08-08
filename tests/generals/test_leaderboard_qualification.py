from copy import deepcopy
import hashlib
from pathlib import Path

from agentbench_frame.generals.assets import load_pilot_config
from agentbench_frame.generals.leaderboard_qualification import (
    QUALIFICATION_SCHEMA,
    evaluate_leaderboard_qualification,
    load_leaderboard_qualification_config,
    wilson_lower_bound,
)


FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = FIXTURES / "leaderboard-qualification-v1.toml"
PILOT = FIXTURES / "pilot-v1.toml"
ENGINE_HASH = "4ca898365a6fe9a0cca39c3c8be0e49f449334814b106882c0b78b3aabdb1d97"
OPPONENT_HASH = "c261f48bea35b0757e45d28118a29e675f3ceabf608039fc0421104e88643db4"
SKILL_HASH = "0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48"


def _config():
    return load_leaderboard_qualification_config(
        MANIFEST,
        load_pilot_config(PILOT),
        engine_sha256=ENGINE_HASH,
        opponent_tree_sha256=OPPONENT_HASH,
        replay_skill_sha256=SKILL_HASH,
    )


def _results(config, wins):
    rows = []
    wins_by_seat = {0: (wins + 1) // 2, 1: wins // 2}
    seen_by_seat = {0: 0, 1: 0}
    for seed in config.evaluation_seeds:
        for seat in config.seats:
            outcome = (
                "win"
                if seen_by_seat[seat] < wins_by_seat[seat]
                else "loss"
            )
            seen_by_seat[seat] += 1
            rows.append(
                {
                    "case_id": f"qualify-{config.opponent_id}-s{seed}-p{seat}",
                    "seed": seed,
                    "seat": seat,
                    "outcome": outcome,
                    "valid": True,
                }
            )
    return rows


def _receipt(crossings=(8, 16, None)):
    config = _config()
    replicates = []
    for replicate_index, crossing in enumerate(crossings, start=1):
        checkpoints = []
        for acts in config.budget_checkpoints:
            wins = 26 if crossing is not None and acts >= crossing else 25
            checkpoints.append(
                {
                    "budget": {
                        "coding_agent_acts": acts,
                        "total_tokens": acts * 1_000,
                        "wall_time_s": acts * 10.0,
                        "learning_episodes": acts * 6,
                        "cost_usd": acts * 1.5,
                    },
                    "policy_hash": hashlib.sha256(
                        f"{replicate_index}:{acts}".encode()
                    ).hexdigest(),
                    "evaluation_status": "complete",
                    "results": _results(config, wins),
                }
            )
        replicates.append(
            {
                "replicate_id": f"replicate-{replicate_index}",
                "run_id": f"qualification-run-{replicate_index}",
                "provider_trace_sha256": hashlib.sha256(
                    f"provider-trace:{replicate_index}".encode()
                ).hexdigest(),
                "checkpoints": checkpoints,
            }
        )
    return {
        "schema": QUALIFICATION_SCHEMA,
        "qualification_id": config.qualification_id,
        "provider": "reference-provider",
        "model": "frontier-model",
        "model_revision": "2026-08-08",
        "harness_hash": "a" * 64,
        "initial_policy_hash": config.initial_policy_hash,
        "engine_sha256": config.engine_sha256,
        "opponent_id": config.opponent_id,
        "opponent_tree_sha256": config.opponent_tree_sha256,
        "replay_skill_sha256": config.replay_skill_sha256,
        "qualification_results_exposed_to_provider": False,
        "selection_uses_qualification_results": False,
        "replicates": replicates,
    }


def test_manifest_freezes_finite_frontier_qualification_contract():
    config = _config()

    assert config.budget_checkpoints == (1, 2, 4, 8, 16, 32)
    assert len(config.evaluation_seeds) == 20
    assert config.seats == (0, 1)
    assert config.replicates == 3
    assert config.minimum_qualifying_replicates == 2
    assert config.minimum_wins_per_seat == 11
    assert config.initial_policy_version == "v7"


def test_one_sided_wilson_boundary_requires_26_of_40_wins():
    assert wilson_lower_bound(25, 40, 0.95) < 0.5
    assert wilson_lower_bound(26, 40, 0.95) > 0.5


def test_twenty_six_wins_still_fail_when_one_seat_has_only_ten():
    receipt = _receipt()
    checkpoint = receipt["replicates"][0]["checkpoints"][3]
    seat_zero_losses = [
        item
        for item in checkpoint["results"]
        if item["seat"] == 0 and item["outcome"] == "loss"
    ]
    seat_one_wins = [
        item
        for item in checkpoint["results"]
        if item["seat"] == 1 and item["outcome"] == "win"
    ]
    for item in seat_zero_losses[:3]:
        item["outcome"] = "win"
    for item in seat_one_wins[:3]:
        item["outcome"] = "loss"

    result = evaluate_leaderboard_qualification(_config(), receipt)
    row = next(
        item
        for item in result.checkpoints
        if item.replicate_id == "replicate-1"
        and item.coding_agent_acts == 8
    )

    assert row.wins == 26
    assert row.per_seat_wins == {0: 16, 1: 10}
    assert row.passed is False


def test_two_of_three_replicates_qualify_at_second_crossing_budget():
    result = evaluate_leaderboard_qualification(_config(), _receipt())

    assert result.status == "qualified"
    assert result.qualified is True
    assert result.qualifying_replicates == ("replicate-1", "replicate-2")
    assert result.qualification_budget_acts == 16
    passing = [item for item in result.checkpoints if item.passed]
    assert passing[0].wins == 26
    assert passing[0].per_seat_wins == {0: 13, 1: 13}
    assert passing[0].win_rate_lower_bound > 0.5


def test_complete_matrix_with_one_crossing_is_unqualified():
    result = evaluate_leaderboard_qualification(
        _config(), _receipt((8, None, None))
    )

    assert result.status == "unqualified"
    assert result.qualified is False
    assert result.reasons == ("insufficient_qualifying_replicates",)


def test_missing_checkpoint_preserves_incomplete_status():
    receipt = _receipt()
    receipt["replicates"][2]["checkpoints"].pop()

    result = evaluate_leaderboard_qualification(_config(), receipt)

    assert result.status == "incomplete"
    assert result.qualified is False
    assert any(
        item.reasons == ("missing_checkpoint",)
        for item in result.checkpoints
    )


def test_invalid_game_does_not_become_a_loss_or_score():
    receipt = _receipt()
    checkpoint = receipt["replicates"][0]["checkpoints"][0]
    checkpoint["results"][0]["valid"] = False

    result = evaluate_leaderboard_qualification(_config(), receipt)

    assert result.status == "incomplete"
    row = result.checkpoints[0]
    assert row.status == "incomplete"
    assert row.score is None


def test_case_identity_or_budget_regression_invalidates_receipt():
    identity = _receipt()
    identity["replicates"][0]["checkpoints"][0]["results"][0]["seat"] = 1
    assert evaluate_leaderboard_qualification(
        _config(), identity
    ).status == "invalid"

    budget = _receipt()
    budget["replicates"][0]["checkpoints"][1]["budget"][
        "total_tokens"
    ] = 1
    result = evaluate_leaderboard_qualification(_config(), budget)
    assert result.status == "invalid"
    assert "checkpoint_budget_decreased" in result.reasons


def test_boolean_case_identity_or_duplicate_run_invalidates_receipt():
    identity = _receipt()
    identity["replicates"][0]["checkpoints"][0]["results"][0][
        "seat"
    ] = False
    result = evaluate_leaderboard_qualification(_config(), identity)
    assert result.status == "invalid"
    assert "case_result_identity_changed" in result.reasons

    duplicate = _receipt()
    duplicate["replicates"][1]["run_id"] = duplicate["replicates"][0][
        "run_id"
    ]
    result = evaluate_leaderboard_qualification(_config(), duplicate)
    assert result.status == "invalid"
    assert "replicate_run_id_invalid" in result.reasons


def test_provider_feedback_or_contract_mutation_invalidates_receipt():
    exposed = _receipt()
    exposed["qualification_results_exposed_to_provider"] = True
    result = evaluate_leaderboard_qualification(_config(), exposed)
    assert result.status == "invalid"
    assert "qualification_results_exposed_to_provider" in result.reasons

    changed = deepcopy(_receipt())
    changed["engine_sha256"] = "b" * 64
    result = evaluate_leaderboard_qualification(_config(), changed)
    assert result.status == "invalid"
    assert "engine_hash_changed" in result.reasons
