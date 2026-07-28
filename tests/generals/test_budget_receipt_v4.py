import json

from agentbench_frame.generals.pipeline_v4 import (
    derive_round4_campaign_budget,
)


def test_campaign_budget_receipt_adds_prior_attempt_without_mutating_runs(
    tmp_path,
):
    success = tmp_path / "success"
    prior = tmp_path / "prior"
    success.mkdir()
    prior.mkdir()
    success_summary = {
        "run_id": "v4-success",
        "status": "complete",
        "parent_run_id": "v3-parent",
        "parent_version": "v3",
        "learning_id": "generals-hl-v4-strongest-v1",
        "act_count": 5,
        "prior_attempt_run_id": None,
        "cumulative_learning_budget": {
            "learning_coding_agent_acts": 5,
            "learning_episodes": 52,
            "learning_env_steps": 29273,
            "learning_game_agent_decision_steps": 11288,
            "learning_primitive_commands": 46221,
            "learning_prompt_tokens": None,
            "learning_completion_tokens": None,
            "learning_total_tokens": None,
            "learning_time_s": 774.0,
        },
    }
    prior_summary = {
        "run_id": "v4-prior",
        "status": "prompt_incomplete",
        "parent_run_id": "v3-parent",
        "parent_version": "v3",
        "learning_id": "generals-hl-v4-strongest-v1",
        "act_count": 4,
        "round_act_count": 0,
        "budget": {
            "learning_coding_agent_acts": 0,
            "learning_episodes": 6,
            "learning_env_steps": 1609,
            "learning_game_agent_decision_steps": 803,
            "learning_primitive_commands": 4244,
            "learning_prompt_tokens": None,
            "learning_completion_tokens": None,
            "learning_total_tokens": None,
            "learning_time_s": 5.5,
        },
    }
    (success / "summary.json").write_text(
        json.dumps(success_summary),
        encoding="utf-8",
    )
    (prior / "summary.json").write_text(
        json.dumps(prior_summary),
        encoding="utf-8",
    )
    success_before = (success / "summary.json").read_bytes()
    prior_before = (prior / "summary.json").read_bytes()
    output = tmp_path / "derived" / "campaign-budget.json"

    receipt = derive_round4_campaign_budget(success, prior, output)

    assert receipt["status"] == "derived_prior_attempt_added"
    assert receipt["before"]["learning_episodes"] == 52
    assert receipt["prior_attempt"]["learning_episodes"] == 6
    assert receipt["after"]["learning_episodes"] == 58
    assert receipt["after"]["learning_coding_agent_acts"] == 5
    assert receipt["after"]["learning_env_steps"] == 30882
    assert receipt["after"]["learning_total_tokens"] is None
    assert output.is_file()
    assert json.loads(output.read_text()) == receipt
    assert (success / "summary.json").read_bytes() == success_before
    assert (prior / "summary.json").read_bytes() == prior_before
