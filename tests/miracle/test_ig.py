"""确定性 Miracle 策略的严格 KL 状态与曲线契约。"""

import json

from agentbench_frame.miracle.decision_space import Action
from agentbench_frame.miracle.ig import (
    aggregate_episode,
    build_ig_curve,
    compare_agents_on_trace,
    compare_deterministic,
    save_episode_ig,
    save_ig_curve,
)


END = Action("endround", {})
MOVE = Action("move", {"mover": 7, "position": [1, -1, 0]})


def test_same_deterministic_action_has_zero_strict_kl():
    row = compare_deterministic("obs-1", END, END, support=[END, MOVE])

    assert row["status"] == "unchanged"
    assert row["kl"] == 0.0
    assert row["missing_reason"] is None


def test_changed_deterministic_action_is_strict_kl_infinite():
    row = compare_deterministic("obs-1", END, MOVE, support=[END, MOVE])

    assert row["status"] == "infinite"
    assert row["kl"] is None
    assert row["missing_reason"] == "support_expansion"


def test_action_outside_legal_support_is_missing_not_kl():
    row = compare_deterministic("obs-1", END, MOVE, support=[END])

    assert row["status"] == "missing"
    assert row["missing_reason"] == "action_outside_support"


def test_episode_aggregation_keeps_status_ratios():
    rows = [
        compare_deterministic("a", END, END, [END, MOVE]),
        compare_deterministic("b", END, END, [END, MOVE]),
        compare_deterministic("c", END, MOVE, [END, MOVE]),
        compare_deterministic("d", END, MOVE, [END]),
    ]

    result = aggregate_episode(rows, episode_id="ep-7", iteration=3)

    assert result["finite_kl_mean"] == 0.0
    assert result["unchanged_ratio"] == 0.5
    assert result["infinite_ratio"] == 0.25
    assert result["missing_ratio"] == 0.25
    assert result["counts"] == {"unchanged": 2, "infinite": 1, "missing": 1}


def test_curve_preserves_baseline_and_null_finite_kl(tmp_path):
    ep = aggregate_episode(
        [compare_deterministic("a", END, MOVE, [END, MOVE])],
        episode_id="ep-1", iteration=1,
    )
    curve = build_ig_curve([ep], versions={0: "baseline", 1: "candidate-v1"})

    assert curve["points"][0]["iteration"] == 0
    assert curve["points"][0]["status"] == "baseline"
    assert curve["points"][1]["finite_kl_mean"] is None
    assert curve["points"][1]["infinite_ratio"] == 1.0

    out = tmp_path / "ig-curve.json"
    save_ig_curve(curve, out)
    assert json.loads(out.read_text(encoding="utf-8")) == curve


def _trace_row(seq, obs):
    content = "000000" + json.dumps(obs)
    return {
        "seq": seq,
        "kind": "from_logic",
        "payload": {"listen": [obs["camp"]], "content": [content]},
    }


def test_real_trace_is_replayed_through_old_and_new_agents(tmp_path):
    obs = {"camp": 0, "round": 3, "map": {"units": [], "barracks": []},
           "players": [[[], 0, 0, [], []], [[], 0, 0, [], []]]}
    trace = tmp_path / "match.trace.jsonl"
    trace.write_text(json.dumps(_trace_row(7, obs)) + "\n", encoding="utf-8")

    class Old:
        def act(self, observation):
            return {"operation_type": "endround", "operation_parameters": {}}

    class New:
        def act(self, observation):
            return {"operation_type": "surrender", "operation_parameters": {}}

    result = compare_agents_on_trace(
        trace, Old(), New(), camp=0, iteration=1,
        old_version="old", new_version="new",
    )

    assert result["source_trace"] == str(trace.resolve())
    assert result["old_version"] == "old"
    assert result["new_version"] == "new"
    assert result["n_decisions"] == 1
    assert result["infinite_ratio"] == 1.0
    assert result["decisions"][0]["trace_seq"] == 7


def test_malformed_agent_action_is_missing_with_reason(tmp_path):
    obs = {"camp": 0, "round": 1, "map": {"units": [], "barracks": []},
           "players": [[[], 0, 0, [], []], [[], 0, 0, [], []]]}
    trace = tmp_path / "bad.trace.jsonl"
    trace.write_text(json.dumps(_trace_row(1, obs)) + "\n", encoding="utf-8")

    class Good:
        def act(self, observation):
            return {"operation_type": "endround", "operation_parameters": {}}

    class Bad:
        def act(self, observation):
            return {"wrong": "shape"}

    result = compare_agents_on_trace(
        trace, Good(), Bad(), camp=0, iteration=2,
        old_version="good", new_version="bad",
    )

    assert result["missing_ratio"] == 1.0
    assert result["decisions"][0]["missing_reason"] == "new_action_invalid"


def test_episode_save_uses_iteration_directory(tmp_path):
    episode = aggregate_episode([], episode_id="ep-1", iteration=4)
    path = save_episode_ig(episode, tmp_path)

    assert path == tmp_path / "iteration-0004" / "ep-1.json"
    assert json.loads(path.read_text(encoding="utf-8"))["iteration"] == 4
