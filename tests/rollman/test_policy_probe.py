from pathlib import Path

from agentbench_frame.games.rollman.measurement import decision_trace_metrics
from agentbench_frame.games.rollman.policy_probe import run_probe_episode


def _state(round_number):
    return {
        "level": 1,
        "round": round_number,
        "board_size": 3,
        "board": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "pacman_skill_status": [0, 0, 0, 0, 0],
        "pacman_coord": [1, 1],
        "ghosts_coord": [[1, 1], [1, 1], [1, 1]],
        "score": [0, 0],
        "beannumber": 0,
        "portal_available": False,
        "portal_coord": [1, 1],
    }


def test_fixed_reference_probe_turns_deterministic_actions_into_kl(tmp_path):
    sdk = Path("/Users/qingle/Code/SAST/PacmanSDK-python")
    if not sdk.is_dir():
        return
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "ai.py").write_text(
        "def ai_func(s): return {'action': 0, 'memory_id': f'old-{s.round}'}\n",
        encoding="utf-8",
    )
    (new / "ai.py").write_text(
        "def ai_func(s): return {'action': 4 if s.round else 0, 'memory_id': f'new-{s.round}'}\n",
        encoding="utf-8",
    )
    states = [_state(0), _state(1)]

    old_decisions = run_probe_episode(
        workspace=old,
        sdk_root=sdk,
        states=states,
        artifact_path=tmp_path / "old.json",
    )
    new_decisions = run_probe_episode(
        workspace=new,
        sdk_root=sdk,
        states=states,
        artifact_path=tmp_path / "new.json",
    )
    metrics = decision_trace_metrics(
        new_decisions,
        old_decisions,
        epsilon=0.05,
    )

    assert [item["action"] for item in old_decisions] == [0, 0]
    assert [item["action"] for item in new_decisions] == [0, 4]
    assert old_decisions[1]["memory_id"] == "old-1"
    assert metrics["local_policy_kl_trace"][0] == 0
    assert metrics["local_policy_kl_trace"][1] > 0


def test_probe_reconstructs_official_space_info(tmp_path):
    sdk = Path("/Users/qingle/Code/SAST/PacmanSDK-python")
    if not sdk.is_dir():
        return
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "ai.py").write_text(
        "def ai_func(s):\n"
        "    keys = {'observation_space', 'pacman_action_space', 'ghost_action_space'}\n"
        "    return 4 if keys <= set(s.space_info) else 0\n",
        encoding="utf-8",
    )

    decisions = run_probe_episode(
        workspace=policy,
        sdk_root=sdk,
        states=[_state(0)],
        artifact_path=tmp_path / "probe.json",
    )

    assert decisions[0]["action"] == 4
