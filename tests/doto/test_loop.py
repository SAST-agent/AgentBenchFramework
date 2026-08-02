import json
from pathlib import Path

from agentbench_frame.doto.loop import build_messages, read_evidence


def test_messages_contain_only_declared_context():
    messages = build_messages("CPP", {"harness": "H", "replay": "R"}, "SDK",
                              [{"episode": 1}], {"score": 2}, {"tokens": 3})
    user = messages[1]["content"]
    assert "# Current playerAI.cpp\nCPP" in user
    assert "# DOTO Harness Skill\nH" in user
    assert "# DOTO Replay Reader Skill\nR" in user
    assert "# Fixed SDK Reference\nSDK" in user
    assert "API_KEY" not in user


def test_evidence_reads_only_paired_frames_with_limit(tmp_path):
    trace = tmp_path / "trace.jsonl"
    rows = [
        {"kind": "observation", "faction": 0, "frame": 1, "payload": {"frame": 1}},
        {"kind": "action", "faction": 0, "frame": 1, "payload": {"move": []}},
        {"kind": "observation", "faction": 0, "frame": 2, "payload": {"frame": 2}},
        {"kind": "action", "faction": 0, "frame": 2, "payload": {"move": []}},
    ]
    trace.write_text("\n".join(json.dumps(row) for row in rows))
    evidence = read_evidence([{"episode_id": "x", "trace": str(trace)}], 1, 1)
    assert len(evidence) == 1
    assert [row["frame"] for row in evidence[0]["frames"]] == [1]
