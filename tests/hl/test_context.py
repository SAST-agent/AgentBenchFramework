"""Tests for hl/context.py — ContextBuilder prompt assembly."""
from __future__ import annotations

import json
from pathlib import Path

from agentbench_frame.hl.codebase import HLCodebase
from agentbench_frame.hl.context import ContextBuilder
from agentbench_frame.hl.reference import BenchmarkSpec


def _spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        spec_id="bench-v1", opponents=("rank06", "rank12"),
        pairs=1, seats="0", timeout=5.0,
    )


def _codebase(tmp_path: Path) -> HLCodebase:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.py").write_text("THRESHOLD = 10\n", encoding="utf-8")
    return HLCodebase(root=ws, store=tmp_path / "store")


def test_prompt_has_role_and_rules_and_no_history(tmp_path):
    cb = HLCodebase(root=tmp_path / "ws", store=tmp_path / "store")
    (tmp_path / "ws").mkdir()
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec())
    out = builder.build(version_before=None, act_id="r-act0001")
    prompt = out["prompt"]
    assert "LostSpace" in prompt
    assert "manifest.toml" in prompt          # don't-touch rule
    assert "rank06" in prompt and "rank12" in prompt  # opponents listed
    assert "First act" in prompt
    assert out["timeout"] == 600.0


def test_prompt_includes_match_history_when_runs_exist(tmp_path):
    cb = _codebase(tmp_path)
    # fabricate a run dir the MatchHistoryView will pick up
    run_dir = tmp_path / "runs" / "25_lostspace" / "hl-v1" / "run-001"
    run_dir.mkdir(parents=True)
    (run_dir / "matches.jsonl").write_text(
        '{"opponent":"rank06","candidate_result":"win",'
        '"candidate_rank":1,"candidate_score":4,"turns":35,"pair":0,'
        '"candidate_seat":0}\n', encoding="utf-8")
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec())
    prompt = builder.build(version_before=None, act_id="r-act0001")["prompt"]
    assert "Match history" in prompt
    assert "rank06" in prompt
    assert "Replays" in prompt           # replay pointers section
    assert "matches.jsonl" in prompt


def test_prompt_never_writes_files_into_workspace(tmp_path):
    cb = _codebase(tmp_path)
    before = {p for p in cb.root.rglob("*") if p.is_file()}
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec())
    builder.build(version_before=None, act_id="r-act0001")
    after = {p for p in cb.root.rglob("*") if p.is_file()}
    assert before == after   # no CONTEXT.md or similar dropped into workspace


def test_prompt_shows_previous_version_when_given(tmp_path):
    cb = _codebase(tmp_path)
    v0 = cb.snapshot(parent_version_id=None)
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec())
    prompt = builder.build(version_before=v0, act_id="r-act0002")["prompt"]
    assert "previous version" in prompt.lower()
    assert v0.version_id in prompt
    assert v0.content_hash in prompt


def test_prompt_contains_inline_playback_recipe(tmp_path):
    cb = _codebase(tmp_path)
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec())
    prompt = builder.build(version_before=None, act_id="r-act0001")["prompt"]
    # the condensed recipe teaches the agent the replay array layout
    assert "birthplaces" in prompt
    assert "score_dic" in prompt
    assert "ai_error" in prompt


def test_prompt_has_anti_derail_clause_unconditional(tmp_path):
    """The STAY ON MISSION clause is present on every act, including the
    first act with no history — so the agent never abandons editing to
    debug the harness when it sees an all-error history."""
    cb = _codebase(tmp_path)
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec())
    prompt = builder.build(version_before=None, act_id="r-act0001")["prompt"]
    assert "STAY ON MISSION" in prompt
    # explicit direction not to debug the harness on all-error histories
    assert "error" in prompt.lower()
    assert "harness" in prompt.lower()
    # bounds: agentbench_data/eval/logic/judger stay off-limits, but reading
    # the ranked reference corpus for strategy research is now allowed.
    assert "agentbench_data" in prompt.lower()
    assert "25_lostspace_final_ladder" in prompt


def test_prompt_has_anti_derail_clause_with_history(tmp_path):
    """Clause is present even when a real (non-error) history exists."""
    cb = _codebase(tmp_path)
    run_dir = tmp_path / "runs" / "25_lostspace" / "hl-v1" / "run-001"
    run_dir.mkdir(parents=True)
    (run_dir / "matches.jsonl").write_text(
        '{"opponent":"rank06","candidate_result":"win",'
        '"candidate_rank":1,"candidate_score":4,"turns":35,"pair":0,'
        '"candidate_seat":0}\n', encoding="utf-8")
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec())
    prompt = builder.build(version_before=None, act_id="r-act0001")["prompt"]
    assert "STAY ON MISSION" in prompt


def test_prompt_anti_derail_clause_on_all_error_history(tmp_path):
    """Spec scenario: when the match history is 100% errors, the prompt
    still contains the anti-derail clause directing the agent to make a
    small edit to agent.py and not debug the harness."""
    cb = _codebase(tmp_path)
    run_dir = tmp_path / "runs" / "25_lostspace" / "hl-v1" / "run-001"
    run_dir.mkdir(parents=True)
    (run_dir / "matches.jsonl").write_text(
        '{"opponent":"rank06","candidate_result":"error",'
        '"candidate_seat":0,"pair":0,"error":"logic exited"}\n'
        '{"opponent":"rank12","candidate_result":"error",'
        '"candidate_seat":0,"pair":1,"error":"logic exited"}\n',
        encoding="utf-8")
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec())
    prompt = builder.build(version_before=None, act_id="r-act0001")["prompt"]
    assert "STAY ON MISSION" in prompt
    assert "harness" in prompt.lower()


def test_prompt_replay_section_includes_seat0_digest(tmp_path):
    """A4: when a replay file is present, the replay section renders a seat-0
    failure digest (keys / escaped / died / rounds) so the agent sees *why* it
    lost, not just that it lost."""
    cb = _codebase(tmp_path)
    run_dir = tmp_path / "runs" / "25_lostspace" / "hl-v1" / "run-001"
    art = run_dir / "artifacts"
    art.mkdir(parents=True)
    replay = [
        [[0, 0, 1], [6, 0, 1], [6, 6, 1], [0, 6, 1]],
        [[{"type": "getkey", "playerid": 0},
          {"type": "move", "playerid": 0, "pos": [0, 0, 1]}],
         [{"type": "move", "playerid": 1, "pos": [6, 0, 1]}],
         [{"type": "move", "playerid": 2, "pos": [6, 6, 1]}],
         [{"type": "move", "playerid": 3, "pos": [0, 6, 1]}]],
        {"0": 2, "1": 3, "2": 4, "3": 1},
    ]
    (art / "rank06-pair000-seat0.json").write_text(
        json.dumps(replay), encoding="utf-8")
    (run_dir / "matches.jsonl").write_text(
        json.dumps({"opponent": "rank06", "candidate_result": "loss",
                    "candidate_rank": 3, "candidate_score": 2, "turns": 1900,
                    "pair": 0, "candidate_seat": 0,
                    "replay": "artifacts/rank06-pair000-seat0.json"}) + "\n",
        encoding="utf-8")
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec())
    prompt = builder.build(version_before=None, act_id="r-act0001")["prompt"]
    assert "seat 0" in prompt.lower()
    assert "keys=" in prompt
    assert "rounds=" in prompt


def test_prompt_history_table_shows_partial_credit_cols(tmp_path):
    """A5: the match-history table includes avg_score and avg_turns columns."""
    cb = _codebase(tmp_path)
    run_dir = tmp_path / "runs" / "25_lostspace" / "hl-v1" / "run-001"
    run_dir.mkdir(parents=True)
    (run_dir / "matches.jsonl").write_text(
        '{"opponent":"rank06","candidate_result":"loss",'
        '"candidate_rank":3,"candidate_score":2,"turns":1900,"pair":0,'
        '"candidate_seat":0}\n', encoding="utf-8")
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec())
    prompt = builder.build(version_before=None, act_id="r-act0001")["prompt"]
    assert "avg_score" in prompt
    assert "avg_turns" in prompt


def test_prompt_has_data_schema_section(tmp_path):
    """The per-act prompt must carry a Data schema section stating interprops
    are int/object-coded (1=EscapeCapsule, 2=KeyMachine), that string
    membership is always False, and the blind-interact-then-check-success
    pattern. First-act prompt (no history) must still include it."""
    cb = HLCodebase(root=tmp_path / "ws", store=tmp_path / "store")
    (tmp_path / "ws").mkdir()
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec())
    prompt = builder.build(version_before=None, act_id="r-act0001")["prompt"]
    assert "Data schema" in prompt
    assert "1=EscapeCapsule" in prompt
    assert "2=KeyMachine" in prompt
    assert "always False" in prompt or "always false" in prompt
    assert "interact('KeyMachine')" in prompt

