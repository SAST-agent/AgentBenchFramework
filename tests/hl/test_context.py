"""Tests for hl/context.py — ContextBuilder prompt assembly."""
from __future__ import annotations

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
    # bounds: don't read outside the workspace
    assert "outside this workspace" in prompt


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
