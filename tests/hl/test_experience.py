"""Tests for the self-summarized experience store (HL std 5) and the
consolidation/growth-nudge machinery in ContextBuilder (HL std 4)."""
from __future__ import annotations

from pathlib import Path

from agentbench_frame.hl.codebase import HLCodebase
from agentbench_frame.hl.context import ContextBuilder
from agentbench_frame.hl.experience import ExperienceStore
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


# ---- ExperienceStore ----

def test_experience_propose_update_appends_and_renders_fallback(tmp_path):
    store = ExperienceStore(round_root=tmp_path)
    store.propose_update(
        act_id="r-000001",
        feedback={"edit_type": "add_rule", "win_rate": 0.0, "avg_rank": 3.0,
                  "avg_score": 2.0, "active_opponents": ("rank06",),
                  "kl_mean": 0.5, "n_changed": 1, "n_total": 3,
                  "occupancy_shift": 0.1},
        seat0_digest=None,
    )
    obs = store.observations()
    assert len(obs) == 1
    assert obs[0]["act_id"] == "r-000001"
    assert obs[0]["win_rate"] == 0.0

    # No EXPERIENCE.md yet -> render falls back to the raw observations.
    rendered = store.render()
    assert "raw observations" in rendered
    assert "r-000001" in rendered
    assert "rank06" in rendered


def test_experience_render_prefers_authored_doc(tmp_path):
    store = ExperienceStore(round_root=tmp_path)
    store.propose_update(act_id="r-000001",
                         feedback={"win_rate": 0.0, "active_opponents": ()},
                         seat0_digest=None)
    # The agent authors the distilled doc on a consolidation act.
    store.doc_path.write_text(
        "## Lessons\n- against rank06: rush corner keys early\n"
        "## Retired ideas\n- detect spam: never triggers\n",
        encoding="utf-8",
    )
    rendered = store.render()
    assert "Lessons" in rendered
    assert "rank06" in rendered
    assert "Retired ideas" in rendered
    # authored doc shadows the raw observations
    assert "raw observations" not in rendered


def test_experience_render_caps_long_doc(tmp_path):
    store = ExperienceStore(round_root=tmp_path)
    store.doc_path.write_text("\n".join(f"line {i}" for i in range(100)),
                              encoding="utf-8")
    rendered = store.render(max_lines=10)
    assert rendered.count("\n") <= 11  # 10 lines + the 'more lines' notice
    assert "more lines" in rendered


def test_experience_render_empty_when_nothing_yet(tmp_path):
    store = ExperienceStore(round_root=tmp_path)
    assert store.render() == ""


# ---- ContextBuilder: interpretable-code license (std 3) ----

def test_prompt_licenses_interpretable_code_beyond_if_else(tmp_path):
    cb = _codebase(tmp_path)
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec())
    prompt = builder.build(version_before=None, act_id="r-act0001")["prompt"]
    assert "utility/scoring" in prompt
    assert "lookahead" in prompt
    assert "HUMAN-READABLE" in prompt or "human-readable" in prompt


# ---- ContextBuilder: consolidation act (std 4) ----

def test_consolidation_mission_on_every_kth_act(tmp_path):
    cb = _codebase(tmp_path)
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec(),
                             consolidate_every=2)
    # act 1: normal mission
    p1 = builder.build(version_before=None, act_id="a1", act_index=1)["prompt"]
    assert "What to do now" in p1
    assert "CONSOLIDATION ACT" not in p1
    # act 2: consolidation mission
    p2 = builder.build(version_before=None, act_id="a2", act_index=2)["prompt"]
    assert "CONSOLIDATION ACT" in p2
    assert "compress" in p2.lower()
    assert "NOT add new behavior" in p2 or "Do NOT add new behavior" in p2
    # act 3: back to normal
    p3 = builder.build(version_before=None, act_id="a3", act_index=3)["prompt"]
    assert "CONSOLIDATION ACT" not in p3


def test_consolidation_mission_points_at_experience_file(tmp_path):
    cb = _codebase(tmp_path)
    store = ExperienceStore(round_root=tmp_path / "round")
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec(),
                             consolidate_every=2, experience=store)
    p = builder.build(version_before=None, act_id="a2", act_index=2)["prompt"]
    assert "Re-summarize the experience file" in p
    assert str(store.path) in p


def test_consolidate_every_zero_disables(tmp_path):
    cb = _codebase(tmp_path)
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec(),
                             consolidate_every=0)
    p = builder.build(version_before=None, act_id="a4", act_index=4)["prompt"]
    assert "CONSOLIDATION ACT" not in p


# ---- ContextBuilder: code-growth nudge (std 4) ----

def test_growth_nudge_triggers_on_piling_growth(tmp_path):
    cb = _codebase(tmp_path)
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec(),
                             max_growth_pct=40.0)
    fb = {"win_rate": 0.0, "avg_rank": 3.0, "avg_score": 2.0,
          "avg_turns": 100, "evaluation_status": "complete",
          "kl_mean": 0.1, "n_changed": 1, "n_total": 3,
          "occupancy_shift": 0.0, "edit_type": "add_rule",
          "active_opponents": ("rank06",)}
    # 100 -> 200 lines (+100%) over two add_rule acts
    prompt = builder.build(
        version_before=None, act_id="a3", act_index=3, prev_feedback=fb,
        loc_history=[100, 150, 200],
        edit_type_history=["add_rule", "add_rule", "add_rule"],
    )["prompt"]
    assert "CODE GROWTH" in prompt
    assert "consolidation" in prompt.lower()


def test_growth_nudge_silent_below_threshold(tmp_path):
    cb = _codebase(tmp_path)
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec(),
                             max_growth_pct=40.0)
    fb = {"win_rate": 0.0, "edit_type": "add_rule"}
    prompt = builder.build(
        version_before=None, act_id="a3", act_index=3, prev_feedback=fb,
        loc_history=[100, 105, 110],   # +10%, below threshold
        edit_type_history=["add_rule", "add_rule", "add_rule"],
    )["prompt"]
    assert "CODE GROWTH" not in prompt


def test_growth_nudge_silent_when_only_one_piling_act(tmp_path):
    cb = _codebase(tmp_path)
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec(),
                             max_growth_pct=40.0)
    fb = {"win_rate": 0.0, "edit_type": "add_rule"}
    prompt = builder.build(
        version_before=None, act_id="a2", act_index=2, prev_feedback=fb,
        loc_history=[100, 500],            # huge growth, but only 1 piling edit
        edit_type_history=["refactor", "add_rule"],
    )["prompt"]
    assert "CODE GROWTH" not in prompt


# ---- ContextBuilder: experience injected into prompt (std 5) ----

def test_prompt_injects_lessons_when_experience_present(tmp_path):
    cb = _codebase(tmp_path)
    store = ExperienceStore(round_root=tmp_path / "round")
    store.doc_path.parent.mkdir(parents=True, exist_ok=True)
    store.doc_path.write_text(
        "## Lessons\n- against rank06: rush corner keys early\n",
        encoding="utf-8",
    )
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec(), experience=store)
    prompt = builder.build(version_before=None, act_id="a1", act_index=1)["prompt"]
    assert "Lessons learned so far" in prompt
    assert "rush corner keys early" in prompt
    assert str(store.path) in prompt


def test_prompt_no_lessons_section_when_experience_disabled(tmp_path):
    cb = _codebase(tmp_path)
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec(), experience=None)
    prompt = builder.build(version_before=None, act_id="a1", act_index=1)["prompt"]
    assert "Lessons learned so far" not in prompt
