"""Tests for hl/reference_seed.py — hand-authored seed ν."""
from __future__ import annotations

import json
from pathlib import Path

from agentbench_frame.hl.reference import ReferenceStateSet
from agentbench_frame.hl.reference_seed import build_seed, _seed_samples


def test_seed_has_three_decision_points():
    samples = _seed_samples()
    assert len(samples) == 3
    for s in samples:
        assert s.status == 0          # all Alive (decision points)
        assert "legal_actions" in s.legal_actions or s.legal_actions
        assert "move" in s.legal_actions
        assert "attack" in s.legal_actions


def test_legal_actions_match_get_legal_actions_shape():
    """Each sample's legal_actions dict has the exact keys the logic returns."""
    for s in _seed_samples():
        la = s.legal_actions
        assert set(la.keys()) == {"attack", "move", "detect", "interprops"}
        assert isinstance(la["move"], list) and len(la["move"]) == 8
        assert all(isinstance(b, bool) for b in la["move"])
        assert isinstance(la["attack"], list)
        assert isinstance(la["detect"], bool)
        assert isinstance(la["interprops"], list)


def test_build_seed_roundtrips_through_save_load(tmp_path: Path):
    nu = build_seed("hl-seed-v1")
    p = tmp_path / "nu.json"
    nu.save(p)
    loaded = ReferenceStateSet.load(p)
    assert loaded.spec_id == "hl-seed-v1"
    assert len(loaded) == 3
    assert loaded == nu


def test_seed_samples_enumerate_to_nonempty_action_sets():
    """enumerate_legal_actions must yield a non-empty A(s) for each seed
    sample (otherwise it's not a real decision point and KL is vacuous)."""
    from agentbench_frame.hl.distribution import enumerate_legal_actions
    for s in _seed_samples():
        las = enumerate_legal_actions(
            s.legal_actions, status=s.status, inventory=s.inventory)
        assert len(las) > 0, "seed sample produced empty action set"
