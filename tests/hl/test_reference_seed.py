"""Tests for hl/reference_seed.py — hand-authored seed ν (TEST FIXTURE ONLY).

The seed is now a **test fixture**, not a production ν source (see the module
docstring of ``reference_seed.py``). Its samples carry ``transcript=()`` —
fine for schema / round-trip / count unit tests, but the ``ReferenceProbe``
explicitly rejects them with ``ReferenceSampleError`` (re-record this ν). The
production ν is produced by ``reference_recorder`` from a real match trace.

These tests cover the seed's STRUCTURAL contract (8 samples, frozen,
round-trips, non-empty legal-action enumeration) — NOT a probe contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentbench_frame.hl.distribution import enumerate_legal_actions
from agentbench_frame.hl.probe import ReferenceProbe, ReferenceSampleError
from agentbench_frame.hl.reference import ReferenceStateSet
from agentbench_frame.hl.reference_seed import build_seed, _seed_samples


def test_seed_has_eight_decision_points_across_situations():
    samples = _seed_samples()
    assert len(samples) == 8
    for s in samples:
        # Alive or WAIT_FOR_ESCAPE are both decision points (the latter allows
        # only the escape-capsule interact + finish).
        assert s.status in (0, 4)
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
    assert len(loaded) == 8
    assert loaded == nu


def test_seed_samples_enumerate_to_nonempty_action_sets():
    """enumerate_legal_actions must yield a non-empty A(s) for each seed
    sample (otherwise it's not a real decision point and KL is vacuous)."""
    for s in _seed_samples():
        las = enumerate_legal_actions(
            s.legal_actions, status=s.status, inventory=s.inventory)
        assert len(las) > 0, "seed sample produced empty action set"


def test_seed_samples_carry_empty_transcript():
    """The seed is a TEST FIXTURE: every sample has ``transcript == ()`` (the
    legacy single-frame shape). The production ν is produced by the recorder
    with real transcripts. This invariant is what makes the probe reject the
    seed — do NOT add transcripts to the seed samples."""
    for s in _seed_samples():
        assert s.transcript == ()


def test_probe_rejects_seed_samples_with_reference_sample_error():
    """The seed is NOT a valid probe input: every sample is rejected by
    ``ReferenceProbe._probe_one_impl`` with ``ReferenceSampleError`` (the
    §1.3 fail-fast). This calls ``_probe_one_impl`` directly — the transcript
    check raises before any candidate process is spawned, so no real candidate
    is needed. ``probe_one`` itself re-raises the same error (verified in
    ``tests/hl/test_probe.py::test_probe_rejects_missing_transcript``).

    This asserts the seed's DEMOTED status: it is a structural fixture, not a
    production ν. Re-record ν via ``reference_recorder`` for real iteration.
    """
    # A dummy probe — never spawns a process; the transcript check fires first.
    probe = ReferenceProbe(cmd=["true"], cwd=".", timeout=1.0)
    for s in _seed_samples():
        las = enumerate_legal_actions(
            s.legal_actions, status=s.status, inventory=s.inventory)
        # status=4 (WAIT_FOR_ESCAPE) samples have no legal actions — skip them
        # (probe_one returns None for those before reaching the transcript check).
        if len(las) == 0:
            continue
        with pytest.raises(ReferenceSampleError, match="re-record"):
            probe._probe_one_impl(s, las)
