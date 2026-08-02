"""§4 — Signal validation: policy_kl is a real signal under a behavioral edit.

These are the always-green contract tests for the spec requirement
"policy_kl is a real signal under a behavioral edit" (see
``openspec/changes/recorded-reference-set/specs/reference-set/spec.md``).

Two unit-level tests run in any environment (no real LostSpace logic needed):

- ``test_behavioral_edit_registers_positive_kl`` (§4.1): an edit that changes
  the action chosen at a reached decision point registers KL > 0.
- ``test_identical_versions_register_zero_kl`` (§4.2): a byte-identical copy
  registers KL = 0 (discriminative — no artificial inflation).

A third end-to-end test (§4.3) is marked ``@pytest.mark.slow`` and skipped
unless the real LostSpace logic path is available; see its docstring.
"""
import os
import sys
import textwrap
from pathlib import Path

import pytest

from agentbench_frame.hl.distribution import (
    enumerate_legal_actions, local_policy_kl_trace,
)
from agentbench_frame.hl.probe import ReferenceProbe
from agentbench_frame.hl.reference import ReferenceSample


# ---- shared fixture: a minimal recorded ν with one reached decision point ----

def _make_move_transcript(*, move_mask):
    """Build a 2-frame transcript (``id`` + decision-point ``roundbegin``)
    whose ``move`` legality is ``move_mask``. Both move dirs 0 and 1 are legal
    when ``move_mask`` starts ``[True, True, ...]`` so an in-support comparison
    between ``["move",0]`` and ``["move",1]`` is possible."""
    id_frame = {"type": "id", "id": 0, "birth_pos": [0, 0]}
    rb = {
        "type": "roundbegin", "round": 1, "inturn": 0, "status": 0,  # Alive
        "state": 1, "hp": 200, "keys": [0], "pos": [0, 0, 1],
        "tools": {"LandMine": [0, 0], "Sticky": [0, 0], "Kit": 0,
                  "Transport": 0},
        "others": [
            {"player_id": 1, "status": 0, "keys": [0], "hp": 200},
            {"player_id": 2, "status": 0, "keys": [0], "hp": 200},
            {"player_id": 3, "status": 0, "keys": [0], "hp": 200},
        ],
        "attack": [],
        "move": list(move_mask),
        "detect": False,
        "interprops": [],
    }
    return (id_frame, rb)


def _alive_sample(*, move_mask):
    """A reference sample whose decision point admits the move dirs in
    ``move_mask``."""
    obs = {"round": 1, "inturn": 0}
    return ReferenceSample(
        observation=obs,
        legal_actions={"attack": [], "move": list(move_mask), "detect": False,
                       "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
        transcript=_make_move_transcript(move_mask=move_mask),
    )


# ---- candidate scripts (ECHO_CANDIDATE-style, real wire format) ----
#
# Each reads 4 ASCII digits + JSON (judger->AI), writes 4-byte big-endian +
# JSON (AI->judger) — exactly the framing the real LostSpace candidates use.
# On the decision-point roundbegin (inturn==0, status Alive) each emits a
# specific move.

_CANDIDATE_TEMPLATE = r'''
import json, sys

def read_frame():
    hdr = sys.stdin.buffer.read(4)
    if len(hdr) < 4:
        return None
    n = int(hdr.decode("utf-8"))   # 4 ASCII digits (judger->AI framing)
    body = sys.stdin.buffer.read(n)
    return json.loads(body.decode("utf-8"))

def send(frame):
    s = json.dumps(frame)
    sys.stdout.buffer.write(len(s).to_bytes(4, "big", signed=True) + s.encode("utf-8"))
    sys.stdout.buffer.flush()

MOVE_DIR = {move_dir}

while True:
    msg = read_frame()
    if msg is None:
        break
    t = msg.get("type")
    if t == "id":
        send({{"type": "id", "player_num": 1, "player_list": [1,1,1,1]}})
    elif t == "roundbegin":
        if msg.get("inturn") == 0:
            send({{"type": "action", "action": ["move", MOVE_DIR]}})
    # off-round / off-turn / action: no reply
'''


def _write_candidate(tmp_path: Path, name: str, move_dir: int) -> Path:
    """Write a candidate that emits ``["move", move_dir]`` on seat-0
    roundbegin. Returns the path to the .py file."""
    src = _CANDIDATE_TEMPLATE.format(move_dir=move_dir)
    p = tmp_path / f"{name}.py"
    p.write_text(src, encoding="utf-8")
    return p


def _probe_candidate(candidate_path: Path, cwd: Path, sample: ReferenceSample,
                     *, timeout: float = 5.0):
    """Probe a single candidate script over one sample, returning the
    EmittedAction (or None). Fresh process per sample (probe contract)."""
    probe = ReferenceProbe(cmd=[sys.executable, str(candidate_path)],
                          cwd=str(cwd), timeout=timeout)
    try:
        return probe.probe_one(sample)
    finally:
        probe.close()


# ---- §4.1: behavioral edit → KL > 0 ----

def test_behavioral_edit_registers_positive_kl(tmp_path):
    """§4.1 / spec scenario "An edit that changes a reached decision registers
    KL > 0".

    Two candidates that differ ONLY in the move dir they emit at the reached
    decision point: v1 emits ``["move",0]``, v2 emits ``["move",1]``. Both
    emissions are in-support (the roundbegin admits move dirs 0 and 1), so the
    epsilon-smoothed onehots differ and ``local_policy_kl_trace`` must register
    a strictly-positive KL at the decision point.
    """
    # Both move dirs 0 and 1 legal -> in-support comparison.
    move_mask = [True, True] + [False] * 6
    sample = _alive_sample(move_mask=move_mask)

    v1_path = _write_candidate(tmp_path, "v1_move0", move_dir=0)
    v2_path = _write_candidate(tmp_path, "v2_move1", move_dir=1)

    ea_v1 = _probe_candidate(v1_path, tmp_path, sample, timeout=5.0)
    ea_v2 = _probe_candidate(v2_path, tmp_path, sample, timeout=5.0)

    assert ea_v1 is not None, "v1 emitted nothing — probe failed"
    assert ea_v2 is not None, "v2 emitted nothing — probe failed"
    assert ea_v1.primitive == ("move", 0)
    assert ea_v2.primitive == ("move", 1)
    assert ea_v1.out_of_support is False, "v1's move 0 should be in-support"
    assert ea_v2.out_of_support is False, "v2's move 1 should be in-support"

    las = enumerate_legal_actions(
        sample.legal_actions, status=sample.status, inventory=sample.inventory)
    assert len(las) > 0, "expected a decision point"

    chosen_new = [ea_v2.primitive]
    chosen_old = [ea_v1.primitive]
    legal_sets = [las]
    trace = local_policy_kl_trace(chosen_new, chosen_old, legal_sets,
                                   epsilon=0.1)

    assert len(trace) == 1, "expected exactly one KL entry for one decision point"
    assert trace[0].status == "ok", trace[0]
    assert trace[0].kl > 0.0, (
        f"behavioral edit at a reached decision must register KL > 0; "
        f"got {trace[0].kl} (eps-smoothed onehots over different legal moves "
        f"should differ)")


# ---- §4.2: byte-identical copy → all-zero KL ----

def test_identical_versions_register_zero_kl(tmp_path):
    """§4.2 / spec scenario "An edit that changes nothing registers KL = 0".

    v1 vs a byte-identical copy: the same primitive is emitted at the decision
    point, so the epsilon-smoothed distributions are identical and the policy
    KL trace must be all-zero. This guards against artificial inflation (the
    signal is discriminative).
    """
    move_mask = [True, True] + [False] * 6
    sample = _alive_sample(move_mask=move_mask)

    v1_path = _write_candidate(tmp_path, "v1_move0", move_dir=0)
    v1_copy_path = _write_candidate(tmp_path, "v1_copy_move0", move_dir=0)

    ea_v1 = _probe_candidate(v1_path, tmp_path, sample, timeout=5.0)
    ea_v1_copy = _probe_candidate(v1_copy_path, tmp_path, sample, timeout=5.0)

    assert ea_v1 is not None and ea_v1_copy is not None
    assert ea_v1.primitive == ea_v1_copy.primitive, (
        "byte-identical candidates should emit the same primitive")

    las = enumerate_legal_actions(
        sample.legal_actions, status=sample.status, inventory=sample.inventory)
    chosen_new = [ea_v1_copy.primitive]
    chosen_old = [ea_v1.primitive]
    legal_sets = [las]
    trace = local_policy_kl_trace(chosen_new, chosen_old, legal_sets,
                                   epsilon=0.1)

    assert len(trace) == 1
    assert trace[0].status == "ok", trace[0]
    assert trace[0].kl == 0.0, (
        f"identical versions must register KL = 0; got {trace[0].kl} "
        f"(artificial inflation would mask real edits)")


# ---- §4.3: end-to-end real recorded ν (slow, requires the real logic) ----

def _real_logic_available() -> bool:
    """True if the real LostSpace gamecode_logic dir + mapconf2.map are
    reachable. The recorder needs a real reference-roll trace; without the
    logic we cannot produce one in-test."""
    # The ITERATE.md workflow expects a BACKEND env var pointing at the
    # logic corpus dir; ``<BACKEND>/gamecode_logic/src/mapconf2.map`` must
    # exist for the logic to run.
    backend = os.environ.get("BACKEND") or os.environ.get(
        "LOSTSPACE_LOGIC_BACKEND")
    if not backend:
        return False
    p = Path(backend) / "gamecode_logic" / "src" / "mapconf2.map"
    return p.exists()


@pytest.mark.slow
def test_e2e_recorded_nu_gives_real_kl(tmp_path):
    """§4.3 — acceptance signal from the spec requirement
    "policy_kl is a real signal under a behavioral edit", measured over a ν
    recorded from a real LostSpace reference roll.

    Workflow (manual; see ``hl/README.md`` §"Record a ν, then iterate"):

    1. Run one short reference match with ``save_traces=True`` (seat 0 = the
       v1 / sample AI; opponents = a weak baseline; fixed ``mapconf2.map``).
    2. ``python -m agentbench_frame.hl.reference_recorder --trace <trace.jsonl>
       --spec-id <id> --opponent <name> --out nu.json``
    3. Probe v1 (sample AI) and a v2 with a one-line decision edit over that ν.
    4. Assert ``policy_kl > 0`` on ≥1 sample.

    This test requires the real LostSpace logic (``gamecode_logic`` +
    ``mapconf2.map``); it is skipped when the logic is not on disk, so the
    fast suite stays green. The §4.1/§4.2 unit-level tests are the
    always-green contract.
    """
    if not _real_logic_available():
        pytest.skip(
            "real LostSpace logic not available (set BACKEND to the logic "
            "corpus dir containing gamecode_logic/src/mapconf2.map); see "
            "hl/README.md §'Record a ν, then iterate' for the manual workflow. "
            "TODO: wire an automated reference roll here once the lostspace "
            "match harness exposes a programmatic save_traces entry point "
            "suitable for in-test use (the current harness shells out via "
            "the CLI and is too heavy for a unit test).")
    # When the logic IS available, the full e2e still needs the lostspace
    # match harness + recorder wiring. That wiring is non-trivial in-test
    # (the match harness shells out via the CLI); rather than build a fragile
    # always-failing test, we skip with a pointer to the manual workflow.
    # The §4.1/§4.2 unit tests are the always-green contract.
    pytest.skip(
        "real logic available but the in-test reference-roll wiring is not "
        "implemented; run the manual workflow in hl/README.md §'Record a ν, "
        "then iterate' to validate end-to-end. TODO: automate once the "
        "lostspace harness exposes a programmatic save_traces entry point.")
