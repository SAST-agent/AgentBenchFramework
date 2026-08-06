from pathlib import Path

import pytest

from agentbench_frame.generals.historical_policy import HistoricalPolicySource
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


V7_STRATEGY = '''"""Fixture v7 policy."""

RECRUIT_RESERVE = 30

def _neighbors(view, seat, position):
    return ()

def _hostile(cell, seat):
    return False

def _position(key):
    return tuple(map(int, key.split(",")))

def _main(view, seat):
    return None

def _effect_fields_known(view):
    return True

def _main_threatened(view: dict, seat: int) -> bool:
    main = _main(view, seat)
    if main is None:
        return False
    position = tuple(main["position"])
    return any(_hostile(cell, seat) for _, _, cell in _neighbors(view, seat, position))


def _urgent_main_defense(view: dict, seat: int):
    return None


def _best_valuable_move(view: dict, seat: int, used_sources: set):
    amount = 30
    required = 0
    route_cost = 2
    first_cell = {"army": 1, "player": seat}
    source = (1, 1)
    first = (1, 2)
    main_position = (0, 0)
    objective = (2, 2)
    DIRECTION = {(0, 1): 4}
    def _target_class(*args): return 0
    def _move(*args): return [8]
    sent = amount if required == 0 else required
    # Objective value, route cost, defender cost, attacking surplus,
    # main exposure, row/column, direction.
    score = (
        _target_class(view, seat, objective),
        route_cost,
        int(first_cell["army"]) if int(first_cell["player"]) != seat else 0,
        -(amount - required),
        source == main_position,
        source,
        DIRECTION[(first[0] - source[0], first[1] - source[1])],
    )
    return score, _move(source, first, sent)


def _upgrade_candidate(view: dict, seat: int, round_number: int):
    """Return command, cost, field, new level under exact official tables."""
    if _main_threatened(view, seat) or not _effect_fields_known(view):
        return None
    return [8]


def choose_actions(*args):
    return [[8]]
'''


def _policy(tmp_path: Path, version: str, strategy: str) -> HistoricalPolicySource:
    root = tmp_path / version
    source = root / "source"
    (source / "tests").mkdir(parents=True)
    (source / "main.py").write_text("from strategy import choose_actions\n", encoding="utf-8")
    (source / "state_view.py").write_text("def normalize(x): return x\n", encoding="utf-8")
    (source / "strategy.py").write_text(strategy, encoding="utf-8")
    (source / "STRATEGY.md").write_text(f"{version} strategy\n", encoding="utf-8")
    (source / "EXPERIENCE.md").write_text(f"{version} experience\n", encoding="utf-8")
    (source / "tests/test_strategy.py").write_text("def test_ok(): assert True\n", encoding="utf-8")
    snapshotter = LocalWorkspaceSnapshotter()
    manifest = snapshotter.capture(source)
    return HistoricalPolicySource(
        version=version,
        run_id=f"run-{version}",
        content_hash=manifest.content_hash,
        source=source,
        manifest=manifest,
    )


def test_attribution_cells_isolate_the_two_v8_interventions(tmp_path):
    from agentbench_frame.generals.ablation_v9 import (
        materialize_attribution_policies,
    )

    v7 = _policy(tmp_path, "v7", V7_STRATEGY)
    v8_strategy = V7_STRATEGY.replace(
        "RECRUIT_RESERVE = 30\n",
        "RECRUIT_RESERVE = 30\nLARGE_STACK = 24\n",
    )
    v8 = _policy(tmp_path, "v8", v8_strategy)

    policies = materialize_attribution_policies(
        v7,
        v8,
        tmp_path / "materialized",
    )
    by_cell = {item.cell: item for item in policies}
    sources = {
        cell: (item.source / "strategy.py").read_text(encoding="utf-8")
        for cell, item in by_cell.items()
    }

    assert tuple(by_cell) == ("A", "B", "C", "D")
    assert by_cell["A"].content_hash == v7.content_hash
    assert by_cell["D"].content_hash == v8.content_hash
    assert by_cell["A"].interventions == ()
    assert by_cell["B"].interventions == ("large_stack_priority",)
    assert by_cell["C"].interventions == ("contact_before_economy",)
    assert by_cell["D"].interventions == (
        "large_stack_priority",
        "contact_before_economy",
    )
    assert "LARGE_STACK = 24" not in sources["A"]
    assert "LARGE_STACK = 24" in sources["B"]
    assert "def _contact_exists" not in sources["B"]
    assert "def _contact_exists" in sources["C"]
    assert "LARGE_STACK = 24" not in sources["C"]
    assert "force_priority" not in sources["C"]
    compile(sources["B"], "B/strategy.py", "exec")
    compile(sources["C"], "C/strategy.py", "exec")


def test_attribution_materialization_is_byte_stable(tmp_path):
    from agentbench_frame.generals.ablation_v9 import (
        materialize_attribution_policies,
    )

    v7 = _policy(tmp_path, "v7", V7_STRATEGY)
    v8 = _policy(tmp_path, "v8", V7_STRATEGY + "\nV8 = True\n")

    first = materialize_attribution_policies(v7, v8, tmp_path / "first")
    second = materialize_attribution_policies(v7, v8, tmp_path / "second")

    assert [item.content_hash for item in first] == [
        item.content_hash for item in second
    ]


def test_attribution_materialization_rejects_changed_v7_preimage(tmp_path):
    from agentbench_frame.generals.ablation_v9 import (
        materialize_attribution_policies,
    )

    changed = V7_STRATEGY.replace("RECRUIT_RESERVE = 30", "RECRUIT_RESERVE = 31")
    v7 = _policy(tmp_path, "v7", changed)
    v8 = _policy(tmp_path, "v8", V7_STRATEGY)

    with pytest.raises(ValueError, match="large-stack constant preimage"):
        materialize_attribution_policies(v7, v8, tmp_path / "materialized")


def test_attribution_materialization_rejects_existing_destination(tmp_path):
    from agentbench_frame.generals.ablation_v9 import (
        materialize_attribution_policies,
    )

    v7 = _policy(tmp_path, "v7", V7_STRATEGY)
    v8 = _policy(tmp_path, "v8", V7_STRATEGY)
    destination = tmp_path / "materialized"
    destination.mkdir()

    with pytest.raises(ValueError, match="destination already exists"):
        materialize_attribution_policies(v7, v8, destination)
