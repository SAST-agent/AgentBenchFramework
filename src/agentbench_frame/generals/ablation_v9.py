"""Reproducible A/B/C/D policies for the clean-room v8 attribution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter

from .historical_policy import HistoricalPolicySource


@dataclass(frozen=True)
class AttributionPolicy:
    cell: str
    policy_id: str
    source: Path
    content_hash: str
    interventions: tuple[str, ...]
    source_authority: str


def _verify_policy(policy: HistoricalPolicySource) -> None:
    actual = LocalWorkspaceSnapshotter().capture(policy.source)
    if (
        actual.content_hash != policy.content_hash
        or actual.content_hash != policy.manifest.content_hash
        or actual.files != policy.manifest.files
    ):
        raise ValueError(
            f"{policy.version} source does not match its frozen manifest"
        )


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"{label} must occur exactly once in v7 preimage")
    return text.replace(old, new, 1)


def _add_large_stack_priority(text: str) -> str:
    text = _replace_once(
        text,
        "RECRUIT_RESERVE = 30\n",
        "RECRUIT_RESERVE = 30\nLARGE_STACK = 24\n",
        "large-stack constant preimage",
    )
    lines = text.splitlines(keepends=True)
    score_index = None
    for index in range(2, len(lines) - 2):
        if (
            lines[index - 2].strip()
            == "# Objective value, route cost, defender cost, attacking surplus,"
            and lines[index - 1].strip()
            == "# main exposure, row/column, direction."
            and lines[index].strip() == "score = ("
            and lines[index + 1].strip()
            == "_target_class(view, seat, objective),"
            and lines[index + 2].strip() == "route_cost,"
        ):
            if score_index is not None:
                raise ValueError(
                    "large-stack score preimage must occur exactly once"
                )
            score_index = index
    if score_index is None:
        raise ValueError("large-stack score preimage must occur exactly once")
    indent = lines[score_index][: -len(lines[score_index].lstrip())]
    lines.insert(
        score_index,
        f"{indent}force_priority = -amount if amount >= LARGE_STACK else 0\n",
    )
    target_line = score_index + 2
    target_indent = lines[target_line][: -len(lines[target_line].lstrip())]
    lines.insert(target_line + 1, f"{target_indent}force_priority,\n")
    return "".join(lines)


def _add_contact_before_economy(text: str) -> str:
    guard = "if _main_threatened(view, seat) or not _effect_fields_known(view):"
    text = _replace_once(
        text,
        guard,
        (
            "if _main_threatened(view, seat) or _contact_exists(view, seat) "
            "or not _effect_fields_known(view):"
        ),
        "contact upgrade guard preimage",
    )
    anchor = "\n\ndef _urgent_main_defense(view: dict, seat: int)"
    helper = '''

def _contact_exists(view: dict, seat: int) -> bool:
    """Whether an owned army has an immediately actionable hostile neighbor."""
    for key, cell in view["cells"].items():
        if int(cell["player"]) != seat or int(cell["army"]) <= 1:
            continue
        position = _position(key)
        if any(
            _hostile(other, seat)
            for _, _, other in _neighbors(view, seat, position)
        ):
            return True
    return False
'''
    return _replace_once(
        text,
        anchor,
        helper + anchor,
        "contact helper insertion preimage",
    )


def _materialize_exact(
    policy: HistoricalPolicySource,
    target_source: Path,
) -> str:
    snapshotter = LocalWorkspaceSnapshotter()
    result = snapshotter.materialize_manifest(
        policy.source,
        target_source,
        policy.manifest,
    )
    if result.content_hash != policy.content_hash:
        raise ValueError(f"{policy.version} exact materialization changed")
    return result.content_hash


def _materialize_ablation(
    v7: HistoricalPolicySource,
    root: Path,
    *,
    cell: str,
    policy_id: str,
    interventions: tuple[str, ...],
) -> AttributionPolicy:
    source = root / "source"
    _materialize_exact(v7, source)
    strategy_path = source / "strategy.py"
    strategy = strategy_path.read_text(encoding="utf-8")
    if "large_stack_priority" in interventions:
        strategy = _add_large_stack_priority(strategy)
    if "contact_before_economy" in interventions:
        strategy = _add_contact_before_economy(strategy)
    strategy_path.write_text(strategy, encoding="utf-8")
    (source / "ABLATION.md").write_text(
        "# Scientific attribution policy\n\n"
        f"Cell: {cell}\n\n"
        f"Frozen interventions: {', '.join(interventions)}\n",
        encoding="utf-8",
    )
    manifest = LocalWorkspaceSnapshotter().capture(source)
    LocalWorkspaceSnapshotter().write_manifest(manifest, root / "manifest.json")
    policy = AttributionPolicy(
        cell=cell,
        policy_id=policy_id,
        source=source.resolve(),
        content_hash=manifest.content_hash,
        interventions=interventions,
        source_authority="v7",
    )
    (root / "transformation.json").write_text(
        json.dumps(
            {
                **asdict(policy),
                "source": str(policy.source),
                "input_content_hash": v7.content_hash,
                "changed_files": ["ABLATION.md", "strategy.py"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return policy


def materialize_attribution_policies(
    v7: HistoricalPolicySource,
    v8: HistoricalPolicySource,
    destination: Path,
) -> tuple[AttributionPolicy, ...]:
    """Materialize the paired control, single-factor, and combined cells."""

    destination = Path(destination)
    if destination.exists():
        raise ValueError("attribution destination already exists")
    _verify_policy(v7)
    _verify_policy(v8)
    destination.mkdir(parents=True)

    a_source = destination / "A/source"
    a_hash = _materialize_exact(v7, a_source)
    a = AttributionPolicy(
        cell="A",
        policy_id="v7-control",
        source=a_source.resolve(),
        content_hash=a_hash,
        interventions=(),
        source_authority="v7",
    )
    b = _materialize_ablation(
        v7,
        destination / "B",
        cell="B",
        policy_id="v7-large-stack-only",
        interventions=("large_stack_priority",),
    )
    c = _materialize_ablation(
        v7,
        destination / "C",
        cell="C",
        policy_id="v7-contact-only",
        interventions=("contact_before_economy",),
    )
    d_source = destination / "D/source"
    d_hash = _materialize_exact(v8, d_source)
    d = AttributionPolicy(
        cell="D",
        policy_id="v8-combined",
        source=d_source.resolve(),
        content_hash=d_hash,
        interventions=("large_stack_priority", "contact_before_economy"),
        source_authority="v8",
    )

    snapshotter = LocalWorkspaceSnapshotter()
    snapshotter.write_manifest(v7.manifest, destination / "A/manifest.json")
    snapshotter.write_manifest(v8.manifest, destination / "D/manifest.json")
    for item, input_hash in ((a, v7.content_hash), (d, v8.content_hash)):
        root = destination / item.cell
        (root / "transformation.json").write_text(
            json.dumps(
                {
                    **asdict(item),
                    "source": str(item.source),
                    "input_content_hash": input_hash,
                    "changed_files": [],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return a, b, c, d
