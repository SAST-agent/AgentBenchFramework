import hashlib
import json
from pathlib import Path

import pytest

from agentbench_frame.generals.paper_figure_v9 import load_v9_figure_data


EPSILONS = ("0.001", "0.01", "0.05", "0.1")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def transitions(count):
    return [
        {
            "version_before": f"v{index}",
            "version_after": f"v{index + 1}",
            "mean_kl_nats": float(index),
            "mean_kl_nats_decimal": str(index),
            "coverage": {"complete": count, "total": count},
            "action_disagreement_rate": index / 10,
            "sensitivity": {
                epsilon: {
                    "mean_kl_nats": float(index) * (epsilon_index + 1),
                    "coverage": {"complete": count, "total": count},
                }
                for epsilon_index, epsilon in enumerate(EPSILONS)
            },
        }
        for index in range(9)
    ]


def domain(domain_id, count):
    return {
        "domain_id": domain_id,
        "metric": "controlled_reference_policy_kl",
        "primary_epsilon": "0.01",
        "epsilons": list(EPSILONS),
        "reference_state_count": count,
        "support_size": {"complete": count, "total": count},
        "transitions": transitions(count),
    }


def write_figure_sources(tmp_path):
    attribution = tmp_path / "attribution"
    v9 = tmp_path / "v9"
    legacy = tmp_path / "legacy"
    expanded = tmp_path / "expanded"
    report = {
        "status": "complete",
        "policies": [
            {"cell": cell, "content_hash": cell.lower() * 64}
            for cell in "ABCD"
        ],
        "attribution": {
            "metrics": {
                "outcome_score": {
                    "large_stack": 0.1,
                    "contact": None,
                    "interaction": -0.05,
                    "complete_pair_count": 12,
                }
            }
        },
    }
    report_path = attribution / "diagnosis/report.json"
    write_json(report_path, report)
    report_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
    write_json(
        attribution / "summary.json",
        {
            "run_id": attribution.name,
            "status": "complete",
            "report_hash": report_hash,
        },
    )
    write_json(
        v9 / "summary.json",
        {
            "run_id": v9.name,
            "status": "complete",
            "runnable": True,
            "candidate_hash": "9" * 64,
            "diagnosis_report_hash": report_hash,
            "score_history": [0.0, 0.1, None, 0.2, 0.3, 0.4, 0.5, 0.6, 0.55, 0.65, 0.7],
        },
    )
    legacy_metric = {
        key: value
        for key, value in domain("legacy-12", 12).items()
        if key != "domain_id"
    }
    legacy_metric["transitions"] = legacy_metric["transitions"][:8]
    write_json(
        legacy / "summary.json",
        {
            "run_id": legacy.name,
            "status": "complete",
            "controlled_reference_policy_kl": legacy_metric,
        },
    )
    write_json(
        expanded / "summary.json",
        {
            "run_id": expanded.name,
            "status": "complete",
            "source_run_id": legacy.name,
            "target_run_id": v9.name,
            "target_content_hash": "9" * 64,
            "domains": {
                "legacy-12": domain("legacy-12", 12),
                "expanded-24": domain("expanded-24", 24),
            },
        },
    )
    (expanded / "events.jsonl").write_text(
        "".join(
            json.dumps({
                "event_type": "action_space_count",
                "domain_id": "expanded-24",
                "measurement_state_id": f"state-{index:02d}",
                "support_size": str(7 + index),
                "status": "complete",
            }) + "\n"
            for index in range(24)
        )
    )
    return attribution, v9, legacy, expanded


def test_loads_separate_legacy12_and_expanded24_domains(tmp_path):
    data = load_v9_figure_data(*write_figure_sources(tmp_path))

    assert data.legacy.domain_id == "legacy-12"
    assert data.legacy.reference_state_count == 12
    assert data.expanded.domain_id == "expanded-24"
    assert data.expanded.reference_state_count == 24
    assert data.expanded.transitions[-1] == "v8→v9"
    assert tuple(data.expanded.sensitivity) == EPSILONS
    assert len(data.expanded.support_sizes) == 24


def test_paper_loader_refuses_domain_splicing(tmp_path):
    sources = write_figure_sources(tmp_path)
    summary_path = sources[-1] / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["domains"]["legacy-12"]["transitions"][0]["mean_kl_nats"] = 99
    write_json(summary_path, summary)

    with pytest.raises(ValueError, match="reference domain"):
        load_v9_figure_data(*sources)


def test_loader_preserves_historical_score_gap(tmp_path):
    data = load_v9_figure_data(*write_figure_sources(tmp_path))

    assert data.score_history[2] is None
    assert data.score_history[1] == 0.1
    assert data.score_history[3] == 0.2
