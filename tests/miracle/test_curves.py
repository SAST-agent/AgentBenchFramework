"""曲线测试（要求 5）：构造数据 + 无数据如实报告 + SVG 渲染。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentbench_frame.miracle.curves import (
    build_curves,
    curves_ascii,
    curves_to_svg,
    save_curves,
)


def _write_run(tmp_path, iteration, version, score, ig, seed=11):
    d = tmp_path / "24_miracle" / "sample" / f"iter{iteration}_{version}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({
        "iteration": iteration, "agent_version": version, "seed": seed,
        "score": score, "winner": 0 if score >= 1 else 1,
        "episode_ig": ig, "ig_coverable_ratio": 1.0,
        "ig_missing": {"support_expansion": 0},
    }), encoding="utf-8")


def test_build_curves_version_aligned(tmp_path):
    _write_run(tmp_path, 0, "sample", 30000, None)
    _write_run(tmp_path, 1, "sample_v2", 0, 1.9474)
    curves = build_curves(runs_root=tmp_path, agent="sample")
    assert curves["no_data"] is False
    assert len(curves["points"]) == 2
    by_iter = {p["iteration"]: p for p in curves["points"]}
    assert by_iter[0] == {"iteration": 0, "version": "sample", "seed": 11,
                          "score": 30000, "winner": 0, "episode_ig": None,
                          "ig_coverable_ratio": 1.0,
                          "ig_missing": {"support_expansion": 0}}
    assert by_iter[1]["version"] == "sample_v2"
    assert by_iter[1]["episode_ig"] == 1.9474


def test_no_data_reports_honestly(tmp_path):
    curves = build_curves(runs_root=tmp_path, agent="nobody")
    assert curves["no_data"] is True
    assert "无真实迭代数据" in curves["note"]
    text = curves_ascii(curves)
    assert "无数据" in text
    svg = curves_to_svg(curves)
    assert "无真实迭代数据" in svg


def test_curves_to_svg_renders_points(tmp_path):
    _write_run(tmp_path, 0, "sample", 30000, None)
    _write_run(tmp_path, 1, "sample_v2", 0, 1.9474)
    curves = build_curves(runs_root=tmp_path, agent="sample")
    svg = curves_to_svg(curves)
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert "score(蓝)" in svg
    assert "30000" in svg and "1.947" in svg
    assert "iter0" in svg and "iter1" in svg
    assert "sample_v2" in svg


def test_save_curves_writes_json_and_svg(tmp_path):
    _write_run(tmp_path, 0, "sample", 30000, None)
    _write_run(tmp_path, 1, "sample_v2", 0, 1.9474)
    curves = build_curves(runs_root=tmp_path, agent="sample")
    jp = tmp_path / "out" / "curves.json"
    sp = tmp_path / "out" / "curves.svg"
    save_curves(curves, jp, sp)
    assert jp.exists() and sp.exists()
    data = json.loads(jp.read_text(encoding="utf-8"))
    assert len(data["points"]) == 2
    assert sp.read_text(encoding="utf-8").startswith("<svg")


def test_ascii_table_columns(tmp_path):
    _write_run(tmp_path, 0, "sample", 30000, None)
    curves = build_curves(runs_root=tmp_path, agent="sample")
    text = curves_ascii(curves)
    assert "iteration | version" in text
    assert "30000" in text
