import hashlib

from agentbench_frame.generals.paper_figure_v9 import (
    load_v9_figure_data,
    render_v9_paper_figures,
)

from .test_policy_kl_expanded_figure import write_figure_sources


def hashes(paths):
    return tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)


def test_renders_three_deterministic_english_png_svg_figures(tmp_path):
    data = load_v9_figure_data(*write_figure_sources(tmp_path / "sources"))

    first = render_v9_paper_figures(data, tmp_path / "figures")
    first_hashes = hashes(first)
    second = render_v9_paper_figures(data, tmp_path / "figures")

    assert len(first) == 6
    assert first == second
    assert first_hashes == hashes(second)
    assert all(path.stat().st_size > 1_000 for path in first)
    svg_text = "\n".join(path.read_text() for path in first if path.suffix == ".svg")
    for label in (
        "Scientific Attribution",
        "2×2 Attribution Policy Cells",
        "Formal Score over Heuristic-Learning Iterations",
        "Behavioral Information Gain",
        "legacy-12: 12 Reference States",
        "expanded-24: 24 Reference States",
        "epsilon = 0.001",
        "epsilon = 0.01",
        "epsilon = 0.05",
        "epsilon = 0.1",
        "Exact support |A(s)|",
        "v8→v9",
        "missing",
    ):
        assert label in svg_text
