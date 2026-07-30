"""Demo: record a game → save JSON replay → build HTML viewer.

    python -m snakego.replay_demo
"""
from __future__ import annotations

from pathlib import Path

from .agents import RandomAgent, GreedyAgent
from .replay import play_with_replay
from .visualizer import build_html


def main() -> int:
    out_dir = Path("outputs/snakego_replays")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== SnakeGo replay demo ===\n")

    # play a game and record
    p0 = GreedyAgent()
    p1 = RandomAgent(seed=5)
    result, rec = play_with_replay(p0, p1, seed=5)
    print(f"game: {p0.name} vs {p1.name}")
    print(f"  winner=P{result['winner']} score={result['scores']} "
          f"turns={result['turns']} steps={result['steps']}")
    print(f"  recorded {len(rec.frames)} frames")

    # save JSON replay
    json_path = rec.save(out_dir / "greedy_vs_random.json")
    print(f"\nsaved replay JSON: {json_path}")

    # build HTML viewer
    html_path = build_html(json_path, out_dir / "greedy_vs_random.html")
    print(f"saved HTML viewer: {html_path}")

    # sanity checks
    import json
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["n_frames"] == len(rec.frames)
    assert len(data["frames"]) > 0
    f0 = data["frames"][0]
    assert "snake_map" in f0 and "wall_map" in f0 and "snakes" in f0
    assert len(f0["snakes"]) >= 2, "should start with 2 snakes"
    html = html_path.read_text(encoding="utf-8")
    assert "REPLAY" in html and "frames" in html
    assert html.startswith("<!DOCTYPE html>")

    print("\nREPLAY DEMO PASSED")
    print(f"\nOpen this in your browser:\n  {html_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
