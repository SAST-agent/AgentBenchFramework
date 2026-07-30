"""Demo: round-robin between 3 agents → payoff matrix + Elo.

    python -m snakego.arena_demo
"""
from __future__ import annotations

from .arena import Arena
from .agents import RandomAgent, GreedyAgent


def main() -> int:
    # three agents of escalating strength
    agents = {
        "random": RandomAgent(seed=0),
        "greedy": GreedyAgent(),
    }
    arena = Arena(agents)

    print("=== SnakeGo Arena: round-robin (4 seeds, swapped sides) ===\n")
    result = arena.round_robin(n_seeds=4, base_seed=0, verbose=True)

    # payoff matrix
    names = result.payoff.agents
    print("\nPayoff matrix  (row win-rate vs column)\n")
    hdr = "          " + "".join(f"{n:>10}" for n in names)
    print(hdr)
    print("-" * len(hdr))
    for i, a in enumerate(names):
        row = "    "
        for b in names:
            if a == b:
                row += "       ---"
            else:
                row += f"{result.payoff.win_rate[a][b]:>10.1%}"
        print(f"{a:>10}{row}")

    # Elo
    print("\nElo ranking\n")
    for rank, name, elo in result.elo.rankings():
        print(f"  #{rank}  {name:<12} Elo={elo}")

    # match count
    print(f"\n{len(result.matches)} games played total")

    # sanity assertions
    wr_g_r = result.payoff.win_rate["greedy"]["random"]
    assert wr_g_r >= 0.5, f"greedy should beat random (got {wr_g_r:.1%})"
    el_g = result.elo.get("greedy")
    el_r = result.elo.get("random")
    assert el_g > el_r, "greedy Elo should exceed random"
    print("\nARENA ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
