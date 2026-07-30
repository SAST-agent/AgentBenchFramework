# AgentBench Ludemic Rule Description Complexity

Source: [Aoraku/AgentBench](https://github.com/Aoraku/AgentBench) at `b581bca3ba3d2d7d58a2f8c6bbddd060fc7fdc87`.

The primary value is the cardinality of the disjoint set of atomic rule-proposition occurrences in each formal AB-Rule/1 description. Descriptions are parsed but not compiled, and expression AST shape is not counted.

These values are language-relative, best-known formal description lengths. They are not exact Kolmogorov complexity, implementation size, state-space size, strategic depth, learning difficulty, or information gain.

| Rank | Game ID | Game | Rule atoms | State | Actions | Observations | Setup | Conditions | Transitions | Outcomes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `26_snakego` | SnakeGo | 145 | 42 | 11 | 8 | 4 | 33 | 41 | 6 |
| 2 | `25_aquawar` | AquaWar | 171 | 43 | 12 | 11 | 4 | 39 | 55 | 7 |
| 3 | `29_rollman` | Rollman | 181 | 66 | 4 | 11 | 6 | 29 | 61 | 4 |
| 4 | `25_lostspace` | LostSpace | 196 | 68 | 19 | 9 | 9 | 43 | 46 | 2 |
| 5 | `24_miracle` | Miracle | 203 | 74 | 14 | 7 | 7 | 43 | 54 | 4 |
| 6 | `23_doto` | DOTO | 206 | 72 | 12 | 10 | 8 | 52 | 50 | 2 |
| 7 | `27_antwar` | AntWar | 211 | 58 | 19 | 10 | 5 | 49 | 64 | 6 |
| 8 | `28_generals` | Generals | 225 | 65 | 31 | 9 | 12 | 52 | 50 | 6 |
| 9 | `30_antwar2` | AntWar2 | 405 | 120 | 20 | 10 | 5 | 102 | 142 | 6 |

## Boundary

DeepClue is excluded. The descriptions include abstract state, setup, legal action families, transitions, chance, observations, terminal conditions, and outcomes. They exclude communication, rendering, replay serialization, logging, and implementation optimizations.

AB-Ludi/1 `k_upper_bits` remains a separate implementation-description metric and must not be compared numerically with AB-Rule/1 rule atoms.
