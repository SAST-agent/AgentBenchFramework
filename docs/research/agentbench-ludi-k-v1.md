# AgentBench AB-Ludi/1 K-Complexity Upper Bounds

Source: [https://github.com/Aoraku/AgentBench](https://github.com/Aoraku/AgentBench) at `b581bca3ba3d2d7d58a2f8c6bbddd060fc7fdc87`.

These values are **not exact Kolmogorov complexity**. They are conditional executable-description upper bounds under the fixed AB-Ludi/1 source-module reference machine.

`k_upper_bits` is eight times the byte length of the canonical ludeme tree compressed with the report's fixed zlib-9 profile. The reference-machine ID binds an enforced multi-vector behavior fingerprint, and each v1 game must match its frozen complete compressed-stream SHA-256 and length. The shared decoder and language runtimes are conditioned out.

$$
K(G \mid U_{\mathrm{AB\text{-}Ludi/1}}, R) \leq 8\,\left|\operatorname{zlib9}(\operatorname{encode}_{\mathrm{AB\text{-}Ludi/1}}(G))\right| + \mathcal{O}(1)
$$

| Rank | Game ID | Game | Modules | Source bits | K upper bits | Ratio |
|---:|---|---|---:|---:|---:|---:|
| 1 | `26_snakego` | SnakeGo | 9 | 328104 | 79624 | 0.2407 |
| 2 | `29_rollman` | Rollman | 8 | 498976 | 101920 | 0.2034 |
| 3 | `23_doto` | DOTO | 7 | 2008656 | 113384 | 0.0564 |
| 4 | `28_generals` | Generals | 12 | 734232 | 126888 | 0.1720 |
| 5 | `24_miracle` | Miracle | 23 | 791648 | 144200 | 0.1805 |
| 6 | `27_antwar` | AntWar | 22 | 810024 | 170528 | 0.2091 |
| 7 | `25_aquawar` | AquaWar | 11 | 986072 | 178632 | 0.1806 |
| 8 | `25_lostspace` | LostSpace | 18 | 685304 | 179032 | 0.2594 |
| 9 | `30_deepclue` | DeepClue | 17 | 685736 | 189496 | 0.2742 |
| 10 | `30_antwar2` | AntWar2 | 22 | 1516192 | 299528 | 0.1968 |

## Boundary and interpretation

The measurement reads the exact Git blobs at the pinned commit for the path/SHA-256 manifest identified in the JSON artifact. Working-tree changes cannot affect it. Duplicate judge-dev copies, sample agents, backups, generated/build artifacts, vendored libraries, tests, and DeepClue story data are excluded.

This is implementation-description complexity. It is not game-tree size, strategic depth, learning difficulty, or information gain.

The separate [AB-Rule/1 report](agentbench-rule-complexity-v1.md) measures formal game-rule description length primarily in canonical AST nodes, while retaining rule atoms (RA) as a secondary metric. Bits, canonical AST nodes, and RA have different units and must not be compared numerically or combined into one score.

## Method references

- Cameron Browne and Frederic Maire, [Evolutionary Game Design](https://cambolbro.com/cv/publications/ciaig-browne-maire-19.pdf).
- Tom Schaul, Julian Togelius, and Jürgen Schmidhuber, [Measuring Intelligence through Games](https://arxiv.org/abs/1109.1314).
