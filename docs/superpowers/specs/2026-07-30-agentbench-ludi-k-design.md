# AgentBench Ludi K-Complexity Design

## Objective

Provide a reproducible, source-grounded Kolmogorov-complexity upper bound for
every public game-logic corpus in `Aoraku/AgentBench`, and ship the calculator,
frozen input-selection policy, and results in AgentBenchFramework.

The result is not the uncomputable, machine-independent value \(K(G)\). It is a
conditional description-length upper bound under one fixed reference machine:

\[
K(G \mid U_{\mathrm{AB-Ludi/1}}, R) \leq
8\,\lvert\operatorname{zlib9}(\operatorname{encode}_{\mathrm{AB-Ludi/1}}(G))\rvert
+ \mathcal{O}(1)
\]

Here \(R\) is the shared language runtime/toolchain and the additive decoder
constant is common to every game, so it is omitted from the reported number.

## Why AB-Ludi/1 Is Necessary

Original Ludi is a high-level language for finite, discrete, deterministic,
perfect-information, primarily two-player combinatorial board games. The
AgentBench corpus also contains real-time multiplayer play, hidden information,
four-player survival, maze pursuit, and LLM-mediated deduction. Claiming that
these games have faithful native-Ludi descriptions would be false.

AB-Ludi/1 therefore keeps Ludi's central representation—a game as a typed,
recursive ludeme tree—and adds a universal `source-module` leaf. A source
module contains a canonical path and the exact published source bytes. This
fallback is lossless: a decoder recovers the selected public game logic exactly.
It measures implementation-description complexity of the published logic, not
an abstract minimal rulebook.

## Reference Encoding

Each description is a binary, length-prefixed tree:

```text
game-logic(game_id,
  source-module(path_1, bytes_1),
  ...
  source-module(path_n, bytes_n))
```

The byte stream is:

1. ASCII magic and version `AB-LUDI/1`;
2. an unsigned 64-bit big-endian length followed by the UTF-8 game identifier;
3. an unsigned 64-bit big-endian module count;
4. modules sorted by canonical POSIX path;
5. for each module, a length-prefixed UTF-8 path and length-prefixed raw blob.

Paths are relative to the selected source root. Length prefixes make the
encoding injective even for arbitrary binary contents. The complexity program
is the complete zlib stream produced with level 9, `Z_DEFLATED`,
`MAX_WBITS`, memory level 9, and `Z_DEFAULT_STRATEGY`. The reference-machine
identity includes the zlib runtime version and a SHA-256 fingerprint of the
compressed output for a fixed test vector. The report also records the zlib
compile/runtime versions and the full behavior fingerprint.

## Input Boundary

The source repository is fixed to AgentBench commit
`b581bca3ba3d2d7d58a2f8c6bbddd060fc7fdc87`. The calculator requires `HEAD`
to equal that commit and reads content from its immutable Git blobs, never from
working-tree files. Dirty edits, untracked files, symlinks, and checkout filters
therefore cannot alter a measurement while retaining the same provenance.
Exactly ten games are expected:

- `23_doto`
- `24_miracle`
- `25_aquawar`
- `25_lostspace`
- `26_snakego`
- `27_antwar`
- `28_generals`
- `29_rollman`
- `30_antwar2`
- `30_deepclue`

Each game has an explicit authoritative root. For games with both
`gamecode_logic` and `judge_dev_logic`, only `gamecode_logic` is used to avoid
counting mirrors twice. DOTO uses the published Arena server because the other
public subtree is an agent implementation with many historical backups.

Only executable rule source and machine-readable rule configuration are
included: `.py`, `.c`, `.cpp`, `.h`, `.hpp`, `.json`, `.g4`, and `.map`.
The frozen exclusions remove:

- bundled generic JSON libraries;
- compiled binaries and build outputs;
- sample/player AI code;
- duplicate judge-dev copies;
- backups, documentation, and archives;
- tests, test fixtures, debug/legacy entry points, and generated ANTLR Python
  files when their source `.g4` grammar is present;
- DeepClue scenario content under `data/`, because this metric targets the
  shared game engine rather than individual stories.

The versioned source manifest fixes all 149 included relative paths and their
SHA-256 digests. Selection rules are applied to the pinned Git tree and compared
against that exact manifest; a missing, newly eligible, non-regular, or
hash-mismatched blob fails closed. The manifest itself has a SHA-256 identity in
the result artifact, so a future corpus revision cannot silently reuse the same
input identity.

## Components

### Catalog

`agentbench_frame.research.agentbench_catalog` owns the immutable ten-game
catalog, exact 149-file path/hash manifest, authoritative roots, suffix
allowlist, and exclusions. It resolves the pinned commit through Git plumbing
and reads blobs by object ID. Discovery fails closed when the corpus directory,
an authoritative blob, or an expected game is missing. Unexpected eligible
files or additional corpus games are errors rather than silently changing the
measurement boundary.

### Encoder and Calculator

`agentbench_frame.research.ludi_k` owns lossless framing, deterministic
compression, source hashing, per-game measurement, repository provenance, and
JSON/Markdown rendering. It exposes pure functions for unit tests and a
repository-level calculation API.

### CLI

The framework CLI adds:

```bash
agentbench complexity ludi \
  --agentbench-repo /path/to/Aoraku/AgentBench-at-b581bca \
  --json-output docs/research/agentbench-ludi-k-v1.json \
  --markdown-output docs/research/agentbench-ludi-k-v1.md
```

The command fails nonzero on source-commit or boundary drift, unreadable inputs,
or JSON/Markdown paths that identify the same file. It prints a compact ranking
after writing both artifacts.

### Frozen Results

The PR commits JSON and Markdown results calculated from the current public
AgentBench commit. JSON is the machine-readable source of truth. Markdown
explains the scientific interpretation and displays the ranked measurements.

## Testing

Tests must demonstrate:

- binary framing is injective and round-trips arbitrary bytes;
- file order does not change the encoding or measurement;
- one-byte source changes alter the source digest and generally the bound;
- committed Git blobs, rather than dirty or untracked working-tree files, are
  measured;
- the exact path/hash manifest rejects missing, extra, changed, and symlink
  inputs while preserving genuine rule source/configuration;
- the compressor behavior fingerprint is part of the reference-machine
  identity;
- missing or extra games fail closed;
- the public snapshot contains exactly ten unique game results and every result
  has nonzero file count, hashes, source bits, canonical bits, and upper-bound
  bits;
- CLI output is parseable and deterministic, and colliding output paths are
  rejected before either report is written.

The complete existing suite must still pass. Network access is not needed by
the calculator after the source repository is present.

## Interpretation

The primary comparable value is `k_upper_bits`. Smaller means the selected
published logic has a shorter program under AB-Ludi/1. It must not be described
as:

- exact Kolmogorov complexity;
- semantic strategy depth;
- state-space or game-tree complexity;
- learning difficulty;
- information gain.

Those are separate quantities. The report also includes raw source and
canonical sizes to expose how much of the result comes from compression and
framing.
