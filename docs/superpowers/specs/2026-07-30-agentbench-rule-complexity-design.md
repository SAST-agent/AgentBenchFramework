# AgentBench Formal Rule Complexity Design

## Objective

Define a human-readable formal pseudocode language for abstract game rules,
translate every public AgentBench game except DeepClue into that language, and
publish reproducible rule-complexity measurements.

The primary quantity is not compiled size, runtime implementation size, or
exact Kolmogorov complexity. It is the length of a verified formal rule
description under one frozen vocabulary:

\[
\widehat C_{\mathcal L}(G)=\operatorname{Atoms}(d_G)
\]

where \(d_G\) is the committed AB-Rule/1 description of game \(G\), and
`Atoms` counts occurrences of semantic rule concepts. This is a
language-relative upper bound on the minimum conceptual rule description.

## Scope

The v1 corpus is the nine games at AgentBench commit
`b581bca3ba3d2d7d58a2f8c6bbddd060fc7fdc87`:

- `23_doto`
- `24_miracle`
- `25_aquawar`
- `25_lostspace`
- `26_snakego`
- `27_antwar`
- `28_generals`
- `29_rollman`
- `30_antwar2`

DeepClue is excluded. Communication protocols, rendering, replay
serialization, logging, timeouts imposed only by the judging platform, and
implementation optimizations are outside the rule boundary. Game clocks and
timeouts that alter game outcomes remain rules and are included.

## AB-Rule/1

AB-Rule/1 is an indentation-based formal pseudocode. It is parsed but never
compiled for measurement. Its vocabulary is deliberately small and
domain-neutral.

### Top-level declarations

```text
game SnakeGo
players 2

constant BOARD_SIZE = 16

entity Snake:
  field owner: Player
  field body: Sequence[Cell]
  field length_bank: Integer

action Move:
  field snake: Snake
  field direction: Direction

rule move(snake, direction):
  ...
```

Allowed declarations are:

- `game NAME`
- `players EXPR`
- `constant NAME = EXPR`
- `enum NAME:`
- `entity NAME:`
- `action NAME:`
- `observation NAME:`
- `setup NAME:`
- `rule NAME(PARAMETERS):`
- `terminal NAME:`

Allowed declaration members are enum values and `field NAME: TYPE` with an
optional default expression.

### Rule statements

```text
let destination = neighbor(head(snake.body), direction)
require destination in board.cells
when blocked(destination):
  eliminate snake
otherwise:
  update snake.body = prepend(destination, snake.body)
for unit in living_units:
  apply end_turn_effects(unit)
choose outcome from damage_distribution:
  emit Damage(outcome)
create Snake(owner=snake.owner, body=tail)
delete expired_item
emit TurnEnded(player)
return score
```

The statement vocabulary is:

- `let`
- `require`
- `when`
- `otherwise`
- `for`
- `choose`
- `update`
- `apply`
- `create`
- `delete`
- `emit`
- `reveal`
- `hide`
- `return`
- `pass`

Expressions use a restricted Python-like notation for literals, references,
attribute access, calls, indexing, collections, arithmetic, comparisons, and
Boolean composition. This notation is only a concise mathematical surface
syntax; CPython execution semantics are not part of AB-Rule/1.

### Abstraction boundary

Built-in expression operators and generic mathematical concepts such as
set membership, sequence length, graph neighborhood, distance, probability,
and simultaneous action are acceptable primitives.

Game-specific concepts are not built-ins. `split_snake`,
`update_pheromone`, `cast_fish_skill`, and `respawn_rollman`, for example,
must be defined by rules in the game description. A named rule is counted once
where defined and once at every invocation, allowing honest abstraction and
reuse without hiding its definition.

## Measurement

The parser constructs a semantic rule tree solely to make counting
deterministic. The public scientific concepts are rule atoms and composition,
not compiler AST nodes.

### Primary metric

`rule_atoms` counts:

- every declaration that introduces a rule-bearing concept: player model,
  constant, enum, entity, field, action, observation, setup, rule, terminal;
- every rule statement;
- every operator and function/relation invocation inside an expression.

Names, punctuation, comments, indentation width, and line wrapping do not
change `rule_atoms`. Plain references and attribute selections do not count as
new rule atoms.

### Explanatory metrics

- `composition_depth`: maximum nesting depth of declarations, conditions,
  loops, choices, and expression composition;
- `branch_count`: occurrences of `when`, `otherwise`, and explicit `choose`
  alternatives;
- `parameter_count`: scalar literal occurrences plus entries in literal
  collections and rule tables;
- `canonical_tokens`: lexical token count after comments and nonsemantic
  formatting are removed and identifiers are alpha-normalized;
- `rule_lines`: non-empty, non-comment source lines, for display only.

No arbitrary weighted sum of depth, branching, and parameters is used.
`rule_atoms` is the only primary ranking quantity; the other values explain
where the description length comes from.

## Fairness Rules

1. All games use exactly AB-Rule/1.
2. The vocabulary and counter are frozen before publishing the measurements.
3. A game-specific operation must have an explicit rule definition.
4. Fixed parameter tables are included even when the source stores them in
   JSON or another data file.
5. A map-generation algorithm is a rule. A sampled seed or generated map
   instance is not.
6. A fixed map essential to the game is reported separately as instance
   content and is not allowed to disappear from provenance.
7. Equivalent formatting and identifier renaming do not affect the primary
   metric.
8. Every description carries the exact source commit and a source-path
   coverage manifest.
9. The result is a best-known verified description, not a proof of global
   minimality.
10. Existing AB-Ludi/1 `k_upper_bits` remains available as implementation
    complexity and is never mixed with AB-Rule/1 rankings.

## Translation Evidence

Each game description starts with machine-readable comments declaring:

- game ID and title;
- AgentBench source commit;
- authoritative source root;
- reviewed source paths;
- explicit included and excluded boundaries.

Descriptions must cover:

- player and information model;
- state and equipment;
- initialization and randomness;
- legal action families;
- transition and automatic-event rules;
- observation/hidden-information rules;
- terminal and outcome rules.

The repository validator rejects a missing section, unknown statement,
malformed indentation, duplicate game ID, missing corpus game, unexpected
DeepClue description, or provenance mismatch.

## Components

### Language and parser

`agentbench_frame.research.rule_language` defines the frozen vocabulary,
line-oriented indentation parser, restricted expression parser, semantic tree,
canonical identifier mapping, and syntax diagnostics.

### Metric

`agentbench_frame.research.rule_complexity` calculates the five metrics,
validates the nine-game corpus, hashes descriptions and provenance, and renders
JSON and Markdown reports.

### Corpus

`src/agentbench_frame/research/rules/ab_rule_v1/*.abrule` stores one complete
formal description per game. Package configuration must include these files in
wheels and source distributions.

### CLI

```bash
agentbench complexity rules \
  --json-output docs/research/agentbench-rule-complexity-v1.json \
  --markdown-output docs/research/agentbench-rule-complexity-v1.md
```

The command measures packaged descriptions and needs no AgentBench checkout or
network after installation.

## Testing

Tests demonstrate:

- comments, whitespace, line wrapping, and alpha-renaming do not change
  `rule_atoms`;
- adding a rule concept, statement, condition, relation call, or operator
  increases `rule_atoms`;
- malformed indentation and unsupported syntax fail closed;
- all required semantic sections are present;
- exactly the nine in-scope games are packaged and measured;
- DeepClue is absent;
- report ordering and hashes are deterministic;
- committed JSON and Markdown exactly match regenerated output;
- an installed wheel can load and measure the packaged rule corpus;
- the existing complete test suite remains green.

## Published Interpretation

The report uses the name **Ludemic Rule Description Complexity** and the unit
**rule atoms (RA)**. It states explicitly that RA measures formal conceptual
description length under AB-Rule/1. It does not measure state-space size,
strategic depth, learning difficulty, implementation size, or information
gain.
