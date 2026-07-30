# AgentBench Formal Rule Complexity Design

## Objective

Define a human-readable formal pseudocode language for abstract game rules,
translate every public AgentBench game except DeepClue into that language, and
publish reproducible rule-complexity measurements.

The primary quantity is not compiled size, runtime implementation size, or
exact Kolmogorov complexity. It is the canonical abstract-syntax-tree size of
a verified formal rule description under one frozen vocabulary:

\[
\widehat C_{\mathrm{AST},\mathcal L}(G)
=N_{\mathrm{structure}}(d_G)+N_{\mathrm{expression}}(d_G)
\]

where \(d_G\) is the committed AB-Rule/1 description of game \(G\).
\(N_{\mathrm{structure}}\) counts AB-Rule declaration, block, and statement
nodes except the identifying `game` header. \(N_{\mathrm{expression}}\)
counts canonical expression nodes, excluding the parser wrapper, expression
contexts such as `Load`, and CPython operator-marker objects such as `Add`,
`Eq`, and `And`. The secondary `rule_atoms` quantity remains the cardinality
of the disjoint semantic-proposition partition. Both are language-relative
description measures.

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
syntax; CPython execution semantics are not part of AB-Rule/1. The canonical
expression AST counts calls, names, attributes, literals, collections,
subscripts, keywords, comparisons, and Boolean/arithmetic expression nodes.
Identifier spelling, punctuation, and CPython-only wrapper/context/operator
marker objects are not counted.

### Atomic-proposition discipline

Each declaration member or rule statement expresses exactly one independently
changeable fact:

```text
rule fireball_hit(fireball, target):
  require hostile(fireball.owner, target.owner)
  update target.health = target.health - fireball.damage
  delete fireball
```

The example has one applicability proposition and two transition
propositions. Combining the two effects into one prose sentence or one helper
expression does not reduce the count. Conjunctions that impose independently
changeable requirements are written as separate `require` statements.
Game-specific helper operations must be defined as rules, and invoking such a
rule in a particular context remains one proposition.

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

The parser constructs a canonical AB-Rule AST. Its size is the primary
scientific quantity. This is not the raw node count returned by CPython
`ast.walk`: implementation-only wrapper, context, and operator-marker nodes
are excluded by the frozen AB-Rule/1 reference counter.

### Primary metric

The primary metric is:

\[
\texttt{ast\_nodes}
=\texttt{structural\_ast\_nodes}
+\texttt{expression\_ast\_nodes}.
\]

`structural_ast_nodes` counts every parsed AB-Rule declaration, block, and
statement except `game`. `expression_ast_nodes` counts the canonical
expression nodes attached to those structural nodes. Reusing a named advanced
operation does not duplicate its definition tree, but each static invocation
contributes its own call-site expression subtree. Runtime execution frequency
does not affect either count.

### Secondary metric

Each counted occurrence has a stable tag containing its semantic section and
source position, so repeated propositions in different rule contexts remain
distinct set members. The proposition set is partitioned into:

- `state`: player, constant, enum, entity, and field propositions;
- `action`: available action and action-parameter propositions;
- `observation`: visible, revealed, and hidden-information propositions;
- `setup`: initial-state and initial-randomization propositions;
- `condition`: applicability, branch, iteration, and choice propositions;
- `transition`: state-change, creation, deletion, event, and rule-application
  propositions;
- `outcome`: termination, ranking, winner, and score propositions.

The secondary metric is

\[
\texttt{rule\_atoms}
=|\mathcal A_{\mathrm{state}}\uplus
\mathcal A_{\mathrm{action}}\uplus
\mathcal A_{\mathrm{observation}}\uplus
\mathcal A_{\mathrm{setup}}\uplus
\mathcal A_{\mathrm{condition}}\uplus
\mathcal A_{\mathrm{transition}}\uplus
\mathcal A_{\mathrm{outcome}}|.
\]

The sum has a set-theoretic meaning: it is the cardinality of a disjoint union,
not a weighted score. `setup`, `rule`, and `terminal` block names group
propositions, while the `game` header identifies the document; those headers
do not add rule atoms. Entity, action, and observation declarations do add
atoms because the existence of each concept is itself a rule proposition.
Names, comments, indentation width, punctuation, line wrapping, expression
operators, and AST shape do not change `rule_atoms`.

### Explanatory output

The report publishes `structural_ast_nodes`, `expression_ast_nodes`, total
`ast_nodes`, `rule_atoms`, and the seven `atom_breakdown` cardinalities. It
also publishes source bytes, SHA-256, and provenance for reproduction.
Nesting depth, lexical token count, source line count, and raw CPython AST
node count are not game-complexity outputs.

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
7. Equivalent formatting and identifier renaming do not affect either metric.
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

`agentbench_frame.research.rule_complexity` counts the canonical AST,
constructs the proposition partition, validates the nine-game corpus, hashes
descriptions and provenance, and renders JSON and Markdown reports.

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

- comments, whitespace, and alpha-renaming do not change either metric;
- adding a canonical expression node increases `ast_nodes` without
  necessarily changing `rule_atoms`;
- `ast_nodes` equals `structural_ast_nodes + expression_ast_nodes`;
- adding an independently changeable declaration, condition, transition, or
  outcome proposition increases `rule_atoms`;
- `rule_atoms` equals the sum of the disjoint `atom_breakdown` partitions;
- malformed indentation and unsupported syntax fail closed;
- all required semantic sections are present;
- exactly the nine in-scope games are packaged and measured;
- DeepClue is absent;
- report ordering and hashes are deterministic;
- committed JSON and Markdown exactly match regenerated output;
- an installed wheel can load and measure the packaged rule corpus;
- the existing complete test suite remains green.

## Published Interpretation

The report uses the name **Ludemic Rule Description Complexity**. Its primary
unit is canonical **AST nodes**; **rule atoms (RA)** remain a secondary
semantic breakdown. Neither measures state-space size, strategic depth,
learning difficulty, implementation size, or information gain.
