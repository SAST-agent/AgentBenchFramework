# AgentBench Formal Rule Complexity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define AB-Rule/1, translate the nine non-DeepClue AgentBench games, and publish deterministic Ludemic Rule Description Complexity measurements.

**Architecture:** A small indentation parser converts formal pseudocode into a semantic rule tree without compiling it. A separate metric module counts rule atoms and explanatory dimensions, validates a packaged nine-game corpus, and renders frozen JSON/Markdown artifacts through the existing CLI.

**Tech Stack:** Python 3.10+, standard-library `ast`, `tokenize`, `dataclasses`, `importlib.resources`, pytest, Hatch packaging.

## Global Constraints

- DeepClue is excluded.
- The source provenance commit is `b581bca3ba3d2d7d58a2f8c6bbddd060fc7fdc87`.
- `rule_atoms` is the only primary ranking metric.
- No compilation or bytecode length participates in any AB-Rule/1 metric.
- Existing AB-Ludi/1 implementation-complexity output remains unchanged.
- Execution is inline in the existing `codex/ludi-k-complexity` worktree; no subagents.

---

### Task 1: Frozen AB-Rule/1 Parser

**Files:**
- Create: `src/agentbench_frame/research/rule_language.py`
- Create: `tests/research/test_rule_language.py`

**Interfaces:**
- Produces: `parse_rule_description(source: str, *, path: str = "<memory>") -> RuleDocument`
- Produces: immutable `RuleDocument`, `RuleNode`, and `ExpressionStats`
- Produces: `RuleSyntaxError(ValueError)` with path and line diagnostics

- [ ] **Step 1: Write parser tests for declarations, nested statements, expressions, and rejection**

```python
def test_parse_formal_rule_document():
    document = parse_rule_description(
        """game Demo
players 2
entity Token:
  field owner: Player
rule move(token, target):
  require adjacent(token.position, target)
  when occupied(target):
    delete token
  otherwise:
    update token.position = target
terminal captured:
  return count(Token) == 0
"""
    )
    assert document.game_id == "Demo"
    assert {node.kind for node in document.walk()} >= {
        "players", "entity", "field", "rule", "require", "when",
        "delete", "otherwise", "update", "terminal", "return",
    }


@pytest.mark.parametrize(
    "source",
    ["game Demo\n  players 2\n", "game Demo\nunknown thing\n"],
)
def test_reject_invalid_rule_syntax(source):
    with pytest.raises(RuleSyntaxError):
        parse_rule_description(source)
```

- [ ] **Step 2: Run focused tests and confirm they fail**

Run: `pytest tests/research/test_rule_language.py -q`

Expected: collection fails because `rule_language` does not exist.

- [ ] **Step 3: Implement the frozen line/indentation parser and restricted expression analysis**

The implementation must:

- recognize exactly the declarations and statements in the design;
- require two-space indentation and reject skipped indentation levels;
- use `ast.parse(..., mode="eval")` only to analyze restricted expressions;
- reject comprehensions, lambdas, assignment expressions, imports, and arbitrary statements;
- count expression calls, Boolean/binary/unary/comparison operators, literals,
  and expression nesting;
- preserve source line numbers and comments beginning with `# provenance:`,
  `# source-root:`, `# reviewed:`, `# includes:`, and `# excludes:`.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/research/test_rule_language.py -q`

Expected: all parser tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/research/rule_language.py tests/research/test_rule_language.py
git commit -m "feat: define AB-Rule formal pseudocode"
```

### Task 2: Rule-Atom Metric and Corpus Validation

**Files:**
- Create: `src/agentbench_frame/research/rule_complexity.py`
- Create: `tests/research/test_rule_complexity.py`

**Interfaces:**
- Consumes: `parse_rule_description`
- Produces: `measure_rule_document(document: RuleDocument, source: str) -> dict[str, int | str]`
- Produces: `measure_rule_corpus() -> dict[str, object]`
- Produces: `render_rule_markdown(report: dict[str, object]) -> str`
- Produces: `write_rule_reports(json_path: Path, markdown_path: Path) -> dict[str, object]`

- [ ] **Step 1: Write failing metric invariance and sensitivity tests**

```python
def test_rule_atoms_ignore_comments_and_identifier_names():
    left = "game A\nplayers 2\nrule r(x):\n  return adjacent(x, 1)\n"
    right = (
        "game A\n# comment\nplayers 2\nrule renamed(long_name):\n"
        "  return adjacent(long_name, 1)\n"
    )
    assert metric(left)["rule_atoms"] == metric(right)["rule_atoms"]


def test_new_relation_increases_rule_atoms():
    base = "game A\nplayers 2\nrule r(x):\n  return occupied(x)\n"
    richer = "game A\nplayers 2\nrule r(x):\n  return occupied(x) and hostile(x)\n"
    assert metric(richer)["rule_atoms"] > metric(base)["rule_atoms"]
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `pytest tests/research/test_rule_complexity.py -q`

Expected: collection fails because `rule_complexity` does not exist.

- [ ] **Step 3: Implement the metric**

Use these exact rules:

- each semantic `RuleNode` contributes one rule atom;
- every restricted expression call or operator contributes one rule atom;
- scalar literal occurrences and literal collection entries contribute to
  `parameter_count`;
- `composition_depth` is the maximum node nesting plus expression nesting;
- `branch_count` counts `when`, `otherwise`, and `choose`;
- `rule_lines` ignores blank and comment-only lines;
- `canonical_tokens` uses Python lexical token categories for expression
  fragments and one token per normalized AB-Rule keyword, identifier, type,
  and literal.

- [ ] **Step 4: Add fail-closed corpus checks**

Require exactly the nine expected game IDs, mandatory provenance keys, and at
least one declaration in each category: player model, state/entity, action,
setup, transition rule, and terminal outcome.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/research/test_rule_complexity.py -q`

Expected: all metric tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/agentbench_frame/research/rule_complexity.py tests/research/test_rule_complexity.py
git commit -m "feat: measure ludemic rule descriptions"
```

### Task 3: Translate the Nine Games

**Files:**
- Create: `src/agentbench_frame/research/rules/ab_rule_v1/23_doto.abrule`
- Create: `src/agentbench_frame/research/rules/ab_rule_v1/24_miracle.abrule`
- Create: `src/agentbench_frame/research/rules/ab_rule_v1/25_aquawar.abrule`
- Create: `src/agentbench_frame/research/rules/ab_rule_v1/25_lostspace.abrule`
- Create: `src/agentbench_frame/research/rules/ab_rule_v1/26_snakego.abrule`
- Create: `src/agentbench_frame/research/rules/ab_rule_v1/27_antwar.abrule`
- Create: `src/agentbench_frame/research/rules/ab_rule_v1/28_generals.abrule`
- Create: `src/agentbench_frame/research/rules/ab_rule_v1/29_rollman.abrule`
- Create: `src/agentbench_frame/research/rules/ab_rule_v1/30_antwar2.abrule`
- Create: `tests/research/test_rule_corpus.py`

**Interfaces:**
- Consumes: AB-Rule/1 parser and corpus validator
- Produces: a packaged, complete nine-game rule corpus

- [ ] **Step 1: Write corpus completeness tests**

```python
def test_rule_corpus_has_exact_expected_games():
    report = measure_rule_corpus()
    assert [game["game_id"] for game in report["games_by_id"]] == [
        "23_doto", "24_miracle", "25_aquawar", "25_lostspace",
        "26_snakego", "27_antwar", "28_generals", "29_rollman",
        "30_antwar2",
    ]
    assert all(game["rule_atoms"] > 0 for game in report["games_by_id"])


def test_deepclue_is_not_packaged():
    assert "30_deepclue" not in {
        game["game_id"] for game in measure_rule_corpus()["games_by_id"]
    }
```

- [ ] **Step 2: Run the corpus tests and confirm failure**

Run: `pytest tests/research/test_rule_corpus.py -q`

Expected: validation fails because descriptions are missing.

- [ ] **Step 3: Translate DOTO, Miracle, and AquaWar**

Cover real-time tick/joint actions and projectiles for DOTO; hex movement,
summoning, event ordering, health/mana/cooldowns/buffs for Miracle; and hidden
selection, assertions, three rounds, random damage, fish actives/passives, and
outcomes for AquaWar.

- [ ] **Step 4: Translate LostSpace, SnakeGo, and AntWar**

Cover LostSpace's four-player observation, generated multi-floor map,
inventory, traps, delayed events, damage/death, shrinking area, and escape;
SnakeGo's alternating multi-snake actions, ordered bodies, growth, items,
split/fire/solidification, death, territory, and tie-breaking; and AntWar's
pheromone initialization/evolution, autonomous ant movement, towers,
resources, upgrades, weapons, automatic combat, and headquarters outcomes.

- [ ] **Step 5: Translate Generals, Rollman, and AntWar2**

Cover Generals' random map, ownership/army transfer, generals, production,
coins, technologies, skills, super-weapons, cooldowns, and victory; Rollman's
random levels, simultaneous Pacman/ghost movement, path collision, beans,
skills, scores, respawn, portals, level completion, and timeout; and AntWar2's
full AntWar base plus ant kinds, behavior modes, evasion, bewitching, siege,
random effects, and expanded weapons.

- [ ] **Step 6: Run corpus tests**

Run: `pytest tests/research/test_rule_corpus.py -q`

Expected: all corpus descriptions parse, satisfy section coverage, and exclude
DeepClue.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/research/rules tests/research/test_rule_corpus.py
git commit -m "feat: formalize AgentBench game rules"
```

### Task 4: CLI, Packaging, and Frozen Reports

**Files:**
- Modify: `src/agentbench_frame/cli.py`
- Modify: `pyproject.toml`
- Create: `tests/research/test_rule_complexity_cli.py`
- Create: `tests/research/test_rule_complexity_wheel.py`
- Create: `docs/research/agentbench-rule-complexity-v1.json`
- Create: `docs/research/agentbench-rule-complexity-v1.md`

**Interfaces:**
- Consumes: `write_rule_reports`
- Produces: `agentbench complexity rules`
- Produces: deterministic JSON and Markdown artifacts

- [ ] **Step 1: Write failing CLI tests**

```python
def test_rule_complexity_cli_writes_reports(tmp_path):
    json_path = tmp_path / "rules.json"
    markdown_path = tmp_path / "rules.md"
    result = cli_runner.invoke(
        app,
        [
            "complexity", "rules",
            "--json-output", str(json_path),
            "--markdown-output", str(markdown_path),
        ],
    )
    assert result.exit_code == 0
    assert len(json.loads(json_path.read_text())["games"]) == 9
    assert "Ludemic Rule Description Complexity" in markdown_path.read_text()
```

- [ ] **Step 2: Run CLI tests and confirm failure**

Run: `pytest tests/research/test_rule_complexity_cli.py -q`

Expected: CLI rejects the unknown `rules` subcommand.

- [ ] **Step 3: Add the CLI and package resources**

Reuse existing argparse conventions. Reject identical output paths before
writing either file. Ensure Hatch wheels and sdists include every `.abrule`
resource.

- [ ] **Step 4: Add installed-wheel smoke coverage**

Build a wheel, install it into a temporary target, import
`agentbench_frame.research.rule_complexity`, and assert that
`measure_rule_corpus()` returns nine games without a source checkout.

- [ ] **Step 5: Generate committed reports**

Run:

```bash
agentbench complexity rules \
  --json-output docs/research/agentbench-rule-complexity-v1.json \
  --markdown-output docs/research/agentbench-rule-complexity-v1.md
```

Expected: both reports contain exactly nine results sorted primarily by
`rule_atoms`.

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest tests/research/test_rule_complexity_cli.py \
  tests/research/test_rule_complexity_wheel.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/agentbench_frame/cli.py \
  tests/research/test_rule_complexity_cli.py \
  tests/research/test_rule_complexity_wheel.py \
  docs/research/agentbench-rule-complexity-v1.json \
  docs/research/agentbench-rule-complexity-v1.md
git commit -m "feat: publish AgentBench rule complexity"
```

### Task 5: Full Verification and Documentation Alignment

**Files:**
- Modify: `README.md`
- Modify: `docs/research/agentbench-ludi-k-v1.md`
- Modify: `tests/research/test_ludi_k_snapshot.py`

**Interfaces:**
- Consumes: committed AB-Rule/1 reports and existing AB-Ludi/1 reports
- Produces: explicit separation of `K_impl` and rule-atom complexity

- [ ] **Step 1: Add documentation assertions**

Assert that the existing implementation report calls itself implementation
complexity and links to the new rule-complexity report without mixing units.

- [ ] **Step 2: Update documentation**

Explain:

- AB-Ludi/1 `k_upper_bits` is implementation-description complexity;
- AB-Rule/1 `rule_atoms` is formal conceptual rule-description complexity;
- neither metric is strategic depth, state-space size, learning difficulty, or
  information gain.

- [ ] **Step 3: Regenerate both reports and check for drift**

Run the existing Ludi command against `/tmp/AgentBench-source`, then run the
new rules command. `git diff --exit-code` must show no report drift.

- [ ] **Step 4: Run the complete suite**

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Run repository hygiene checks**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and only intended changes before the final
commit.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/research/agentbench-ludi-k-v1.md \
  tests/research/test_ludi_k_snapshot.py
git commit -m "docs: distinguish rule and implementation complexity"
```
