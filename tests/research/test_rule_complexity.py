import pytest

from agentbench_frame.research.rule_complexity import (
    RuleCorpusError,
    measure_rule_document,
    validate_rule_document,
)
from agentbench_frame.research.rule_language import parse_rule_description


def metric(source: str) -> dict:
    document = parse_rule_description(source)
    return measure_rule_document(document, source)


def test_rule_atoms_ignore_comments_formatting_and_identifier_length():
    left = """\
game A
players 2
rule r(x):
  return adjacent(x, 1)
"""
    right = """\
game A
# a prose comment does not describe another rule
players 2
rule renamed(long_identifier):
  return adjacent(long_identifier, 1)
"""

    left_metric = metric(left)
    right_metric = metric(right)

    assert left_metric["rule_atoms"] == right_metric["rule_atoms"]
    assert left_metric["atom_breakdown"] == right_metric["atom_breakdown"]


def test_expression_ast_shape_does_not_increase_rule_atoms():
    base = "game A\nplayers 2\nrule r(x):\n  return occupied(x)\n"
    richer = (
        "game A\nplayers 2\nrule r(x):\n"
        "  return occupied(x) and hostile(x)\n"
    )

    assert metric(richer)["rule_atoms"] == metric(base)["rule_atoms"]


def test_new_atomic_statement_increases_rule_atoms():
    base = "game A\nplayers 2\nrule r(x):\n  return occupied(x)\n"
    richer = (
        "game A\nplayers 2\nrule r(x):\n"
        "  require hostile(x)\n"
        "  return occupied(x)\n"
    )

    assert metric(richer)["rule_atoms"] == metric(base)["rule_atoms"] + 1


def test_rule_atoms_are_disjoint_semantic_proposition_partitions():
    source = """\
game A
players 2
constant LIMIT = 3
entity Token:
  field owner: Player
action Move:
  field target: Cell
observation View:
  field visible: Set[Cell]
setup initial:
  create Token(owner=first_player)
rule resolve(x):
  require active(x)
  when occupied(x):
    update x.health = x.health - 1
  reveal visible_to(x, x.owner)
terminal finished:
  when x.health <= 0:
    return winner(x.owner)
"""

    result = metric(source)

    assert result["atom_breakdown"] == {
        "state": 4,
        "action": 2,
        "observation": 3,
        "setup": 1,
        "condition": 2,
        "transition": 1,
        "outcome": 2,
    }
    assert result["rule_atoms"] == sum(result["atom_breakdown"].values())
    assert set(result) >= {
        "description_bytes",
        "description_sha256",
    }
    assert set(result).isdisjoint(
        {
            "composition_depth",
            "branch_count",
            "parameter_count",
            "canonical_tokens",
            "rule_lines",
        }
    )


def test_rule_document_validation_requires_provenance_and_semantic_sections():
    source = """\
game 23_doto
players 2
rule move(x):
  return x
"""
    document = parse_rule_description(source)

    with pytest.raises(RuleCorpusError, match="metadata"):
        validate_rule_document(document, expected_game_id="23_doto")


def test_rule_document_validation_rejects_game_id_mismatch():
    source = """\
# provenance: b581bca3ba3d2d7d58a2f8c6bbddd060fc7fdc87
# source-root: corpus/demo
# reviewed: rules.py
# includes: rules
# excludes: rendering
game wrong
players 2
entity State:
  field value: Integer
action Move:
  field value: Integer
observation Public:
  field value: Integer
setup initial:
  create State(value=0)
rule move(x):
  update x.value = 1
terminal end:
  return x.value == 1
"""
    document = parse_rule_description(source)

    with pytest.raises(RuleCorpusError, match="expected game ID"):
        validate_rule_document(document, expected_game_id="23_doto")
