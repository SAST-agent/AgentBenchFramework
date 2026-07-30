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
    assert left_metric["parameter_count"] == right_metric["parameter_count"]
    assert left_metric["canonical_tokens"] == right_metric["canonical_tokens"]


def test_new_relation_and_boolean_composition_increase_rule_atoms():
    base = "game A\nplayers 2\nrule r(x):\n  return occupied(x)\n"
    richer = (
        "game A\nplayers 2\nrule r(x):\n"
        "  return occupied(x) and hostile(x)\n"
    )

    assert metric(richer)["rule_atoms"] == metric(base)["rule_atoms"] + 2


def test_explanatory_metrics_describe_branching_depth_and_parameters():
    source = """\
game A
players 2
constant LIMITS = [3, 5]
rule resolve(x):
  when active(x):
    choose damage from [1, 2, 3]:
      update x.health = x.health - damage
  otherwise:
    pass
"""

    result = metric(source)

    assert result["branch_count"] == 3
    assert result["parameter_count"] == 6
    assert result["composition_depth"] >= 4
    assert result["rule_lines"] == 9
    assert result["canonical_tokens"] > result["rule_atoms"]


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
