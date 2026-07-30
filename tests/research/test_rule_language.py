import pytest

from agentbench_frame.research.rule_language import (
    RuleSyntaxError,
    parse_rule_description,
)


DEMO_RULES = """\
# provenance: b581bca3ba3d2d7d58a2f8c6bbddd060fc7fdc87
# source-root: corpus/demo
# reviewed: rules.py
# includes: abstract game rules
# excludes: rendering
game Demo
players 2
constant BOARD_SIZE = 3
enum Owner:
  value Neutral
  value First
  value Second
entity Token:
  field owner: Owner
  field position: Cell
action Move:
  field token: Token
  field target: Cell
setup initial:
  create Token(owner=First, position=Cell(0, 0))
rule move(token, target):
  require adjacent(token.position, target)
  when occupied(target):
    delete token
  otherwise:
    update token.position = target
terminal captured:
  return count(Token) == 0
"""


def test_parse_formal_rule_document():
    document = parse_rule_description(DEMO_RULES, path="demo.abrule")

    assert document.game_id == "Demo"
    assert document.path == "demo.abrule"
    assert document.metadata == {
        "provenance": "b581bca3ba3d2d7d58a2f8c6bbddd060fc7fdc87",
        "source-root": "corpus/demo",
        "reviewed": "rules.py",
        "includes": "abstract game rules",
        "excludes": "rendering",
    }
    assert {node.kind for node in document.walk()} >= {
        "players",
        "constant",
        "enum",
        "value",
        "entity",
        "field",
        "action",
        "setup",
        "create",
        "rule",
        "require",
        "when",
        "delete",
        "otherwise",
        "update",
        "terminal",
        "return",
    }
    require = next(node for node in document.walk() if node.kind == "require")
    assert require.expression.calls == 1
    assert require.expression.operators == 0
    terminal_return = [
        node for node in document.walk() if node.kind == "return"
    ][-1]
    assert terminal_return.expression.calls == 1
    assert terminal_return.expression.operators == 1


def test_expression_stats_cover_calls_operators_literals_and_depth():
    source = """\
game Expressions
players 2
rule score(state):
  return max(0, state.value + bonus(state)) if active(state) else -1
"""

    document = parse_rule_description(source)
    expression = next(
        node.expression
        for node in document.walk()
        if node.kind == "return"
    )

    assert expression.calls == 3
    assert expression.operators == 3
    assert expression.literals == 2
    assert expression.depth >= 4


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("game Demo\n  players 2\n", "unexpected indentation"),
        ("game Demo\nunknown thing\n", "unsupported AB-Rule/1 syntax"),
        (
            "game Demo\nplayers 2\nrule r(x):\n  return [y for y in x]\n",
            "unsupported expression",
        ),
        (
            "game Demo\nplayers 2\nrule r(x):\n    return x\n",
            "two spaces",
        ),
        (
            "game Demo\nplayers 2\nrule r(x):\n  when x:\n      return x\n",
            "two spaces",
        ),
    ],
)
def test_reject_invalid_rule_syntax(source, message):
    with pytest.raises(RuleSyntaxError, match=message):
        parse_rule_description(source, path="bad.abrule")


def test_requires_one_game_declaration():
    with pytest.raises(RuleSyntaxError, match="exactly one game"):
        parse_rule_description("players 2\n")

    with pytest.raises(RuleSyntaxError, match="exactly one game"):
        parse_rule_description("game One\ngame Two\nplayers 2\n")
