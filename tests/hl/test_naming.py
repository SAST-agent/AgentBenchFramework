"""Tests for hl/naming.py — round/act name format."""
from __future__ import annotations

from agentbench_frame.hl.naming import (
    round_name, act_name, parse_round_number, ROUND_NAME_RE,
)


def test_round_name_format():
    assert round_name("26-07-29", 3) == "hl-v26-07-29-round3"
    assert ":" not in round_name("26-07-29", 3)


def test_act_name_5_digit_zero_padded():
    rn = round_name("26-07-29", 3)
    assert act_name(rn, 2) == "hl-v26-07-29-round3-000002"
    assert act_name(rn, 13) == "hl-v26-07-29-round3-000013"
    assert act_name(rn, 1) == "hl-v26-07-29-round3-000001"


def test_act_name_no_colons():
    rn = round_name("26-07-29", 3)
    assert ":" not in act_name(rn, 5)


def test_parse_round_number():
    assert parse_round_number("hl-v26-07-29-round3") == 3
    assert parse_round_number("hl-v26-07-29-round42") == 42
    assert parse_round_number("not-a-round") is None
    assert parse_round_number("hl-v26-07-29-round3-000001") is None  # act, not round


def test_round_name_re_distinguishes_round_from_round30():
    # round3 must NOT match round30 (next char after 'round3' is end, not a digit)
    assert ROUND_NAME_RE.match("hl-v26-07-29-round3")
    assert ROUND_NAME_RE.match("hl-v26-07-29-round30")
    assert parse_round_number("hl-v26-07-29-round3") == 3
    assert parse_round_number("hl-v26-07-29-round30") == 30
