"""Leak-resistant prompt construction for the single Codex act."""

from __future__ import annotations

from collections.abc import Sequence

from .replay import LearningReplay


def build_codex_prompt(
    benchmark_id: str,
    strategy_doc: str,
    rules_text: str,
    replay_guide: str,
    learning_replays: Sequence[LearningReplay],
) -> str:
    ordered = sorted(
        learning_replays,
        key=lambda item: (item.seed, item.evaluated_seat, item.opponent_tier),
    )
    evidence = "\n".join(item.to_json() for item in ordered)
    return f"""You are improving the deterministic Generals rule baseline for {benchmark_id}.

Inspect the editable workspace and use only the official-rule summary and the
redacted learning episodes below. You may change only strategy.py,
STRATEGY.md, and files under tests/. Preserve
choose_actions(round_number, my_seat, view) -> list[list[int]] and the SDK
entrypoint. Do not use randomness, time, network access, opponent-specific
hard-coding, evaluation seeds, or external files.

Run the strategy tests. Make one coherent improvement justified by the
learning evidence, then finish with a concise change summary.

CURRENT STRATEGY
{strategy_doc}

OFFICIAL RULE SUMMARY
{rules_text}

REPLAY FIELD GUIDE
{replay_guide}

REDACTED LEARNING EPISODES
{evidence}
"""
