from dataclasses import replace
import json
from pathlib import Path

import pytest

from agentbench_frame.eval.benchmark import GameResult
from agentbench_frame.generals.assets import load_pilot_config
from agentbench_frame.generals.challenge_v9 import load_round9_challenge_config
from agentbench_frame.generals.evaluator import GeneralsEvaluation
from agentbench_frame.generals.models import AssetLayout, ReplaySkillAsset
from agentbench_frame.tracking.provider import ProviderInvocation, ProviderUsage
from tests.generals.test_lineage_v9 import _authorities
from tests.generals.test_pipeline_v8 import _match
from tests.generals.test_prompt_v9 import _attribution_run


FIXTURES = Path(__file__).parent / "fixtures"
ASSET_ROOT = Path("/home/cathy/AgentBench/AgentBench/.worktrees/generals-assets")
SKILL_PATH = ASSET_ROOT / "backend_sources/corpus/28_generals/skills/replay-analysis-v2/SKILL.md"
RULES_PATH = ASSET_ROOT / "backend_sources/corpus/28_generals/benchmark/rules.md"
ENGINE_SHA256 = "4ca898365a6fe9a0cca39c3c8be0e49f449334814b106882c0b78b3aabdb1d97"
SKILL_SHA256 = "0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48"
CONFIG = load_pilot_config(FIXTURES / "pilot-v1.toml")
BASE_CHALLENGE = load_round9_challenge_config(
    FIXTURES / "v9-scientific-attribution-v1.toml",
    CONFIG,
    engine_hash=ENGINE_SHA256,
    replay_skill_sha256=SKILL_SHA256,
)


class V9Provider:
    provider_name = "codex"

    def __init__(self, mode="valid"):
        self.mode = mode
        self.calls = 0
        self.prompt = None

    def invoke(self, context):
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("v9 provider called more than once")
        self.prompt = context["prompt"]
        workspace = Path(context["workspace_root"])
        (workspace / "STRATEGY.md").write_text(
            "Deterministic planner policy with lexicographic tie order, macro and "
            "primitive command limits, main-general phase safety, and verified-prefix fallback.\n"
        )
        (workspace / "EXPERIENCE.md").write_text(
            "Retained safe fallback; rejected replay memorization; new contact lesson; remaining risk.\n"
        )
        (workspace / "policy").mkdir(exist_ok=True)
        (workspace / "policy/planner.py").write_text("MAX_MACRO = 8\n")
        if self.mode == "protected":
            (workspace / "main.py").write_text("BROKEN = True\n")
        elif self.mode == "random":
            (workspace / "strategy.py").write_text(
                "import random\ndef choose_actions(*args): return [[8]]\n"
            )
        elif self.mode == "seed_branch":
            (workspace / "strategy.py").write_text(
                "def choose_actions(round_number, my_seat, view):\n"
                "    seed = view.get('seed', 0)\n    return [[8]] if seed else [[8]]\n"
            )
        elif self.mode == "long_macro":
            commands = ", ".join("[1, 0, 0, 4, 1]" for _ in range(9))
            (workspace / "strategy.py").write_text(
                "def choose_actions(*args):\n"
                f"    return [{commands}, [8]]\n"
            )
        elif self.mode == "illegal":
            (workspace / "strategy.py").write_text(
                "def choose_actions(*args): return [[99], [8]]\n"
            )
        elif self.mode == "missing_docs":
            (workspace / "EXPERIENCE.md").unlink()
        elif self.mode == "latency":
            (workspace / "strategy.py").write_text(
                "import time\ndef choose_actions(*args):\n    time.sleep(2)\n    return [[8]]\n"
            )
        raw = Path(context["raw_output_path"])
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text('{"type":"turn.completed"}\n')
        return ProviderInvocation(
            status="completed",
            usage=ProviderUsage(100, 20, 120, "exact"),
            tool_call_count=2,
            elapsed_time_s=0.1,
            raw_output_ref=str(raw),
        )


class V9Evaluator:
    def __init__(self, *, validation=(1, 1), formal_mode="champion", raise_phase=None):
        self.validation = validation
        self.formal_mode = formal_mode
        self.raise_phase = raise_phase
        self.calls = []

    def _validation_outcomes(self, cases):
        remaining = list(self.validation)
        outcomes = []
        for case in cases:
            seat = case.first_player
            if remaining[seat]:
                remaining[seat] -= 1
                outcomes.append("win")
            else:
                outcomes.append("loss")
        return outcomes

    def _formal_outcomes(self, cases):
        if self.formal_mode == "champion":
            winners = set(range(13))
        elif self.formal_mode == "total12":
            winners = set(range(12))
        elif self.formal_mode == "high1":
            winners = {0, *range(6, 18)}
        elif self.formal_mode == "high_one_seat":
            winners = {0, 2, *range(6, 17)}
        else:
            winners = set()
        return ["win" if index in winners else "loss" for index in range(len(cases))]

    def evaluate(self, workspace, version, phase, run, cases=None):
        del workspace
        selected = tuple(cases or ())
        self.calls.append((version, phase, len(selected)))
        if phase == self.raise_phase:
            raise RuntimeError(f"synthetic {phase} crash")
        outcomes = (
            self._validation_outcomes(selected)
            if phase == "validation9"
            else self._formal_outcomes(selected)
        )
        invalid = self.formal_mode == "invalid" and phase == "formal9"
        matches = tuple(
            _match(case, version, outcome, valid=not (invalid and index == 0))
            for index, (case, outcome) in enumerate(zip(selected, outcomes))
        )
        results = tuple(
            GameResult(
                case_id=case.case_id,
                outcome=outcome,
                valid=not (invalid and index == 0),
                error="invalid" if invalid and index == 0 else None,
                metadata={
                    "tier": case.metadata["tier"],
                    "phase": phase,
                    "seed": case.seed,
                    "evaluated_seat": case.first_player,
                },
            )
            for index, (case, outcome) in enumerate(zip(selected, outcomes))
        )
        for match, game_result in zip(matches, results):
            run.log_budget(
                "evaluation",
                episodes=1,
                env_steps=len(match.turns),
                game_agent_decision_steps=1,
                primitive_commands=2,
                time_s=match.elapsed_time_s,
            )
            run.log_game_result(phase, version, {
                "case_id": match.case_id,
                "valid": match.valid,
                "outcome": game_result.outcome,
            })
        complete = all(result.valid for result in results)
        wins = sum(result.outcome == "win" for result in results if result.valid)
        per_tier = {}
        for tier in ("high", "medium", "low"):
            tier_results = [r for r in results if r.metadata["tier"] == tier]
            per_tier[tier] = (
                sum(r.outcome == "win" for r in tier_results) / len(tier_results)
                if tier_results and complete else None
            )
        return GeneralsEvaluation(
            version=version,
            status="complete" if complete else "incomplete",
            score=wins / len(results) if complete else None,
            wins=wins,
            losses=sum(r.outcome == "loss" for r in results if r.valid),
            draws=0,
            per_tier=per_tier,
            seat_gap=0.0 if complete else None,
            results=results,
            matches=matches,
        )


def _pipeline(tmp_path, provider=None, evaluator=None):
    from agentbench_frame.generals.pipeline_v9 import GeneralsHLRound9Pipeline

    v7, v8, _, v7_manifest, v8_manifest = _authorities(tmp_path / "history")
    challenge = replace(
        BASE_CHALLENGE,
        parent_run_id=v7.name,
        parent_content_hash=v7_manifest.content_hash,
        predecessor_run_id=v8.name,
        predecessor_content_hash=v8_manifest.content_hash,
    )
    policy_hashes = {
        "A": v7_manifest.content_hash,
        "B": "b" * 64,
        "C": "c" * 64,
        "D": v8_manifest.content_hash,
    }
    attribution = _attribution_run(tmp_path / "attribution", policy_hashes)
    skill_text = SKILL_PATH.read_text()
    skill = ReplaySkillAsset(SKILL_PATH, skill_text, SKILL_SHA256)
    layout = AssetLayout(
        root=tmp_path / "assets",
        engine_root=tmp_path / "assets",
        baseline_root=tmp_path / "assets",
        opponents=CONFIG.opponents,
        engine_hash=ENGINE_SHA256,
    )
    pipeline = GeneralsHLRound9Pipeline(
        config=CONFIG,
        challenge=challenge,
        assets=layout,
        replay_skill=skill,
        policy_parent_run_dir=v7,
        iteration_predecessor_run_dir=v8,
        attribution_run_dir=attribution,
        data_dir=tmp_path / "data",
        provider=provider or V9Provider(),
        evaluator=evaluator or V9Evaluator(),
        rules_path=RULES_PATH,
    )
    return pipeline


def _summary(run_dir):
    return json.loads((run_dir / "summary.json").read_text())


def test_v9_runs_one_act_then_all_validation_and_formal_cases(tmp_path):
    provider = V9Provider()
    evaluator = V9Evaluator()
    result = _pipeline(tmp_path, provider, evaluator).run()

    assert provider.calls == 1
    assert result.runnable is True
    assert result.validation_attempted is True
    assert result.formal_attempted is True
    assert result.champion_claim is True
    assert ("v9", "validation9", 12) in evaluator.calls
    assert ("v9", "formal9", 18) in evaluator.calls
    summary = _summary(result.run_dir)
    assert summary["policy_parent_version"] == "v7"
    assert summary["iteration_predecessor_version"] == "v8"
    assert summary["round_act_count"] == 1
    assert summary["act_count"] == 10
    assert len(summary["score_history"]) == 11
    assert summary["score_history"][2] is None
    assert not (result.run_dir / "workspace/versions/v8/source").exists()


def test_validation_failure_cannot_hide_formal(tmp_path):
    evaluator = V9Evaluator(validation=(0, 0))
    result = _pipeline(tmp_path, evaluator=evaluator).run()

    assert result.validation_passed is False
    assert result.formal_attempted is True
    assert ("v9", "formal9", 18) in evaluator.calls
    assert sum(
        json.loads(line).get("phase") == "formal9"
        and json.loads(line)["event_type"] == "game_result"
        for line in (result.run_dir / "events.jsonl").read_text().splitlines()
    ) == 18


@pytest.mark.parametrize(
    ("validation", "formal_mode"),
    [
        ((1, 0), "champion"),
        ((2, 0), "champion"),
        ((1, 1), "total12"),
        ((1, 1), "high1"),
        ((1, 1), "high_one_seat"),
        ((1, 1), "invalid"),
    ],
)
def test_v9_champion_gate_rejects_each_missing_requirement(
    tmp_path, validation, formal_mode
):
    result = _pipeline(
        tmp_path,
        evaluator=V9Evaluator(validation=validation, formal_mode=formal_mode),
    ).run()

    assert result.champion_claim is False
    assert (result.run_dir / "versions/v9/source/strategy.py").is_file()
    assert _summary(result.run_dir)["champion"]["version"] == "v7"


@pytest.mark.parametrize(
    "mode",
    ["protected", "random", "seed_branch", "long_macro", "illegal", "missing_docs", "latency"],
)
def test_v9_rejects_unsafe_or_invalid_candidates(tmp_path, mode):
    provider = V9Provider(mode)
    result = _pipeline(tmp_path, provider=provider).run()

    assert provider.calls == 1
    assert result.runnable is False
    assert result.formal_attempted is False
    assert result.champion_claim is False
    assert (result.run_dir / "versions/v9/manifest.json").is_file()
