from dataclasses import replace
import json
from pathlib import Path

import pytest

from agentbench_frame.eval.benchmark import GameResult
from agentbench_frame.generals.assets import load_pilot_config
from agentbench_frame.generals.champion_campaign import (
    CAMPAIGN_ACTION_SPACE_SPEC_ID,
    CAMPAIGN_DECISION_SPACE_SHA256,
    CAMPAIGN_QUALIFICATION_MANIFEST_SHA256,
    CAMPAIGN_REPLAY_SKILL_SHA256,
    CAMPAIGN_RULES_SHA256,
    campaign_learning_seeds,
    load_champion_campaign_config,
)
from agentbench_frame.generals.evaluator import GeneralsEvaluation
from agentbench_frame.generals.leaderboard_qualification import (
    evaluate_leaderboard_qualification,
    load_leaderboard_qualification_config,
)
from agentbench_frame.generals.models import AssetLayout, ReplaySkillAsset
from agentbench_frame.generals.pipeline_campaign import ChampionCampaignPipeline
from agentbench_frame.generals.replay import (
    CriticalDecision,
    CriticalLearningEvidence,
)
from agentbench_frame.tracking.provider import ProviderInvocation, ProviderUsage
from agentbench_frame.tracking.providers import CodexProvider
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


FIXTURES = Path(__file__).parent / "fixtures"
ASSETS = (
    Path(__file__).resolve().parents[5]
    / "AgentBench/.worktrees/generals-assets"
    / "backend_sources/corpus/28_generals"
)
ENGINE_HASH = "4ca898365a6fe9a0cca39c3c8be0e49f449334814b106882c0b78b3aabdb1d97"
OPPONENT_HASH = "c261f48bea35b0757e45d28118a29e675f3ceabf608039fc0421104e88643db4"


class FakeProvider:
    provider_name = "fake-frontier"

    def __init__(self):
        self.contexts = []

    def invoke(self, context):
        self.contexts.append(dict(context))
        workspace = Path(context["workspace_root"])
        strategy = workspace / "strategy.py"
        strategy.write_text(strategy.read_text(encoding="utf-8") + "\n# act\n", encoding="utf-8")
        raw = Path(context["raw_output_path"])
        raw.write_text('{"type":"turn.completed"}\n', encoding="utf-8")
        return ProviderInvocation(
            status="completed",
            usage=ProviderUsage(10, 5, 15, "exact"),
            elapsed_time_s=0.01,
            raw_output_ref=str(raw),
            metadata={"total_cost_usd": 0.25},
        )


class OtherFakeProvider(FakeProvider):
    provider_name = "other-frontier"


class FakeEvaluator:
    def evaluate(self, _workspace, version, _phase, _run, *, cases):
        results = tuple(
            GameResult(case.case_id, "loss", True, metadata={"tier": "high"})
            for case in cases
        )
        return GeneralsEvaluation(
            version=version,
            status="complete",
            score=0.0,
            wins=0,
            losses=len(results),
            draws=0,
            per_tier={"high": 0.0, "medium": None, "low": None},
            seat_gap=0.0,
            results=results,
            matches=(),
        )


def _campaign():
    return load_champion_campaign_config(
        FIXTURES / "champion-campaign-v1.toml",
        load_pilot_config(FIXTURES / "pilot-v1.toml"),
        engine_sha256=ENGINE_HASH,
        action_space_spec_id=CAMPAIGN_ACTION_SPACE_SPEC_ID,
        decision_space_sha256=CAMPAIGN_DECISION_SPACE_SHA256,
        rules_sha256=CAMPAIGN_RULES_SHA256,
        replay_skill_sha256=CAMPAIGN_REPLAY_SKILL_SHA256,
        qualification_manifest_sha256=CAMPAIGN_QUALIFICATION_MANIFEST_SHA256,
    )


def _qualification():
    return load_leaderboard_qualification_config(
        FIXTURES / "leaderboard-qualification-v1.toml",
        load_pilot_config(FIXTURES / "pilot-v1.toml"),
        engine_sha256=ENGINE_HASH,
        opponent_tree_sha256=OPPONENT_HASH,
        replay_skill_sha256=CAMPAIGN_REPLAY_SKILL_SHA256,
    )


def _source(path):
    path.mkdir(parents=True)
    (path / "main.py").write_text("from strategy import choose_actions\n", encoding="utf-8")
    (path / "strategy.py").write_text(
        "def choose_actions(round_number, my_seat, view):\n    return [[8]]\n",
        encoding="utf-8",
    )
    (path / "state_view.py").write_text(
        "skills_cd = rest_move = rest_move_step = super_weapon_unlocked = super_weapon_cd = next_generals_id = None\n",
        encoding="utf-8",
    )
    (path / "STRATEGY.md").write_text(
        "beam top-k macro phase tie fallback\n", encoding="utf-8"
    )
    (path / "EXPERIENCE.md").write_text(
        "retained rejected risk\n", encoding="utf-8"
    )


def _evidence(config, act=1):
    return tuple(
        CriticalLearningEvidence(
            replay_id=f"campaign-replicate-1-a{act}-high-s{seed}-p{seat}",
            seed=seed,
            evaluated_seat=seat,
            opponent_tier="high",
            termination_type="normal",
            outcome="loss",
            dense={},
            total_decision_count=1,
            omitted_decision_count=0,
            decisions=(
                CriticalDecision(
                    state_id=f"state-{seed}-{seat}",
                    round_number=1,
                    seat=seat,
                    selection_reasons=("first_decision",),
                    action=((8,),),
                    decision_class="end_only",
                    features={},
                ),
            ),
        )
        for seed in campaign_learning_seeds(config, act)
        for seat in (0, 1)
    )


def _pipeline(tmp_path, provider):
    config = load_pilot_config(FIXTURES / "pilot-v1.toml")
    campaign = _campaign()
    qualification = _qualification()
    source = tmp_path / "initial"
    _source(source)
    initial_hash = LocalWorkspaceSnapshotter().capture(source).content_hash
    campaign = replace(campaign, initial_policy_hash=initial_hash)
    qualification = replace(qualification, initial_policy_hash=initial_hash)
    skill_path = ASSETS / "skills/replay-analysis-v2/SKILL.md"
    skill = ReplaySkillAsset(
        path=skill_path,
        text=skill_path.read_text(encoding="utf-8"),
        sha256=CAMPAIGN_REPLAY_SKILL_SHA256,
    )
    pipeline = ChampionCampaignPipeline(
        config=config,
        campaign=campaign,
        qualification=qualification,
        assets=AssetLayout(
            root=ASSETS.parents[2],
            engine_root=tmp_path / "engine",
            baseline_root=tmp_path / "baseline",
            opponents=config.opponents,
            engine_hash=ENGINE_HASH,
        ),
        replay_skill=skill,
        decision_space_path=ASSETS / "benchmark/decision-space-v1.md",
        initial_source=source,
        data_dir=tmp_path / "data",
        campaign_root=tmp_path / "campaign",
        provider=provider,
        model="fake-model",
        model_revision="r1",
        evaluator_factory=lambda _run_dir: FakeEvaluator(),
    )
    pipeline._critical_evidence = lambda _evaluation, **_kwargs: (_evidence(campaign), (), ())
    pipeline._baseline_tests = lambda _workspace: (True, "passed")
    pipeline._verify_probes = lambda _workspace, _probes: {"deterministic": True}
    return pipeline


def test_campaign_advances_one_act_without_leaking_checkpoint_feedback(tmp_path):
    provider = FakeProvider()
    pipeline = _pipeline(tmp_path, provider)

    result = pipeline.advance("replicate-1")

    assert result.status == "promoted"
    assert result.next_act_index == 2
    assert result.checkpoint_evaluated is True
    state = json.loads((tmp_path / "campaign/replicate-1/state.json").read_text())
    assert state["budget"] == {
        "coding_agent_acts": 1,
        "total_tokens": 15,
        "wall_time_s": state["budget"]["wall_time_s"],
        "learning_episodes": 6,
        "cost_usd": 0.25,
    }
    assert state["checkpoints"][0]["act_index"] == 1
    assert len(state["checkpoints"][0]["results"]) == 40
    prompt = provider.contexts[0]["prompt"]
    assert not Path(provider.contexts[0]["workspace_root"]).is_relative_to(
        result.run_dir
    )
    assert "304101" not in prompt
    assert "leaderboard-qualification" not in prompt
    assert "qualification receipt" not in prompt


def test_campaign_state_freezes_provider_model_and_revision(tmp_path):
    pipeline = _pipeline(tmp_path, FakeProvider())
    state = pipeline._initial_state("replicate-1")
    pipeline.provider = OtherFakeProvider()

    with pytest.raises(ValueError, match="campaign state identity changed"):
        pipeline._validate_state(state, "replicate-1")


def test_complete_replicates_render_a_gate_valid_receipt(tmp_path):
    pipeline = _pipeline(tmp_path, FakeProvider())
    config = pipeline.qualification
    snapshot = LocalWorkspaceSnapshotter().capture(pipeline.initial_source)
    for replica in range(1, 4):
        replicate_id = f"replicate-{replica}"
        root = tmp_path / "campaign" / replicate_id
        source = root / "policies/a32/source"
        pipeline._copy_manifest(pipeline.initial_source, source, snapshot)
        checkpoints = []
        for acts in config.budget_checkpoints:
            results = []
            seen = {0: 0, 1: 0}
            for seed in config.evaluation_seeds:
                for seat in config.seats:
                    results.append(
                        {
                            "case_id": f"qualify-{config.opponent_id}-s{seed}-p{seat}",
                            "seed": seed,
                            "seat": seat,
                            "outcome": "win" if seen[seat] < 13 else "loss",
                            "valid": True,
                        }
                    )
                    seen[seat] += 1
            checkpoints.append(
                {
                    "act_index": acts,
                    "budget": {
                        "coding_agent_acts": acts,
                        "total_tokens": acts * 15,
                        "wall_time_s": float(acts),
                        "learning_episodes": acts * 6,
                        "cost_usd": acts * 0.25,
                    },
                    "policy_hash": snapshot.content_hash,
                    "evaluation_status": "complete",
                    "results": results,
                }
            )
        acts = []
        for act_index in range(1, 33):
            trace = root / "provider-traces" / f"a{act_index}.jsonl"
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text(f"{{\"act\":{act_index}}}\n", encoding="utf-8")
            acts.append({"act_index": act_index, "run_dir": f"/runs/{replicate_id}-{act_index}"})
        state = {
            "schema": "generals-champion-campaign-state-v1",
            "campaign_id": pipeline.campaign.campaign_id,
            "replicate_id": replicate_id,
            "initial_policy_hash": pipeline.campaign.initial_policy_hash,
            "harness_hash": pipeline._harness_hash(),
            "next_act_index": 33,
            "current_policy_version": "campaign-final",
            "current_policy_hash": snapshot.content_hash,
            "current_policy_source": "policies/a32/source",
            "budget": checkpoints[-1]["budget"],
            "acts": acts,
            "checkpoints": checkpoints,
            "inflight": None,
        }
        (root / "state.json").write_text(json.dumps(state), encoding="utf-8")

    receipt = json.loads(pipeline.write_qualification_receipt().read_text())
    result = evaluate_leaderboard_qualification(config, receipt)

    assert result.status == "qualified"
    assert result.qualified is True
    assert result.qualifying_replicates == ("replicate-1", "replicate-2", "replicate-3")


def test_campaign_from_paths_resolves_the_production_skill_location(tmp_path):
    pipeline = ChampionCampaignPipeline.from_paths(
        agentbench_root=ASSETS.parents[2],
        manifest_path=ASSETS / "benchmark/pilot-v1.toml",
        campaign_manifest_path=ASSETS / "benchmark/champion-campaign-v1.toml",
        qualification_manifest_path=(
            ASSETS / "benchmark/leaderboard-qualification-v1.toml"
        ),
        replay_skill_path=Path(
            "backend_sources/corpus/28_generals/skills/replay-analysis-v2/SKILL.md"
        ),
        decision_space_path=ASSETS / "benchmark/decision-space-v1.md",
        data_dir=tmp_path / "data",
        campaign_root=tmp_path / "campaign",
        provider=CodexProvider(),
        model="frontier-model",
        model_revision="r1",
    )

    assert pipeline.replay_skill.sha256 == CAMPAIGN_REPLAY_SKILL_SHA256
    assert pipeline.initial_source == (
        tmp_path
        / "data/runs/28_generals/generals-hl/20260730_1739_680b1632/versions/v7/source"
    )
