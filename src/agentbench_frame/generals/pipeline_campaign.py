"""Resumable, leak-closed execution for the Generals champion campaign."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from agentbench_frame.eval.benchmark import BenchmarkCase
from agentbench_frame.tracking.provider import ProviderAdapter
from agentbench_frame.tracking.run import Run
from agentbench_frame.tracking.snapshot import WorkspaceManifest

from .action_profile import action_profile_payload, summarize_action_profile
from .assets import (
    _stable_tree_hash,
    load_pilot_config,
    require_valid_assets,
    resolve_assets,
    resolve_replay_skill,
)
from .champion_campaign import (
    build_campaign_learning_cases,
    load_champion_campaign_config,
)
from .dense import build_dense_trace, summarize_dense_trace
from .evaluator import GeneralsEvaluation, GeneralsEvaluator
from .leaderboard_qualification import (
    QUALIFICATION_SCHEMA,
    LeaderboardQualificationConfig,
    load_leaderboard_qualification_config,
)
from .measurement import ProbeState
from .models import AssetLayout, ChampionCampaignConfig, PilotConfig, ReplaySkillAsset
from .pipeline import GeneralsHLPipeline
from .prompt_campaign import build_champion_campaign_prompt
from .replay import (
    CriticalLearningEvidence,
    CriticalWindowSelection,
    build_critical_learning_evidence,
    build_learning_replay,
)


CAMPAIGN_STATE_SCHEMA = "generals-champion-campaign-state-v1"
CAMPAIGN_COMMIT_SCHEMA = "generals-champion-campaign-commit-v1"
_STATE_FIELDS = {
    "schema",
    "campaign_id",
    "replicate_id",
    "initial_policy_hash",
    "harness_hash",
    "next_act_index",
    "current_policy_version",
    "current_policy_hash",
    "current_policy_source",
    "budget",
    "acts",
    "checkpoints",
    "inflight",
}
_BUDGET_FIELDS = (
    "coding_agent_acts",
    "total_tokens",
    "wall_time_s",
    "learning_episodes",
    "cost_usd",
)
_REQUIRED_SOURCE_FILES = frozenset(
    {"main.py", "strategy.py", "state_view.py", "STRATEGY.md", "EXPERIENCE.md"}
)
_STATE_VIEW_MARKERS = (
    "skills_cd",
    "rest_move",
    "rest_move_step",
    "super_weapon_unlocked",
    "super_weapon_cd",
    "next_generals_id",
)
_RUNTIME_FORBIDDEN = (
    "random",
    "socket",
    "requests",
    "urllib",
    "subprocess",
    "replay_id",
    "opponent_id",
    "time.time",
    "os.environ",
    "__file__",
    "top_algorithms/",
    "advanced-rank02-robinliu-v18",
)


@dataclass(frozen=True)
class ChampionCampaignActResult:
    campaign_root: Path
    run_dir: Path
    replicate_id: str
    act_index: int
    status: str
    promoted: bool
    current_policy_hash: str
    checkpoint_evaluated: bool
    next_act_index: int


class CampaignRecoveryRequired(ValueError):
    """An interrupted provider act requires recovery instead of replay."""


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    os.replace(temporary, path)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read campaign {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"campaign {label} must be an object")
    return value


def _is_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _state_budget(value: Mapping[str, Any]) -> dict[str, int | float | None]:
    if set(value) != set(_BUDGET_FIELDS):
        raise ValueError("campaign budget fields changed")
    result: dict[str, int | float | None] = {}
    for field in _BUDGET_FIELDS:
        item = value[field]
        if item is None:
            result[field] = None
        elif field in {"coding_agent_acts", "total_tokens", "learning_episodes"}:
            if type(item) is not int or item < 0:
                raise ValueError(f"campaign budget {field} is invalid")
            result[field] = item
        elif not _is_number(item) or float(item) < 0:
            raise ValueError(f"campaign budget {field} is invalid")
        else:
            result[field] = float(item)
    return result


def _zero_budget() -> dict[str, int | float | None]:
    return {
        "coding_agent_acts": 0,
        "total_tokens": 0,
        "wall_time_s": 0.0,
        "learning_episodes": 0,
        "cost_usd": 0.0,
    }


def _add_budget(
    budget: Mapping[str, int | float | None],
    *,
    usage_tokens: int | None,
    elapsed_time_s: float,
    learning_episodes: int,
    cost_usd: float | None,
) -> dict[str, int | float | None]:
    current = _state_budget(budget)
    if current["coding_agent_acts"] is None:
        raise ValueError("campaign act count cannot be unknown")
    result = dict(current)
    result["coding_agent_acts"] = int(current["coding_agent_acts"]) + 1
    result["learning_episodes"] = int(current["learning_episodes"] or 0) + learning_episodes
    result["wall_time_s"] = float(current["wall_time_s"] or 0.0) + elapsed_time_s
    result["total_tokens"] = (
        None
        if current["total_tokens"] is None or usage_tokens is None
        else int(current["total_tokens"]) + usage_tokens
    )
    result["cost_usd"] = (
        None
        if current["cost_usd"] is None or cost_usd is None
        else float(current["cost_usd"]) + cost_usd
    )
    return result


class ChampionCampaignPipeline(GeneralsHLPipeline):
    """Advance exactly one fresh-learning act for one fixed replicate."""

    def __init__(
        self,
        *,
        config: PilotConfig,
        campaign: ChampionCampaignConfig,
        qualification: LeaderboardQualificationConfig,
        assets: AssetLayout,
        replay_skill: ReplaySkillAsset,
        decision_space_path: Path,
        initial_source: Path,
        data_dir: Path,
        campaign_root: Path,
        provider: ProviderAdapter,
        model: str,
        model_revision: str,
        evaluator_factory: Callable[[Path], GeneralsEvaluator] | None = None,
    ) -> None:
        super().__init__(config, assets, data_dir, provider)
        if not model.strip() or not model_revision.strip():
            raise ValueError("campaign model and model_revision are required")
        self.campaign = campaign
        self.qualification = qualification
        self.replay_skill = replay_skill
        self.decision_space_path = Path(decision_space_path)
        self.initial_source = Path(initial_source)
        self.campaign_root = Path(campaign_root)
        self.model = model
        self.model_revision = model_revision
        self.evaluator_factory = evaluator_factory

    @classmethod
    def from_paths(
        cls,
        *,
        agentbench_root: Path,
        manifest_path: Path,
        campaign_manifest_path: Path,
        qualification_manifest_path: Path,
        replay_skill_path: Path,
        decision_space_path: Path,
        data_dir: Path,
        campaign_root: Path,
        provider: ProviderAdapter,
        model: str,
        model_revision: str,
    ) -> "ChampionCampaignPipeline":
        config = load_pilot_config(manifest_path)
        assets = resolve_assets(config, agentbench_root)
        require_valid_assets(assets)
        replay_skill = resolve_replay_skill(agentbench_root, replay_skill_path)
        opponent = next(
            item for item in assets.opponents if item.opponent_id == config.opponents[0].opponent_id
        )
        qualification = load_leaderboard_qualification_config(
            qualification_manifest_path,
            config,
            engine_sha256=assets.engine_hash,
            opponent_tree_sha256=_stable_tree_hash(opponent.source),
            replay_skill_sha256=replay_skill.sha256,
        )
        campaign = load_champion_campaign_config(
            campaign_manifest_path,
            config,
            engine_sha256=assets.engine_hash,
            action_space_spec_id="a383f61cba2b284623c0b377eaddee4ef7522b8e8adb53e0ea3efb7bc9bc329e",
            decision_space_sha256=hashlib.sha256(
                Path(decision_space_path).read_bytes()
            ).hexdigest(),
            rules_sha256=hashlib.sha256(
                (assets.root / "backend_sources/corpus/28_generals/benchmark/rules.md").read_bytes()
            ).hexdigest(),
            replay_skill_sha256=replay_skill.sha256,
            qualification_manifest_sha256=hashlib.sha256(
                Path(qualification_manifest_path).read_bytes()
            ).hexdigest(),
        )
        initial_source = (
            Path(data_dir)
            / "runs/28_generals/generals-hl"
            / campaign.initial_run_id
            / "versions/v7/source"
        )
        return cls(
            config=config,
            campaign=campaign,
            qualification=qualification,
            assets=assets,
            replay_skill=replay_skill,
            decision_space_path=decision_space_path,
            initial_source=initial_source,
            data_dir=data_dir,
            campaign_root=campaign_root,
            provider=provider,
            model=model,
            model_revision=model_revision,
        )

    def _replicate_root(self, replicate_id: str) -> Path:
        expected = {
            f"replicate-{index}" for index in range(1, self.campaign.replicates + 1)
        }
        if replicate_id not in expected:
            raise ValueError("campaign replicate ID is not frozen")
        return self.campaign_root / replicate_id

    def _state_path(self, replicate_id: str) -> Path:
        return self._replicate_root(replicate_id) / "state.json"

    def _source_path(self, replicate_id: str, reference: str) -> Path:
        root = self._replicate_root(replicate_id).resolve()
        path = (root / reference).resolve()
        if root != path and root not in path.parents:
            raise ValueError("campaign policy source escapes its replicate root")
        return path

    def _copy_manifest(self, source: Path, destination: Path, manifest: WorkspaceManifest) -> None:
        self.snapshotter.materialize_manifest(source, destination, manifest)

    def _initial_state(self, replicate_id: str) -> dict[str, Any]:
        if not self.initial_source.is_dir():
            raise ValueError("frozen v7 initial source is missing")
        manifest = self.snapshotter.capture(self.initial_source)
        if manifest.content_hash != self.campaign.initial_policy_hash:
            raise ValueError("frozen v7 initial source hash changed")
        root = self._replicate_root(replicate_id)
        target = root / "policies/a0/source"
        self._copy_manifest(self.initial_source, target, manifest)
        self.snapshotter.write_manifest(manifest, target.parent / "manifest.json")
        return {
            "schema": CAMPAIGN_STATE_SCHEMA,
            "campaign_id": self.campaign.campaign_id,
            "replicate_id": replicate_id,
            "initial_policy_hash": self.campaign.initial_policy_hash,
            "harness_hash": self._harness_hash(),
            "next_act_index": 1,
            "current_policy_version": self.campaign.initial_policy_version,
            "current_policy_hash": manifest.content_hash,
            "current_policy_source": "policies/a0/source",
            "budget": _zero_budget(),
            "acts": [],
            "checkpoints": [],
            "inflight": None,
        }

    def _validate_state(self, state: Mapping[str, Any], replicate_id: str) -> dict[str, Any]:
        if set(state) != _STATE_FIELDS:
            raise ValueError("campaign state fields changed")
        if state.get("schema") != CAMPAIGN_STATE_SCHEMA:
            raise ValueError("campaign state schema changed")
        identities = (
            (state.get("campaign_id"), self.campaign.campaign_id),
            (state.get("replicate_id"), replicate_id),
            (state.get("initial_policy_hash"), self.campaign.initial_policy_hash),
            (state.get("harness_hash"), self._harness_hash()),
        )
        if any(actual != expected for actual, expected in identities):
            raise ValueError("campaign state identity changed")
        next_act = state.get("next_act_index")
        if type(next_act) is not int or not 1 <= next_act <= self.campaign.max_acts + 1:
            raise ValueError("campaign next act index is invalid")
        current_hash = state.get("current_policy_hash")
        if not isinstance(current_hash, str) or len(current_hash) != 64:
            raise ValueError("campaign current policy hash is invalid")
        source_ref = state.get("current_policy_source")
        if not isinstance(source_ref, str):
            raise ValueError("campaign current policy source is invalid")
        source = self._source_path(replicate_id, source_ref)
        manifest = self.snapshotter.capture(source)
        if manifest.content_hash != current_hash:
            raise ValueError("campaign current policy source hash changed")
        if not isinstance(state.get("acts"), list) or not isinstance(state.get("checkpoints"), list):
            raise ValueError("campaign state history is invalid")
        inflight = state.get("inflight")
        if inflight is not None and not isinstance(inflight, dict):
            raise ValueError("campaign inflight state is invalid")
        normalized = dict(state)
        normalized["budget"] = _state_budget(state["budget"])
        return normalized

    def _load_or_create_state(self, replicate_id: str) -> dict[str, Any]:
        path = self._state_path(replicate_id)
        if not path.exists():
            state = self._initial_state(replicate_id)
            _write_json_atomic(path, state)
            return state
        state = self._validate_state(_read_json_object(path, "state"), replicate_id)
        if state["inflight"] is not None:
            inflight = state["inflight"]
            assert isinstance(inflight, dict)
            commit_path = Path(str(inflight.get("run_dir", ""))) / "campaign-commit.json"
            if not commit_path.is_file():
                raise CampaignRecoveryRequired(
                    "a previous provider act is inflight; recover its recorded run instead of invoking again"
                )
            commit = _read_json_object(commit_path, "act commit")
            if commit.get("schema") != CAMPAIGN_COMMIT_SCHEMA or not isinstance(commit.get("state"), dict):
                raise CampaignRecoveryRequired("campaign inflight commit is invalid")
            state = self._validate_state(commit["state"], replicate_id)
            if state["inflight"] is not None:
                raise CampaignRecoveryRequired("campaign inflight commit did not finalize")
            _write_json_atomic(path, state)
        return state

    def _evaluator_for(self, run_dir: Path) -> GeneralsEvaluator:
        if self.evaluator_factory is not None:
            return self.evaluator_factory(run_dir)
        if self.evaluator is not None:
            return self.evaluator
        return self._production_evaluator(run_dir, capture_measurement_states=True)

    @staticmethod
    def _critical_evidence(
        evaluation: GeneralsEvaluation,
        *,
        replicate_id: str,
        act_index: int,
    ) -> tuple[
        tuple[CriticalLearningEvidence, ...],
        tuple[CriticalWindowSelection, ...],
        tuple[ProbeState, ...],
    ]:
        evidence: list[CriticalLearningEvidence] = []
        selections: list[CriticalWindowSelection] = []
        probes: list[ProbeState] = []
        for match in evaluation.matches:
            replay_id = (
                f"campaign-{replicate_id}-a{act_index}-high-"
                f"s{match.seed}-p{match.evaluated_seat}"
            )
            raw = build_learning_replay(match, "campaign", "high")
            replay = replace(
                raw,
                replay_id=replay_id,
                decisions=tuple(
                    replace(
                        decision,
                        state_id=f"campaign:{replay_id}-d{index}",
                    )
                    for index, decision in enumerate(raw.decisions, start=1)
                ),
            )
            trace = build_dense_trace(match)
            dense = summarize_dense_trace(match, trace)
            compact, selection = build_critical_learning_evidence(
                replay,
                dense,
                trace,
                max_decisions=6,
                selection_reasons=(
                    "first_decision",
                    "first_non_end_action",
                    "first_main_pressure",
                    "before_steepest_territory_drop",
                    "before_steepest_army_drop",
                    "first_strategic_opportunity",
                    "final_decision",
                ),
                include_strategic_targets=True,
            )
            selected = set(selection.selected_state_ids)
            evidence.append(compact)
            selections.append(selection)
            probes.extend(
                ProbeState(
                    decision.state_id,
                    {**decision.state, "my_seat": decision.seat},
                    decision.action,
                )
                for decision in replay.decisions
                if decision.state_id in selected
            )
        return tuple(evidence), tuple(selections), tuple(probes)

    @staticmethod
    def _scope_valid(changed_files: Sequence[str]) -> bool:
        return all(
            path in {"strategy.py", "state_view.py", "STRATEGY.md", "EXPERIENCE.md"}
            or path.startswith("tests/")
            or path.startswith("policy/")
            for path in changed_files
        )

    @staticmethod
    def _required_files_valid(source: Path, manifest: WorkspaceManifest) -> bool:
        return all(
            relative in manifest.files
            and (source / relative).is_file()
            and not (source / relative).is_symlink()
            for relative in _REQUIRED_SOURCE_FILES
        )

    @staticmethod
    def _runtime_source_valid(source: Path) -> bool:
        total = 0
        for path in source.rglob("*"):
            if path.is_symlink():
                return False
            if not path.is_file():
                continue
            total += path.stat().st_size
            relative = path.relative_to(source).as_posix()
            if relative.startswith("tests/") or path.suffix != ".py":
                continue
            text = path.read_text(encoding="utf-8")
            if any(token in text for token in _RUNTIME_FORBIDDEN):
                return False
        return total <= 10_485_760

    @staticmethod
    def _documents_valid(source: Path) -> bool:
        try:
            strategy = (source / "STRATEGY.md").read_text(encoding="utf-8").casefold()
            experience = (source / "EXPERIENCE.md").read_text(encoding="utf-8").casefold()
        except OSError:
            return False
        required_strategy = (("beam",), ("top-k", "top k"), ("macro",), ("phase",), ("tie",), ("fallback",))
        required_experience = (("retain", "retained"), ("reject", "rejected"), ("risk",))
        return all(any(token in strategy for token in group) for group in required_strategy) and all(
            any(token in experience for token in group) for group in required_experience
        )

    @staticmethod
    def _full_state_view_valid(source: Path) -> bool:
        try:
            text = (source / "state_view.py").read_text(encoding="utf-8")
        except OSError:
            return False
        return all(marker in text for marker in _STATE_VIEW_MARKERS)

    @staticmethod
    def _command_shape_valid(command: tuple[int, ...]) -> bool:
        opcode = command[0] if command else None
        if opcode == 1:
            return len(command) == 5 and command[4] > 0
        if opcode == 2:
            return len(command) == 4
        if opcode == 3:
            return len(command) == 3
        if opcode == 4:
            return (
                len(command) == 5 if len(command) >= 3 and command[2] in (1, 2)
                else len(command) == 3 if len(command) >= 3 and command[2] in (3, 4, 5)
                else False
            )
        if opcode == 5:
            return len(command) == 2
        if opcode == 6:
            return (
                len(command) == 6 if len(command) >= 2 and command[1] == 3
                else len(command) == 4 if len(command) >= 2 and command[1] in (1, 2, 4)
                else False
            )
        return len(command) == 3 if opcode == 7 else command == (8,)

    def _verify_probes(self, workspace: Path, probes: tuple[ProbeState, ...]) -> dict[str, object]:
        if not probes:
            raise ValueError("campaign learning produced no policy probes")
        started = time.perf_counter()
        first = self._probe_actions(workspace, probes)
        second = self._probe_actions(workspace, probes)
        elapsed = time.perf_counter() - started
        if first != second or set(first) != {probe.state_id for probe in probes}:
            raise ValueError("candidate policy probes are nondeterministic")
        for probe in probes:
            commands = first[probe.state_id]
            if [index for index, command in enumerate(commands) if command == (8,)] != [len(commands) - 1]:
                raise ValueError("candidate macro must contain exactly one final [8]")
            if len(commands) - 1 > self.campaign.max_non_end_primitives:
                raise ValueError("candidate macro exceeds campaign primitive limit")
            if not all(self._command_shape_valid(command) for command in commands):
                raise ValueError("candidate emitted malformed command")
        mean_call_s = elapsed / (2 * len(probes))
        if mean_call_s >= 1.5:
            raise ValueError("candidate lacks headroom below the two-second limit")
        return {"decision_count": len(probes), "deterministic": True, "mean_call_s": mean_call_s}

    def _qualification_cases(self) -> tuple[BenchmarkCase, ...]:
        opponent = next(
            item for item in self.config.opponents if item.opponent_id == self.qualification.opponent_id
        )
        return tuple(
            BenchmarkCase(
                case_id=f"qualify-{opponent.opponent_id}-s{seed}-p{seat}",
                opponent=opponent.opponent_id,
                seed=seed,
                first_player=seat,
                metadata={"tier": opponent.tier, "phase": "leaderboard-qualification"},
            )
            for seed in self.qualification.evaluation_seeds
            for seat in self.qualification.seats
        )

    def _checkpoint_payload(
        self,
        *,
        evaluation: GeneralsEvaluation,
        cases: Sequence[BenchmarkCase],
        budget: Mapping[str, int | float | None],
        policy_hash: str,
    ) -> dict[str, Any]:
        rows = []
        by_id = {result.case_id: result for result in evaluation.results}
        for case in cases:
            result = by_id.get(case.case_id)
            if result is None:
                continue
            rows.append(
                {
                    "case_id": case.case_id,
                    "seed": case.seed,
                    "seat": case.first_player,
                    "outcome": result.outcome,
                    "valid": result.valid,
                }
            )
        return {
            "budget": dict(budget),
            "policy_hash": policy_hash,
            "evaluation_status": evaluation.status,
            "results": rows,
        }

    def _provider_trace_path(self, replicate_id: str, act_index: int) -> Path:
        return self._replicate_root(replicate_id) / "provider-traces" / f"a{act_index}.jsonl"

    def _persist_provider_trace(self, source: str | None, target: Path) -> str | None:
        if not source:
            return None
        original = Path(source)
        if not original.is_file():
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
        return _digest_bytes(target.read_bytes())

    def advance(self, replicate_id: str) -> ChampionCampaignActResult:
        state = self._load_or_create_state(replicate_id)
        act_index = int(state["next_act_index"])
        if act_index > self.campaign.max_acts:
            return ChampionCampaignActResult(
                campaign_root=self.campaign_root,
                run_dir=self._replicate_root(replicate_id),
                replicate_id=replicate_id,
                act_index=self.campaign.max_acts,
                status="complete",
                promoted=False,
                current_policy_hash=str(state["current_policy_hash"]),
                checkpoint_evaluated=False,
                next_act_index=act_index,
            )

        source = self._source_path(replicate_id, str(state["current_policy_source"]))
        source_manifest = self.snapshotter.capture(source)
        if source_manifest.content_hash != state["current_policy_hash"]:
            raise ValueError("campaign parent policy changed before act")
        run = Run.start(
            game="28_generals",
            agent="generals-champion-campaign",
            run_type="rule_iter",
            data_dir=str(self.data_dir),
            config={
                "campaign_id": self.campaign.campaign_id,
                "replicate_id": replicate_id,
                "act_index": act_index,
                "budget_phase": "learning",
                "provider": self.provider.provider_name,
                "model": self.model,
                "model_revision": self.model_revision,
            },
        )
        run_dir = Path(run.run_dir)
        state["inflight"] = {
            "act_index": act_index,
            "run_dir": str(run_dir),
            "parent_policy_hash": source_manifest.content_hash,
        }
        _write_json_atomic(self._state_path(replicate_id), state)

        started = time.perf_counter()
        promoted = False
        checkpoint_evaluated = False
        status = "failed"
        checkpoint: dict[str, Any] | None = None
        candidate_hash = source_manifest.content_hash
        provider_act = None
        learning_case_count = 0
        provider_workspace: Path | None = None
        try:
            learning_cases = build_campaign_learning_cases(
                self.config,
                self.campaign,
                replicate_id=replicate_id,
                act_index=act_index,
            )
            learning_case_count = len(learning_cases)
            evaluator = self._evaluator_for(run_dir)
            learning = evaluator.evaluate(
                source,
                str(state["current_policy_version"]),
                "campaign-learning",
                run,
                cases=learning_cases,
            )
            if learning.status != "complete" or len(learning.results) != len(learning_cases):
                raise ValueError("campaign learning suite is incomplete")
            evidence, selections, probes = self._critical_evidence(
                learning,
                replicate_id=replicate_id,
                act_index=act_index,
            )
            evidence_dir = run_dir / "learning-evidence"
            for item, selection in zip(evidence, selections):
                path = evidence_dir / f"{item.replay_id}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(item.to_json() + "\n", encoding="utf-8")
                run.write(
                    "critical_window_selection",
                    replay_id=selection.replay_id,
                    selected_state_ids=list(selection.selected_state_ids),
                    reasons={key: list(value) for key, value in selection.reasons.items()},
                    artifact_ref=str(path),
                )
            action_profile = action_profile_payload(summarize_action_profile(learning))
            rules_text = self.rules_path.read_text(encoding="utf-8")
            decision_space_text = self.decision_space_path.read_text(encoding="utf-8")
            prompt = build_champion_campaign_prompt(
                config=self.campaign,
                benchmark_id=self.config.benchmark_id,
                replicate_id=replicate_id,
                act_index=act_index,
                current_policy_version=str(state["current_policy_version"]),
                current_policy_hash=source_manifest.content_hash,
                rules_text=rules_text,
                decision_space_text=decision_space_text,
                replay_skill_text=self.replay_skill.text,
                evidence=evidence,
                action_profile=action_profile,
            )
            expected_ids = {case.case_id for case in learning_cases}
            if prompt.truncated or set(prompt.included_episode_ids) != expected_ids:
                raise ValueError("campaign prompt did not include exactly the learning suite")
            provider_dir = run_dir / "provider"
            provider_dir.mkdir(parents=True, exist_ok=True)
            (provider_dir / "prompt.txt").write_text(prompt.prompt, encoding="utf-8")
            _write_json_atomic(provider_dir / "prompt-manifest.json", dict(prompt.manifest))
            provider_workspace = Path(
                tempfile.mkdtemp(
                    prefix=f"generals-campaign-{replicate_id}-a{act_index}-"
                )
            )
            workspace = provider_workspace
            self._copy_manifest(source, workspace, source_manifest)
            raw_path = provider_dir / "provider.raw.jsonl"
            stderr_path = provider_dir / "provider.stderr.log"
            raw_path.touch()
            stderr_path.touch()
            controller = run.create_coding_agent_controller(
                self.provider,
                snapshotter=self.snapshotter,
                budget_phase="learning",
            )
            act = controller.run_act(
                {
                    "prompt": prompt.prompt,
                    "workspace_root": str(workspace),
                    "raw_output_path": str(raw_path),
                    "stderr_output_path": str(stderr_path),
                    "max_artifact_bytes": self.config.limits.max_artifact_bytes,
                },
                workspace_root=str(workspace),
                version_before=str(state["current_policy_version"]),
                previous_manifest=source_manifest,
            )
            provider_act = act
            candidate_manifest = self.snapshotter.capture(workspace, previous=source_manifest)
            candidate_hash = candidate_manifest.content_hash
            candidate_dir = run_dir / "candidate"
            self._copy_manifest(workspace, candidate_dir / "source", candidate_manifest)
            self.snapshotter.write_manifest(candidate_manifest, candidate_dir / "manifest.json")
            tests_ok, test_output = self._baseline_tests(workspace)
            (candidate_dir / "tests.log").write_text(test_output, encoding="utf-8")
            main_unchanged = (
                candidate_manifest.files.get("main.py") == source_manifest.files.get("main.py")
            )
            probe_diagnostics: Mapping[str, object] | None = None
            probe_error = None
            try:
                probe_diagnostics = self._verify_probes(workspace, probes)
            except Exception as exc:
                probe_error = f"{type(exc).__name__}: {exc}"
            gates = {
                "provider_completed": act.status == "completed",
                "scope_valid": self._scope_valid(candidate_manifest.changed_files),
                "main_unchanged": main_unchanged,
                "required_files_valid": self._required_files_valid(workspace, candidate_manifest),
                "runtime_source_valid": self._runtime_source_valid(workspace),
                "documents_valid": self._documents_valid(workspace),
                "full_state_view_valid": self._full_state_view_valid(workspace),
                "tests_passed": tests_ok,
                "probe_valid": probe_diagnostics is not None,
            }
            promoted = all(gates.values())
            run.write(
                "campaign_candidate_gate",
                act_index=act_index,
                parent_policy_hash=source_manifest.content_hash,
                candidate_policy_hash=candidate_manifest.content_hash,
                changed_files=candidate_manifest.changed_files,
                gates=gates,
                probe_diagnostics=probe_diagnostics,
                probe_error=probe_error,
                promoted=promoted,
            )
            trace_hash = self._persist_provider_trace(
                act.raw_output_ref,
                self._provider_trace_path(replicate_id, act_index),
            )
            cost = act.provider_metadata.get(
                "cost_usd", act.provider_metadata.get("total_cost_usd")
            )
            cost_value = float(cost) if _is_number(cost) and float(cost) >= 0 else None
            budget = _add_budget(
                state["budget"],
                usage_tokens=act.total_tokens,
                elapsed_time_s=time.perf_counter() - started,
                learning_episodes=len(learning_cases),
                cost_usd=cost_value,
            )
            if promoted:
                target_ref = f"policies/a{act_index}/source"
                target = self._source_path(replicate_id, target_ref)
                self._copy_manifest(workspace, target, candidate_manifest)
                self.snapshotter.write_manifest(candidate_manifest, target.parent / "manifest.json")
                current_version = f"campaign-{replicate_id}-a{act_index}"
                current_hash = candidate_manifest.content_hash
                status = "promoted"
            else:
                target_ref = str(state["current_policy_source"])
                current_version = str(state["current_policy_version"])
                current_hash = source_manifest.content_hash
                status = "candidate_rejected"
            if act_index in self.campaign.checkpoint_acts:
                qualification_cases = self._qualification_cases()
                checkpoint_workspace = self._source_path(replicate_id, target_ref)
                qualification = evaluator.evaluate(
                    checkpoint_workspace,
                    current_version,
                    "leaderboard-qualification",
                    run,
                    cases=qualification_cases,
                )
                checkpoint = self._checkpoint_payload(
                    evaluation=qualification,
                    cases=qualification_cases,
                    budget=budget,
                    policy_hash=current_hash,
                )
                checkpoint_evaluated = True
                run.write(
                    "sealed_qualification_checkpoint",
                    act_index=act_index,
                    policy_hash=current_hash,
                    evaluation_status=qualification.status,
                    result_count=len(qualification.results),
                    provider_visible=False,
                    selection_uses_results=False,
                )
        except Exception as exc:
            run.write("campaign_error", act_index=act_index, error=f"{type(exc).__name__}: {exc}")
            if provider_act is None:
                status = "learning_or_execution_failed"
                budget = state["budget"]
                trace_hash = None
            else:
                status = "candidate_rejected"
                trace_hash = self._persist_provider_trace(
                    provider_act.raw_output_ref,
                    self._provider_trace_path(replicate_id, act_index),
                )
                cost = provider_act.provider_metadata.get(
                    "cost_usd",
                    provider_act.provider_metadata.get("total_cost_usd"),
                )
                cost_value = (
                    float(cost)
                    if _is_number(cost) and float(cost) >= 0
                    else None
                )
                budget = _add_budget(
                    state["budget"],
                    usage_tokens=provider_act.total_tokens,
                    elapsed_time_s=time.perf_counter() - started,
                    learning_episodes=learning_case_count,
                    cost_usd=cost_value,
                )
            current_version = str(state["current_policy_version"])
            current_hash = source_manifest.content_hash
            target_ref = str(state["current_policy_source"])
        finally:
            run.finish(
                {
                    "campaign_status": status,
                    "replicate_id": replicate_id,
                    "act_index": act_index,
                    "promoted": promoted,
                    "checkpoint_evaluated": checkpoint_evaluated,
                }
            )
            if provider_workspace is not None:
                shutil.rmtree(provider_workspace, ignore_errors=True)

        if status == "learning_or_execution_failed":
            # No coding act was safely recorded, so preserve the same act identity.
            state["inflight"] = None
            _write_json_atomic(self._state_path(replicate_id), state)
            return ChampionCampaignActResult(
                campaign_root=self.campaign_root,
                run_dir=run_dir,
                replicate_id=replicate_id,
                act_index=act_index,
                status=status,
                promoted=False,
                current_policy_hash=str(state["current_policy_hash"]),
                checkpoint_evaluated=False,
                next_act_index=act_index,
            )

        next_state = dict(state)
        next_state.update(
            {
                "next_act_index": act_index + 1,
                "current_policy_version": current_version,
                "current_policy_hash": current_hash,
                "current_policy_source": target_ref,
                "budget": budget,
                "inflight": None,
            }
        )
        next_state["acts"] = [
            *state["acts"],
            {
                "act_index": act_index,
                "run_dir": str(run_dir),
                "parent_policy_hash": source_manifest.content_hash,
                "candidate_policy_hash": candidate_hash,
                "promoted": promoted,
                "status": status,
                "provider_trace_sha256": trace_hash,
            },
        ]
        if checkpoint is not None:
            next_state["checkpoints"] = [
                *state["checkpoints"],
                {"act_index": act_index, **checkpoint},
            ]
        commit = {"schema": CAMPAIGN_COMMIT_SCHEMA, "state": next_state}
        _write_json_atomic(run_dir / "campaign-commit.json", commit)
        _write_json_atomic(self._state_path(replicate_id), next_state)
        self.write_qualification_receipt()
        return ChampionCampaignActResult(
            campaign_root=self.campaign_root,
            run_dir=run_dir,
            replicate_id=replicate_id,
            act_index=act_index,
            status=status,
            promoted=promoted,
            current_policy_hash=current_hash,
            checkpoint_evaluated=checkpoint_evaluated,
            next_act_index=act_index + 1,
        )

    def _provider_trace_digest(self, state: Mapping[str, Any], replicate_id: str) -> str | None:
        if len(state["acts"]) != self.campaign.max_acts:
            return None
        chunks: list[bytes] = []
        for act in state["acts"]:
            if not isinstance(act, Mapping):
                return None
            index = act.get("act_index")
            if type(index) is not int:
                return None
            path = self._provider_trace_path(replicate_id, index)
            if not path.is_file():
                return None
            chunks.append(path.read_bytes())
        return _digest_bytes(b"".join(chunks))

    def _harness_hash(self) -> str:
        module_paths = (
            Path(__file__),
            Path(__file__).with_name("prompt_campaign.py"),
            Path(__file__).with_name("leaderboard_qualification.py"),
        )
        payload = {
            "modules": {path.name: _digest_bytes(path.read_bytes()) for path in module_paths},
            "campaign_id": self.campaign.campaign_id,
            "qualification_id": self.qualification.qualification_id,
            "decision_space_sha256": self.campaign.decision_space_sha256,
            "provider": self.provider.provider_name,
            "model": self.model,
            "model_revision": self.model_revision,
        }
        return _digest_bytes(_json_bytes(payload))

    def write_qualification_receipt(self) -> Path | None:
        """Write a sealed aggregate receipt without making it provider input."""
        replicates = []
        for index in range(1, self.campaign.replicates + 1):
            replicate_id = f"replicate-{index}"
            path = self._state_path(replicate_id)
            if not path.is_file():
                continue
            state = self._validate_state(_read_json_object(path, "state"), replicate_id)
            trace = self._provider_trace_digest(state, replicate_id)
            if trace is None:
                continue
            budget = _state_budget(state["budget"])
            if any(budget[field] is None for field in _BUDGET_FIELDS):
                continue
            checkpoints = []
            for item in state["checkpoints"]:
                if not isinstance(item, Mapping):
                    continue
                checkpoint = {key: item[key] for key in ("budget", "policy_hash", "evaluation_status", "results") if key in item}
                if set(checkpoint) == {"budget", "policy_hash", "evaluation_status", "results"}:
                    checkpoints.append(checkpoint)
            run_id = state["acts"][-1]["run_dir"] if state["acts"] else None
            if not isinstance(run_id, str):
                continue
            replicates.append(
                {
                    "replicate_id": replicate_id,
                    "run_id": Path(run_id).name,
                    "provider_trace_sha256": trace,
                    "checkpoints": checkpoints,
                }
            )
        if len(replicates) != self.campaign.replicates:
            return None
        receipt = {
            "schema": QUALIFICATION_SCHEMA,
            "qualification_id": self.qualification.qualification_id,
            "provider": self.provider.provider_name,
            "model": self.model,
            "model_revision": self.model_revision,
            "harness_hash": self._harness_hash(),
            "initial_policy_hash": self.qualification.initial_policy_hash,
            "engine_sha256": self.qualification.engine_sha256,
            "opponent_id": self.qualification.opponent_id,
            "opponent_tree_sha256": self.qualification.opponent_tree_sha256,
            "replay_skill_sha256": self.qualification.replay_skill_sha256,
            "qualification_results_exposed_to_provider": False,
            "selection_uses_qualification_results": False,
            "replicates": replicates,
        }
        output = self.campaign_root / "sealed" / "qualification-receipt.json"
        _write_json_atomic(output, receipt)
        return output
