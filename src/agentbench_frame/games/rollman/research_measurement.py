"""Fixed-reference policy KL and separate rollout occupancy measurement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agentbench_frame.games.rollman.measurement import decision_trace_metrics
from agentbench_frame.games.rollman.policy_probe import (
    load_trace_decisions,
    run_probe_episode,
)
from agentbench_frame.hl.codebase import Version, VersionStore
from agentbench_frame.hl.evaluator import CandidateEvaluation


ProbeRunner = Callable[..., tuple[dict[str, Any], ...]]


class RollmanMeasurementRunner:
    """Measure every version pair on origin trajectories, not inferred tactics."""

    def __init__(
        self,
        *,
        root: str | Path,
        sdk_root: str | Path,
        epsilon: float,
        probe_runner: ProbeRunner = run_probe_episode,
    ) -> None:
        self.root = Path(root)
        self.sdk_root = Path(sdk_root)
        self.epsilon = float(epsilon)
        self.probe_runner = probe_runner
        self.reference_root = self.root / "reference"
        self.probe_root = self.root / "probes"
        self.manifest_path = self.reference_root / "manifest.json"

    def freeze_reference(self, evaluation: CandidateEvaluation) -> Path:
        if evaluation.status != "complete":
            raise ValueError("origin reference requires a complete evaluation")
        self.reference_root.mkdir(parents=True, exist_ok=True)
        episodes = []
        for index, match in enumerate(evaluation.matches):
            if match.get("status", "complete") != "complete":
                raise ValueError("origin reference contains an incomplete match")
            decisions = load_trace_decisions(str(match["trace"]))
            states = [decision["state"] for decision in decisions]
            path = self.reference_root / f"episode-{index:04d}.json"
            path.write_text(
                json.dumps(states, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            episodes.append(
                {
                    "path": str(path),
                    "opponent": match.get("opponent"),
                    "seed": match.get("seed"),
                    "decision_count": len(states),
                }
            )
        self.manifest_path.write_text(
            json.dumps(
                {"schema_version": "1.0", "episodes": episodes},
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return self.manifest_path

    def _episodes(self) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError("origin reference manifest has not been frozen")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return [
            (
                dict(item),
                json.loads(Path(item["path"]).read_text(encoding="utf-8")),
            )
            for item in manifest["episodes"]
        ]

    def measure(
        self,
        *,
        new_version: Version,
        old_version: Version,
        version_store: VersionStore,
        new_evaluation: CandidateEvaluation,
        old_evaluation: CandidateEvaluation,
    ) -> dict[str, Any]:
        if new_evaluation.status != "complete" or old_evaluation.status != "complete":
            raise ValueError("measurement requires two complete rollout evaluations")
        new_workspace = version_store.objects / new_version.content_hash
        old_workspace = version_store.objects / old_version.content_hash
        all_new = []
        all_old = []
        episode_metadata = []
        for episode_index, (metadata, states) in enumerate(self._episodes()):
            stem = (
                f"{old_version.version_id}-to-{new_version.version_id}"
                f"-episode-{episode_index:04d}"
            )
            old_probe = self.probe_runner(
                workspace=old_workspace,
                sdk_root=self.sdk_root,
                states=states,
                artifact_path=self.probe_root / f"{stem}-old.json",
            )
            new_probe = self.probe_runner(
                workspace=new_workspace,
                sdk_root=self.sdk_root,
                states=states,
                artifact_path=self.probe_root / f"{stem}-new.json",
            )
            for decision in old_probe:
                item = dict(decision)
                item["state_id"] = (
                    f"episode-{episode_index:04d}:"
                    f"{item['reference_index']:06d}:{item['state_id']}"
                )
                all_old.append(item)
            for decision in new_probe:
                item = dict(decision)
                item["state_id"] = (
                    f"episode-{episode_index:04d}:"
                    f"{item['reference_index']:06d}:{item['state_id']}"
                )
                all_new.append(item)
            episode_metadata.append(
                {
                    "episode_index": episode_index,
                    "opponent": metadata.get("opponent"),
                    "seed": metadata.get("seed"),
                    "decision_count": len(states),
                }
            )

        new_occupancy = self._occupancy_ids(new_evaluation)
        old_occupancy = self._occupancy_ids(old_evaluation)
        result = decision_trace_metrics(
            all_new,
            all_old,
            epsilon=self.epsilon,
            new_occupancy_state_ids=new_occupancy,
            old_occupancy_state_ids=old_occupancy,
        )
        cursor = 0
        episode_values = []
        for metadata in episode_metadata:
            count = metadata["decision_count"]
            values = result["local_policy_kl_trace"][cursor : cursor + count]
            cursor += count
            episode_values.append(
                {
                    **metadata,
                    "mean_local_policy_kl": sum(values) / len(values),
                    "trajectory_kl": sum(values),
                }
            )
        result["episode_local_policy_kl"] = episode_values
        result["reference_manifest"] = str(self.manifest_path)
        return result

    @staticmethod
    def _occupancy_ids(evaluation: CandidateEvaluation) -> list[str]:
        ids = []
        for match in evaluation.matches:
            if match.get("status", "complete") != "complete":
                continue
            ids.extend(
                decision["state_id"]
                for decision in load_trace_decisions(str(match["trace"]))
            )
        if not ids:
            raise ValueError("evaluation contains no Rollman occupancy states")
        return ids
