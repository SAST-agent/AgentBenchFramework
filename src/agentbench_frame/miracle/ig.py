"""确定性 Miracle 策略的严格 KL 状态记录与 iteration 曲线。"""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path

from .decision_space import Action, action_mask
from .protocol import decode_content


def _as_action(value) -> Action | None:
    if not isinstance(value, dict):
        return None
    action_type = value.get("operation_type")
    params = value.get("operation_parameters")
    if not isinstance(action_type, str) or not isinstance(params, dict):
        return None
    return Action(action_type, params)


def _missing_row(obs_id: str, reason: str, old_action=None, new_action=None) -> dict:
    return {
        "obs_id": obs_id,
        "old_action": old_action.signature() if isinstance(old_action, Action) else None,
        "new_action": new_action.signature() if isinstance(new_action, Action) else None,
        "status": "missing",
        "kl": None,
        "missing_reason": reason,
    }


def observations_from_trace(path, *, camp: int):
    """按 trace 顺序读取指定阵营实际收到的官方 observation。"""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") != "from_logic":
                continue
            payload = row.get("payload", {})
            msg = decode_content(payload.get("content", []))
            if not isinstance(msg, dict) or "round" not in msg or "map" not in msg:
                continue
            if int(msg.get("camp", -1)) != camp:
                continue
            yield int(row.get("seq", line_no)), msg


def compare_agents_on_trace(
    trace_path,
    old_agent,
    new_agent,
    *,
    camp: int,
    iteration: int,
    old_version: str,
    new_version: str,
) -> dict:
    """在同一条真实 observation 序列上比较两个确定性 Agent。"""
    rows = []
    for trace_seq, obs in observations_from_trace(trace_path, camp=camp):
        canonical = json.dumps(obs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        obs_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        try:
            old_action = _as_action(old_agent.act(obs))
        except Exception:
            old_action = None
        try:
            new_action = _as_action(new_agent.act(obs))
        except Exception:
            new_action = None
        if old_action is None:
            row = _missing_row(obs_id, "old_action_invalid", new_action=new_action)
        elif new_action is None:
            row = _missing_row(obs_id, "new_action_invalid", old_action=old_action)
        else:
            row = compare_deterministic(obs_id, old_action, new_action, action_mask(obs, camp))
        row["trace_seq"] = trace_seq
        row["round"] = int(obs.get("round", 0))
        rows.append(row)

    episode_id = f"{Path(trace_path).name.removesuffix('.trace.jsonl')}-camp{camp}"
    result = aggregate_episode(rows, episode_id=episode_id, iteration=iteration)
    result.update({
        "source_trace": str(Path(trace_path).resolve()),
        "camp": camp,
        "old_version": old_version,
        "new_version": new_version,
    })
    return result


def compare_deterministic(
    obs_id: str,
    old_action: Action,
    new_action: Action,
    support,
) -> dict:
    support_signatures = {
        action.signature() if isinstance(action, Action) else str(action) for action in support
    }
    old_sig = old_action.signature()
    new_sig = new_action.signature()
    base = {"obs_id": obs_id, "old_action": old_sig, "new_action": new_sig}
    if old_sig not in support_signatures or new_sig not in support_signatures:
        return {**base, "status": "missing", "kl": None,
                "missing_reason": "action_outside_support"}
    if old_sig == new_sig:
        return {**base, "status": "unchanged", "kl": 0.0, "missing_reason": None}
    return {**base, "status": "infinite", "kl": None,
            "missing_reason": "support_expansion"}


def aggregate_episode(rows: list[dict], *, episode_id: str, iteration: int) -> dict:
    counts = Counter(row["status"] for row in rows)
    total = len(rows)
    finite = [row["kl"] for row in rows if row.get("kl") is not None]
    return {
        "episode_id": episode_id,
        "iteration": iteration,
        "n_decisions": total,
        "finite_kl_count": len(finite),
        "finite_kl_mean": round(sum(finite) / len(finite), 6) if finite else None,
        "unchanged_ratio": round(counts["unchanged"] / total, 6) if total else 0.0,
        "infinite_ratio": round(counts["infinite"] / total, 6) if total else 0.0,
        "missing_ratio": round(counts["missing"] / total, 6) if total else 0.0,
        "counts": {name: counts[name] for name in ("unchanged", "infinite", "missing")},
        "decisions": rows,
    }


def build_ig_curve(episodes: list[dict], *, versions: dict[int, str]) -> dict:
    points = []
    for iteration in sorted(versions):
        if iteration == 0:
            points.append({
                "iteration": 0, "version": versions[iteration], "status": "baseline",
                "finite_kl_mean": None, "unchanged_ratio": None,
                "infinite_ratio": None, "missing_ratio": None,
            })
            continue
        current = [ep for ep in episodes if ep["iteration"] == iteration]
        if not current:
            points.append({
                "iteration": iteration, "version": versions[iteration], "status": "missing",
                "finite_kl_mean": None, "unchanged_ratio": None,
                "infinite_ratio": None, "missing_ratio": 1.0,
            })
            continue
        weights = [ep["n_decisions"] for ep in current]
        total = sum(weights)
        finite_weight = sum(ep.get("finite_kl_count", 0) for ep in current)
        points.append({
            "iteration": iteration,
            "version": versions[iteration],
            "status": "measured",
            "finite_kl_mean": round(sum(
                ep["finite_kl_mean"] * ep.get("finite_kl_count", 0)
                for ep in current if ep["finite_kl_mean"] is not None
            ) / finite_weight, 6) if finite_weight else None,
            "unchanged_ratio": round(sum(ep["unchanged_ratio"] * w for ep, w in zip(current, weights)) / total, 6) if total else 0.0,
            "infinite_ratio": round(sum(ep["infinite_ratio"] * w for ep, w in zip(current, weights)) / total, 6) if total else 0.0,
            "missing_ratio": round(sum(ep["missing_ratio"] * w for ep, w in zip(current, weights)) / total, 6) if total else 0.0,
        })
    return {
        "metric": "strict_kl_status",
        "note": "确定性策略：相同动作 KL=0；不同动作严格 KL 发散；不以代理指标冒充有限 KL。",
        "points": points,
    }


def save_ig_curve(curve: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(curve, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_episode_ig(episode: dict, output_dir) -> Path:
    path = Path(output_dir) / f"iteration-{int(episode['iteration']):04d}" / f"{episode['episode_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episode, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_episode_ig(output_dir) -> list[dict]:
    episodes = []
    for path in sorted(Path(output_dir).glob("iteration-*/*.json")):
        try:
            episode = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(episode, dict) and "iteration" in episode and "n_decisions" in episode:
            episodes.append(episode)
    return episodes


def versions_from_episodes(episodes: list[dict]) -> dict[int, str]:
    versions = {}
    for episode in sorted(episodes, key=lambda item: int(item["iteration"])):
        iteration = int(episode["iteration"])
        versions.setdefault(iteration, episode.get("new_version", f"iteration-{iteration}"))
        if iteration > 0:
            versions.setdefault(iteration - 1, episode.get("old_version", f"iteration-{iteration - 1}"))
    return versions
