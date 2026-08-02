"""Replay-driven DOTO C++ policy iteration loop."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .build import build_candidate
from .ig import build_ig_curve, compare_policies_on_trace
from .llm_client import ChatCompletionsClient, LLMRequestError
from .loop_config import LoopConfig
from .match import run_match
from .run_store import BudgetExceeded, BudgetLedger, DotoRunStore
from .score import aggregate_episodes, build_score_curve


def _version(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def build_messages(current_source: str, skills: dict[str, str], sdk_reference: str,
                   evidence: list[dict], previous_metrics: dict, budget: dict) -> list[dict]:
    system = (
        "Improve a deterministic DOTO player. Return exactly one JSON object with nonempty "
        'string fields "analysis" and "player_ai_cpp". player_ai_cpp must be the complete '
        "replacement playerAI.cpp defining void playerAI(). Do not return a patch or commands."
    )
    user = "\n\n".join([
        "# DOTO Harness Skill\n" + skills["harness"],
        "# DOTO Replay Reader Skill\n" + skills["replay"],
        "# Fixed SDK Reference\n" + sdk_reference,
        "# Current playerAI.cpp\n" + current_source,
        "# Previous metrics\n" + json.dumps(previous_metrics, ensure_ascii=False, indent=2),
        "# Selected replay evidence\n" + json.dumps(evidence, ensure_ascii=False, indent=2),
        "# Cumulative budget\n" + json.dumps(budget, ensure_ascii=False, indent=2),
    ])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def read_evidence(episodes: list[dict], max_episodes: int, max_frames: int) -> list[dict]:
    selected, remaining = [], max(0, max_frames)
    for episode in episodes[:max(0, max_episodes)]:
        frames, pending = [], {}
        trace = episode.get("trace") or episode.get("trace_path")
        if trace and Path(trace).is_file() and remaining:
            for line in Path(trace).read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                frame = row.get("frame")
                if row.get("kind") == "observation" and isinstance(frame, int) and frame > 0:
                    pending[(row.get("faction"), frame)] = {"frame": frame, "faction": row.get("faction"),
                                                             "observation": row.get("payload")}
                elif row.get("kind") == "action" and (row.get("faction"), frame) in pending:
                    item = pending.pop((row.get("faction"), frame))
                    item["action"] = row.get("payload")
                    frames.append(item)
                    remaining -= 1
                    if remaining == 0:
                        break
        selected.append({"episode_id": episode.get("episode_id"),
                         "candidate_faction": episode.get("candidate_faction"),
                         "scores": episode.get("scores"), "winner": episode.get("winner"),
                         "terminated_by": episode.get("terminated_by"),
                         "errors": episode.get("errors", []), "frames": frames})
        if remaining == 0:
            break
    return selected


def _skills(store: DotoRunStore) -> dict[str, str]:
    return {name: (store.run_dir / f"skills/doto-{filename}.SKILL.md").read_text(encoding="utf-8")
            for name, filename in (("harness", "harness"), ("replay", "replay-reader"))}


def _sdk_reference() -> str:
    return (
        "The harness supplies the fixed official SDK. Maintain only a complete playerAI.cpp. "
        "Implement void playerAI(); use the declarations in playerAI.h and do not add another main()."
    )


def _evaluate(executable: Path, iteration: int, config: LoopConfig, store: DotoRunStore,
              ledger: BudgetLedger, opponents: list[tuple[str, Path]], match_runner) -> tuple[dict, list[dict]]:
    rows, factions = [], {}
    episodes_dir = store.iteration_dir(iteration) / "episodes"
    for opponent_name, opponent in opponents:
        for seed in config.evaluation.seeds:
            for seat in config.evaluation.seats:
                ledger.charge_rollout()
                episode_id = f"iter{iteration:04d}-{opponent_name}-seed{seed}-seat{seat}"
                agents = (executable, opponent) if seat == 0 else (opponent, executable)
                store.write_event("battle_started", iteration=iteration, episode_id=episode_id,
                                  candidate_faction=seat, opponent=opponent_name, seed=seed)
                result = match_runner(agents[0], agents[1], seed=seed, output_dir=episodes_dir,
                                      tag=episode_id)
                ledger.charge_battle_time(result.duration)
                row = {"episode_id": episode_id, "iteration": iteration, "seed": seed,
                       "opponent": opponent_name, "candidate_faction": seat,
                       "winner": result.winner, "scores": list(result.scores),
                       "frames": result.frames, "terminated_by": result.terminated_by,
                       "errors": list(result.errors), "duration": result.duration,
                       "replay": str(result.replay_path), "trace": str(result.trace_path),
                       "metadata": getattr(result, "metadata", {})}
                store.write_json_atomic(episodes_dir / f"{episode_id}.json", row)
                store.write_event("battle_finished", **row)
                rows.append(row)
                factions[episode_id] = seat
    return aggregate_episodes(rows, factions), rows


def run_loop(config: LoopConfig, *, client=None, builder=build_candidate,
             match_runner=run_match, ig_runner=compare_policies_on_trace,
             data_dir=None, run_id: str | None = None) -> Path:
    store = DotoRunStore.create(config, data_dir=data_dir, run_id=run_id)
    ledger, client = BudgetLedger(config.budget), client or ChatCompletionsClient(config.llm)
    iterations, ig_rows, failures = [], [], Counter()
    total_episodes = total_steps = 0
    opponents = []
    for opponent in config.opponents:
        if opponent.executable:
            opponents.append((opponent.name, opponent.executable))
        else:
            ledger.charge_build()
            result = builder(opponent.source, store.run_dir / "opponents" / opponent.name)
            ledger.charge_compile_time(result.seconds)
            if result.exit_code or result.executable is None:
                raise RuntimeError(f"opponent build failed: {opponent.name}")
            opponents.append((opponent.name, result.executable))

    source = config.initial_player_ai.read_text(encoding="utf-8")
    baseline_dir = store.iteration_dir(0)
    baseline_source = baseline_dir / "playerAI.cpp"
    store.write_text_atomic(baseline_source, source)
    ledger.charge_build()
    built = builder(baseline_source, baseline_dir / "build")
    ledger.charge_compile_time(built.seconds)
    if built.exit_code or built.executable is None:
        raise RuntimeError("baseline build failed")
    accepted_source, accepted_executable, accepted_version = source, built.executable, _version(source)
    score, accepted_episodes = _evaluate(accepted_executable, 0, config, store, ledger, opponents, match_runner)
    total_episodes += len(accepted_episodes)
    total_steps += sum(row["frames"] for row in accepted_episodes)
    baseline = {"iteration": 0, "version": accepted_version, "status": "baseline",
                "score": score["score"], "win_rate": score["win_rate"],
                "completion_rate": score["completion_rate"], "score_detail": score,
                "budget": ledger.snapshot()}
    iterations.append(baseline)
    store.write_json_atomic(baseline_dir / "iteration.json", baseline)

    for index in range(1, config.budget.max_iterations + 1):
        directory = store.iteration_dir(index)
        try:
            ledger.charge_iteration()
            evidence = read_evidence(accepted_episodes,
                                     config.budget.max_episode_reads - ledger.episode_reads,
                                     config.budget.max_frame_reads - ledger.frame_reads)
            ledger.charge_read(len(evidence), sum(len(item["frames"]) for item in evidence))
            messages = build_messages(accepted_source, _skills(store), _sdk_reference(), evidence,
                                      iterations[-1], ledger.snapshot())
            store.write_json_atomic(directory / "llm_request.json", {"messages": messages,
                                    "model": config.llm.model, "base_url": config.llm.base_url,
                                    "stream": config.llm.stream,
                                    "max_context_tokens": config.llm.max_context_tokens})
            proposal = client.propose(messages)
            ledger.charge_api_time(proposal.latency_seconds)
            ledger.charge_usage(proposal.usage)
            ledger.charge_context(proposal.usage.get("prompt_tokens", 0), config.llm.max_context_tokens)
            store.write_json_atomic(directory / "llm_request.json",
                                    {"messages": messages, "request_body": proposal.request_body})
            store.write_json_atomic(directory / "llm_response.json",
                                    {"analysis": proposal.analysis, "usage": proposal.usage,
                                     "latency_seconds": proposal.latency_seconds,
                                     "raw_response": proposal.raw_response})
            candidate = directory / "candidate.playerAI.cpp"
            store.write_text_atomic(candidate, proposal.player_ai_cpp)
            ledger.charge_build()
            built = builder(candidate, directory / "build")
            ledger.charge_compile_time(built.seconds)
            if built.exit_code or built.executable is None:
                raise RuntimeError("candidate build failed")
            new_version = _version(proposal.player_ai_cpp)
            score, episodes = _evaluate(built.executable, index, config, store, ledger, opponents, match_runner)
            total_episodes += len(episodes)
            total_steps += sum(row["frames"] for row in episodes)
            for episode in episodes:
                ig = ig_runner(Path(episode["trace"]), accepted_executable, built.executable,
                               faction=episode["candidate_faction"], iteration=index,
                               old_version=accepted_version, new_version=new_version)
                ig_rows.append(ig)
                store.write_json_atomic(directory / "ig" / f"{ig['episode_id']}.json", ig)
            store.write_text_atomic(directory / "playerAI.cpp", proposal.player_ai_cpp)
            record = {"iteration": index, "version": new_version, "status": "accepted",
                      "score": score["score"], "win_rate": score["win_rate"],
                      "completion_rate": score["completion_rate"], "score_detail": score,
                      "analysis": proposal.analysis, "budget": ledger.snapshot()}
            accepted_source, accepted_executable, accepted_version = proposal.player_ai_cpp, built.executable, new_version
            accepted_episodes = episodes
        except Exception as exc:
            stage = getattr(exc, "stage", getattr(exc, "dimension", "iteration"))
            if isinstance(exc, LLMRequestError):
                ledger.charge_api_time(exc.latency_seconds)
                try:
                    ledger.charge_usage(exc.usage)
                    ledger.charge_context(exc.usage.get("prompt_tokens", 0),
                                          config.llm.max_context_tokens)
                except BudgetExceeded as budget_exc:
                    stage = budget_exc.dimension
            failures[stage] += 1
            record = {"iteration": index, "version": accepted_version, "status": "failed",
                      "failure_stage": stage, "failure_reason": str(exc), "score": None,
                      "win_rate": None, "completion_rate": 0.0, "budget": ledger.snapshot()}
            if isinstance(exc, LLMRequestError) and exc.raw_response is not None:
                store.write_json_atomic(directory / "llm_response.json",
                                        {"error": str(exc), "stage": stage,
                                         "raw_response": exc.raw_response, "usage": exc.usage})
        iterations.append(record)
        store.write_json_atomic(directory / "iteration.json", record)
        store.write_event("iteration_finished", iteration=index, status=record["status"])

    score_curve = build_score_curve(iterations)
    ig_curve = build_ig_curve(ig_rows, {int(row["iteration"]): str(row["version"]) for row in iterations})
    store.write_json_atomic(store.run_dir / "score_curve.json", score_curve)
    store.write_json_atomic(store.run_dir / "ig_curve.json", ig_curve)
    measured = [point for point in score_curve["points"] if point["evo"] is not None]
    final, best = (measured[-1] if measured else {}), (max(measured, key=lambda x: x["evo"]) if measured else {})
    store.finish({"status": "complete", "total_iterations": len(iterations),
                  "total_episodes": total_episodes, "total_steps": total_steps,
                  "raw": score_curve["points"][0]["raw"] if score_curve["points"] else None,
                  "final_evo": final.get("evo"), "final_gain": final.get("gain"),
                  "best_evo": best.get("evo"), "best_iteration": best.get("iteration"),
                  "failure_counts": dict(failures), "auc": score_curve["auc"], **ledger.snapshot()})
    return store.run_dir
