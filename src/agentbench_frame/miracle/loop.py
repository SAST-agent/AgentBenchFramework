"""Miracle 的高层 LLM 策略迭代闭环。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .agent_bridge import AGENTS
from .ig import build_ig_curve, compare_agents_on_trace
from .llm_client import ChatCompletionsClient, LLMRequestError
from .loop_config import LoopConfig
from .match import run_match
from .protocol import decode_content
from .run_store import BudgetExceeded, BudgetLedger, MiracleRunStore
from .score import aggregate_score, build_score_curve
from .strategy_loader import (
    StrategyValidationError,
    load_candidate,
    save_source,
    validate_candidate,
)


def _version(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def build_messages(
    current_source: str,
    skills: dict[str, str],
    evidence: list[dict],
    previous_metrics: dict,
    budget: dict,
) -> list[dict]:
    system = (
        "You improve a deterministic Miracle game strategy. Return exactly one JSON object "
        'with nonempty string fields "analysis" and "strategy_code". strategy_code must be '
        "complete Python source defining CandidateAgent(MiracleAgent), including choose_cards(camp) "
        "and act(obs). Do not return a patch or shell commands."
    )
    user = "\n\n".join([
        "# Miracle Harness Skill\n" + skills["harness"],
        "# Miracle Replay Reader Skill\n" + skills["replay"],
        "# Current strategy\n" + current_source,
        "# Previous metrics\n" + json.dumps(previous_metrics, ensure_ascii=False, indent=2),
        "# Selected replay evidence\n" + json.dumps(evidence, ensure_ascii=False, indent=2),
        "# Cumulative budget\n" + json.dumps(budget, ensure_ascii=False, indent=2),
    ])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _read_evidence(episodes: list[dict], max_episodes: int, max_decisions: int) -> list[dict]:
    evidence = []
    remaining = max_decisions
    for episode in episodes[:max_episodes]:
        observations = []
        trace_path = episode.get("trace")
        if trace_path and Path(trace_path).is_file() and remaining > 0:
            with Path(trace_path).open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("kind") != "from_logic":
                        continue
                    msg = decode_content(row.get("payload", {}).get("content", []))
                    if isinstance(msg, dict) and "map" in msg and "round" in msg:
                        observations.append({"trace_seq": row.get("seq"), "observation": msg})
                        remaining -= 1
                        if remaining == 0:
                            break
        evidence.append({
            "episode_id": episode["episode_id"],
            "candidate_camp": episode["candidate_camp"],
            "scores": episode.get("scores"),
            "winner": episode.get("winner"),
            "terminated_by": episode.get("terminated_by"),
            "errors": episode.get("errors", []),
            "observations": observations,
        })
        if remaining == 0:
            break
    return evidence


def _evaluate(
    *,
    source_path: Path,
    iteration: int,
    config: LoopConfig,
    store: MiracleRunStore,
    ledger: BudgetLedger,
    match_runner,
) -> tuple[dict, list[dict]]:
    episode_rows = []
    camps = {}
    episodes_dir = store.iteration_dir(iteration) / "episodes"
    for seed in config.evaluation.seeds:
        for seat in config.evaluation.seats:
            ledger.charge_rollout()
            episode_id = f"iter{iteration:04d}-seed{seed}-camp{seat}"
            candidate = load_candidate(source_path, f"candidate_{iteration}_{seed}_{seat}")
            opponent = AGENTS[config.opponent]()
            agents = (candidate, opponent) if seat == 0 else (opponent, candidate)
            store.write_event(
                "battle_started", iteration=iteration, episode_id=episode_id,
                seed=seed, candidate_camp=seat, opponent=config.opponent,
            )
            result = match_runner(
                agents[0], agents[1], replay_dir=episodes_dir,
                seed=seed, tag=episode_id,
            )
            ledger.charge_battle_time(result.duration)
            row = {
                "episode_id": episode_id,
                "iteration": iteration,
                "seed": seed,
                "candidate_camp": seat,
                "winner": result.winner,
                "scores": list(result.scores),
                "rounds": result.rounds,
                "terminated_by": result.terminated_by,
                "errors": list(result.errors),
                "duration": result.duration,
                "replay": result.replay_path,
                "trace": result.trace_path,
            }
            store.write_json_atomic(episodes_dir / f"{episode_id}.json", row)
            store.write_event("battle_finished", **row)
            episode_rows.append(row)
            camps[episode_id] = seat
    return aggregate_score(episode_rows, camps), episode_rows


def _skills(store: MiracleRunStore) -> dict[str, str]:
    return {
        "harness": (store.run_dir / "skills/miracle-harness.SKILL.md").read_text(encoding="utf-8"),
        "replay": (store.run_dir / "skills/miracle-replay-reader.SKILL.md").read_text(encoding="utf-8"),
    }


def run_loop(
    config: LoopConfig,
    *,
    client=None,
    match_runner=run_match,
    data_dir=None,
    run_id: str | None = None,
) -> Path:
    store = MiracleRunStore.create(config, data_dir=data_dir, run_id=run_id)
    ledger = BudgetLedger(config.budget)
    client = client or ChatCompletionsClient(config.llm)
    iterations = []
    ig_episodes = []
    failure_counts = Counter()
    total_episodes = 0
    total_steps = 0

    initial_source = config.initial_strategy.read_text(encoding="utf-8")
    baseline_dir = store.iteration_dir(0)
    baseline_path = baseline_dir / "strategy.py"
    save_source(baseline_path, initial_source)
    baseline_agent = load_candidate(baseline_path, "candidate_baseline_validation")
    validate_candidate(baseline_agent)
    store.write_event("iteration_started", iteration=0, status="baseline")
    baseline_score, baseline_episodes = _evaluate(
        source_path=baseline_path, iteration=0, config=config, store=store,
        ledger=ledger, match_runner=match_runner,
    )
    total_episodes += len(baseline_episodes)
    total_steps += sum(row["rounds"] for row in baseline_episodes)
    accepted_source = initial_source
    accepted_path = baseline_path
    accepted_version = _version(initial_source)
    accepted_episodes = baseline_episodes
    baseline_record = {
        "iteration": 0, "version": accepted_version, "status": "baseline",
        "score": baseline_score["mean_score"], "win_rate": baseline_score["win_rate"],
        "completion_rate": baseline_score["completion_rate"],
        "score_detail": baseline_score, "budget": ledger.snapshot(),
    }
    iterations.append(baseline_record)
    store.write_json_atomic(baseline_dir / "iteration.json", baseline_record)
    store.write_event("iteration_finished", iteration=0, status="baseline")

    for index in range(1, config.budget.max_iterations + 1):
        iteration_dir = store.iteration_dir(index)
        store.write_event("iteration_started", iteration=index, status="pending")
        evidence = _read_evidence(
            accepted_episodes,
            config.budget.max_episode_reads - ledger.episode_reads,
            config.budget.max_decision_reads - ledger.decision_reads,
        )
        decision_reads = sum(len(item["observations"]) for item in evidence)
        try:
            ledger.charge_read(len(evidence), decision_reads)
            messages = build_messages(
                accepted_source, _skills(store), evidence, iterations[-1], ledger.snapshot(),
            )
            store.write_json_atomic(iteration_dir / "llm_request.json", {
                "messages": messages,
                "model": config.llm.model,
                "base_url": config.llm.base_url,
                "temperature": config.llm.temperature,
                "max_tokens": config.llm.max_tokens,
                "reasoning_effort": config.llm.reasoning_effort,
                "stream": config.llm.stream,
                "max_context_tokens": config.llm.max_context_tokens,
            })
            store.write_event("llm_request_started", iteration=index, model=config.llm.model)
            proposal = client.propose_strategy(messages)
            store.write_json_atomic(iteration_dir / "llm_request.json", {
                "messages": messages, "request_body": proposal.request_body,
            })
            store.write_json_atomic(iteration_dir / "llm_response.json", {
                "raw_response": proposal.raw_response,
                "analysis": proposal.analysis,
                "usage": proposal.usage,
                "latency_seconds": proposal.latency_seconds,
                "normalized_fence": proposal.normalized_fence,
            })
            ledger.charge_api_time(proposal.latency_seconds)
            ledger.charge_usage(proposal.usage)
            ledger.charge_context(
                proposal.usage.get("total_tokens", 0),
                config.llm.max_context_tokens,
            )
            store.write_event(
                "llm_request_finished", iteration=index,
                latency_seconds=proposal.latency_seconds, usage=proposal.usage,
            )
            candidate_path = iteration_dir / "candidate.py"
            save_source(candidate_path, proposal.strategy_code)
            candidate = load_candidate(candidate_path, f"candidate_validation_{index}")
            validate_candidate(candidate)
        except (LLMRequestError, StrategyValidationError, BudgetExceeded) as exc:
            stage = getattr(exc, "stage", getattr(exc, "dimension", "unknown"))
            raw = getattr(exc, "raw_response", None)
            if isinstance(exc, LLMRequestError):
                ledger.charge_api_time(exc.latency_seconds)
                ledger.charge_usage(exc.usage)
                try:
                    ledger.charge_context(
                        exc.usage.get("total_tokens", 0),
                        config.llm.max_context_tokens,
                    )
                except BudgetExceeded as context_exc:
                    exc = context_exc
                    stage = context_exc.dimension
            failure_counts[stage] += 1
            if raw is not None and not (iteration_dir / "llm_response.json").exists():
                store.write_json_atomic(iteration_dir / "llm_response.json", {
                    "raw_response": raw, "error": str(exc), "stage": stage,
                })
            failed = {
                "iteration": index, "version": accepted_version, "status": "failed",
                "failure_stage": stage, "failure_reason": str(exc), "score": None,
                "win_rate": None, "completion_rate": 0.0, "budget": ledger.snapshot(),
            }
            iterations.append(failed)
            store.write_json_atomic(iteration_dir / "iteration.json", failed)
            store.write_event("iteration_failed", iteration=index, stage=stage, reason=str(exc))
            continue

        strategy_path = iteration_dir / "strategy.py"
        save_source(strategy_path, proposal.strategy_code)
        new_version = _version(proposal.strategy_code)
        evolved_score, latest_episodes = _evaluate(
            source_path=strategy_path, iteration=index, config=config, store=store,
            ledger=ledger, match_runner=match_runner,
        )
        total_episodes += len(latest_episodes)
        total_steps += sum(row["rounds"] for row in latest_episodes)
        for episode in latest_episodes:
            old_agent = load_candidate(accepted_path, f"ig_old_{index}_{episode['episode_id']}")
            new_agent = load_candidate(strategy_path, f"ig_new_{index}_{episode['episode_id']}")
            ig_episode = compare_agents_on_trace(
                episode["trace"], old_agent, new_agent,
                camp=episode["candidate_camp"], iteration=index,
                old_version=accepted_version, new_version=new_version,
            )
            ig_episodes.append(ig_episode)
            store.write_json_atomic(
                iteration_dir / "ig" / f"{ig_episode['episode_id']}.json", ig_episode,
            )
        record = {
            "iteration": index, "version": new_version, "status": "accepted",
            "score": evolved_score["mean_score"], "win_rate": evolved_score["win_rate"],
            "completion_rate": evolved_score["completion_rate"],
            "score_detail": evolved_score, "analysis": proposal.analysis,
            "budget": ledger.snapshot(),
        }
        iterations.append(record)
        store.write_json_atomic(iteration_dir / "iteration.json", record)
        store.write_event("iteration_finished", iteration=index, status="accepted")
        accepted_source = proposal.strategy_code
        accepted_path = strategy_path
        accepted_version = new_version
        accepted_episodes = latest_episodes

    score_curve = build_score_curve(iterations)
    versions = {int(row["iteration"]): str(row["version"]) for row in iterations}
    ig_curve = build_ig_curve(ig_episodes, versions=versions)
    store.write_json_atomic(store.run_dir / "score_curve.json", score_curve)
    store.write_json_atomic(store.run_dir / "ig_curve.json", ig_curve)
    points = score_curve["points"]
    measured = [point for point in points if point["evo"] is not None]
    final = measured[-1] if measured else {"evo": None, "gain": None, "win_rate": None}
    best = max(measured, key=lambda point: point["evo"]) if measured else None
    budget = ledger.snapshot()
    summary = {
        "status": "complete",
        "total_iterations": config.budget.max_iterations + 1,
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "win_rate": final.get("win_rate"),
        "raw": points[0]["raw"] if points else None,
        "final_evo": final.get("evo"),
        "final_gain": final.get("gain"),
        "best_evo": best.get("evo") if best else None,
        "best_iteration": best.get("iteration") if best else None,
        "failure_counts": dict(failure_counts),
        "score_history": points,
        "ig_history": ig_curve["points"],
        "auc": score_curve["auc"],
        **budget,
    }
    store.finish(summary)
    return store.run_dir
