"""Run one two-player match through the official engine boundary."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import time

from .engine import OfficialGeneralsEngine
from .models import AgentProcessSpec, AssetLayout, MatchCase, MatchResult, TurnRecord
from .process import DecisionTimeout, ManagedAgentProcess, PlayerProcessError
from .protocol import ProtocolError


class GeneralsMatchRunner:
    def __init__(self, assets: AssetLayout, benchmark_id: str = "generals-hl-pilot-v1"):
        self.assets = assets
        self.benchmark_id = benchmark_id

    def run(
        self,
        case: MatchCase,
        players: tuple[AgentProcessSpec, AgentProcessSpec],
        artifact_dir: Path,
    ) -> MatchResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        replay_path = artifact_dir / "replay.jsonl"
        replay_path.write_text("", encoding="utf-8")
        engine = OfficialGeneralsEngine(
            self.assets.engine_root, case.seed, artifact_dir / "official-replay.jsonl"
        )
        started = time.monotonic()
        turns: list[TurnRecord] = []
        valid = False
        winner = None
        termination = "harness_error"
        error = None
        try:
            with ExitStack() as stack:
                managed = tuple(
                    stack.enter_context(
                        ManagedAgentProcess(
                            spec,
                            self.assets_config_limits,
                            artifact_dir / "players" / str(player),
                        )
                    )
                    for player, spec in enumerate(players)
                )
                for player in range(2):
                    managed[player].send_initial(engine.initial_observation(player))
                step = 0
                while True:
                    if time.monotonic() - started > self.assets_config_limits.match_timeout_s:
                        termination = "match_timeout"
                        error = "match controller timeout"
                        break
                    for player in (0, 1):
                        before = engine.normalized_state()
                        before["my_seat"] = player
                        before_id = engine.state_id()
                        try:
                            commands = managed[player].request_turn()
                        except DecisionTimeout as exc:
                            valid = True
                            winner = 1 - player
                            termination = "time_limit"
                            error = str(exc)
                            raise _MatchFinished
                        outcome = engine.apply_turn(player, commands)
                        after = engine.normalized_state()
                        after["my_seat"] = 1 - player
                        record = TurnRecord(
                            step=step,
                            round_number=int(before["round"]),
                            player=player,
                            state_id_before=before_id,
                            state_before=before,
                            commands=commands,
                            state_id_after=engine.state_id(),
                            state_after=after,
                        )
                        turns.append(record)
                        with replay_path.open("a", encoding="utf-8") as replay:
                            replay.write(
                                json.dumps(asdict(record), sort_keys=True, ensure_ascii=False) + "\n"
                            )
                        step += 1
                        if outcome.done:
                            valid = True
                            winner = outcome.winner
                            termination = outcome.termination_type or "normal"
                            raise _MatchFinished
                        managed[1 - player].send_peer_commands(commands)
        except _MatchFinished:
            pass
        except (PlayerProcessError, ProtocolError, OSError) as exc:
            valid = False
            termination = "process_error"
            error = str(exc)
        except Exception as exc:
            valid = False
            termination = "harness_error"
            error = f"{type(exc).__name__}: {exc}"

        result = MatchResult(
            case_id=case.case_id,
            valid=valid,
            winner=winner,
            termination_type=termination,
            seed=case.seed,
            evaluated_seat=case.evaluated_seat,
            turns=tuple(turns),
            elapsed_time_s=time.monotonic() - started,
            engine_hash=self.assets.engine_hash,
            error=error,
        )
        metadata = {
            "benchmark_id": self.benchmark_id,
            "engine_hash": self.assets.engine_hash,
            "case_id": case.case_id,
            "opponent_id": case.opponent_id,
            "opponent_tier": case.opponent_tier,
            "seed": case.seed,
            "evaluated_seat": case.evaluated_seat,
            "players": [item.agent_id for item in players],
            "winner": winner,
            "valid": valid,
            "termination_type": termination,
            "elapsed_time_s": result.elapsed_time_s,
            "error": error,
        }
        (artifact_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result

    @property
    def assets_config_limits(self):
        # AssetLayout is deliberately immutable and version-neutral. The pilot
        # limits are frozen here and validated when the manifest is loaded.
        from .assets import FROZEN_LIMITS

        return FROZEN_LIMITS


class _MatchFinished(Exception):
    pass
