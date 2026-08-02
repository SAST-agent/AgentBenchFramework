"""评测机（host）：驱动官方 logic 子进程并把 MiracleAgent 桥接成协议消息（自己实现）。

消息流（与官方 ``main.py`` 一致）：
- 启动后先发 init：``{"player_list": [1, 1, 2], "replay": <路径>}``
- 循环读 logic 帧（``target, payload = read_logic_frame``）：
  - ``payload["state"] == 0``  → send_init 初始化消息，忽略
  - ``payload["state"] == -1`` → 终局（``end_info`` 为 ``'{"0": s0, "1": s1}'``）
  - 其余为选卡/回合消息：``content[0]`` 解析出内层 dict
    - 无 ``round`` 字段 → 选卡（含 ``camp``），调 ``choose_cards``
    - 有 ``round`` 字段 → 回合局面，调 ``act``
- 每次决策写回一条 ``{"player", "round", "operation_type", "operation_parameters"}``；
  决策超时则向 logic 发官方约定的异常帧 ``{"player": -1, "content": json.dumps({...})}``。

全部收发逐帧记入 trace（jsonl），保证"日志可追溯"。
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from .agent_bridge import MiracleAgent
from .protocol import (
    ProtocolError,
    decode_content,
    encode_to_logic,
    read_logic_frame,
)

__all__ = ["FrameRecord", "MatchResult", "MiracleHost"]

#: 官方 AI_TIME（非媒体玩家的决策时限，秒）
DEFAULT_DECISION_TIMEOUT = 3.0
#: logic 长时间无输出的兜底超时（秒）
DEFAULT_IDLE_TIMEOUT = 8.0


@dataclass
class FrameRecord:
    """一条可追溯的收发记录。"""

    seq: int
    ts: float
    kind: str  # "from_logic" | "to_logic" | "timeout" | "init"
    state: Optional[int] = None
    player: Optional[int] = None
    summary: str = ""
    payload: dict = field(default_factory=dict)


@dataclass
class MatchResult:
    """一场对局的结果与产物。"""

    winner: int
    scores: tuple  # (s0, s1)，官方 score（平局时后手 +1，winner 记为 1）
    rounds: int
    replay_path: str
    trace_path: str
    duration: float
    terminated_by: str  # normal | timeout | idle_timeout | logic_exit | host_error
    errors: list = field(default_factory=list)
    frames: int = 0
    stderr_tail: str = ""


class MiracleHost:
    """对局驱动：持有 logic 子进程与两个 Agent，跑完整场对局。"""

    def __init__(
        self,
        proc,
        agents: tuple,
        *,
        decision_timeout: float = DEFAULT_DECISION_TIMEOUT,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        trace_path: Optional[str] = None,
    ) -> None:
        self.proc = proc
        self.agents: tuple = agents
        self.decision_timeout = decision_timeout
        self.idle_timeout = idle_timeout
        self.trace_path = trace_path
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="miracle-agent")
        self._trace: list = []
        self._trace_file = None
        self._last_was_timeout = False
        self._rounds_seen = 0
        self._last_obs_key = None  # (listen, obs 指纹)
        self._repeat_count = 0

    # ---- trace ----
    def _open_trace(self) -> None:
        if self.trace_path:
            self._trace_file = open(self.trace_path, "w", encoding="utf-8")

    def _record(self, rec: FrameRecord) -> None:
        self._trace.append(rec)
        if self._trace_file:
            row = {
                "seq": rec.seq,
                "ts": round(rec.ts, 4),
                "kind": rec.kind,
                "state": rec.state,
                "player": rec.player,
                "summary": rec.summary,
                "payload": rec.payload,
            }
            self._trace_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _close_trace(self) -> None:
        if self._trace_file:
            self._trace_file.close()
            self._trace_file = None

    # ---- 消息收发 ----
    def _send(self, payload: dict, kind: str = "to_logic") -> None:
        self.proc.stdin.write(encode_to_logic(payload))
        self.proc.stdin.flush()
        self._record(
            FrameRecord(
                seq=len(self._trace),
                ts=time.time(),
                kind=kind,
                state=payload.get("state"),
                player=payload.get("player"),
                summary=json.dumps(payload, ensure_ascii=False)[:200],
                payload=payload,
            )
        )

    def _send_action(self, player: int, inner: dict) -> None:
        """按官方格式发一条玩家操作：``{"player": p, "content": json字符串}``。

        注意内层操作 dict 也必须含 ``player``（官方 ``Parser.to_object`` 要求）。
        """
        inner = {"player": player, **inner}
        self._send({"player": player, "content": json.dumps(inner)})

    def _decide(self, fn, *args):
        """带超时地调用 Agent 决策；超时/异常返回 None。"""
        future = self._executor.submit(fn, *args)
        try:
            return future.result(timeout=self.decision_timeout)
        except TimeoutError:
            return None
        except Exception as exc:  # Agent 内部异常
            return exc

    # ---- 主循环 ----
    def run(self, replay_path: str) -> MatchResult:
        t0 = time.time()
        self._open_trace()
        errors: list = []
        terminated_by = "host_error"
        winner = 1
        scores = (0, 0)
        rounds = 0

        try:
            # 1) init
            init_payload = {"player_list": [1, 1, 2], "replay": replay_path}
            self._send(init_payload, kind="init")
            self._last_was_timeout = False

            # 2) 主循环
            while True:
                try:
                    target, payload = read_logic_frame(
                        self.proc.stdout, self.idle_timeout, label="logic"
                    )
                except TimeoutError:
                    errors.append("idle timeout: no frame from logic")
                    terminated_by = "idle_timeout"
                    break
                except EOFError:
                    errors.append("logic exited unexpectedly")
                    terminated_by = "logic_exit"
                    break

                state = payload.get("state")
                self._record(
                    FrameRecord(
                        seq=len(self._trace),
                        ts=time.time(),
                        kind="from_logic",
                        state=state,
                        player=None,
                        summary=json.dumps(payload, ensure_ascii=False)[:200],
                        payload=payload,
                    )
                )

                # 终局
                if state == -1:
                    terminated_by = "timeout" if self._last_was_timeout else "normal"
                    end_info = payload.get("end_info", "{}")
                    try:
                        d = json.loads(end_info)
                        scores = (int(d.get("0", 0)), int(d.get("1", 0)))
                    except (ValueError, TypeError):
                        errors.append(f"bad end_info: {end_info!r}")
                    winner = 0 if scores[0] > scores[1] else 1
                    rounds = self._rounds_seen
                    break

                # state==0 的初始化消息：忽略
                if state == 0:
                    continue

                # 选卡 / 回合消息
                content = payload.get("content", [])
                msg = decode_content(content)
                if msg is None:
                    errors.append(f"undecodable content frame: {payload!r}")
                    terminated_by = "host_error"
                    break

                listen = payload.get("listen", [0])[0]
                if listen not in (0, 1):
                    errors.append(f"unexpected listen={listen!r}")
                    terminated_by = "host_error"
                    break

                if "round" not in msg:  # 选卡消息：{"camp": c}
                    camp = int(msg.get("camp", listen))
                    self._last_was_timeout = False
                    result = self._decide(self.agents[camp].choose_cards, camp)
                    if result is None:
                        self._send_timeout(camp, state)
                        continue
                    if isinstance(result, Exception):
                        errors.append(f"agent{camp}.choose_cards raised: {result!r}")
                        self._send_timeout(camp, state)
                        continue
                    action_inner = {
                        "round": 0,
                        "operation_type": "init",
                        "operation_parameters": result,
                    }
                    self._send_action(camp, action_inner)
                else:  # 回合消息
                    round_no = int(msg.get("round", 0))
                    self._rounds_seen = max(self._rounds_seen, round_no)
                    camp = int(msg.get("camp", listen))
                    # 防卡死：同一玩家收到完全相同的 obs ≥3 次 → 按官方超时判负
                    obs_key = (listen, hashlib.md5(content[0].encode()).hexdigest())
                    if obs_key == self._last_obs_key:
                        self._repeat_count += 1
                    else:
                        self._last_obs_key = obs_key
                        self._repeat_count = 1
                    if self._repeat_count >= 3:
                        errors.append(
                            f"agent{listen} stuck (same obs x{self._repeat_count}) at round {round_no}"
                        )
                        self._send_timeout(listen, state)
                        continue
                    self._last_was_timeout = False
                    result = self._decide(self.agents[camp].act, msg)
                    if result is None:
                        errors.append(f"agent{camp}.act timeout after {self.decision_timeout}s")
                        self._send_timeout(camp, state)
                        continue
                    if isinstance(result, Exception):
                        errors.append(f"agent{camp}.act raised: {result!r}")
                        self._send_timeout(camp, state)
                        continue
                    action_inner = {
                        "round": round_no,
                        **result,
                    }
                    self._send_action(camp, action_inner)

            # 3) 收尾
            try:
                self.proc.kill()
            except OSError:
                pass
            self.proc.wait(timeout=5)
        except Exception as exc:  # pragma: no cover - 防御性兜底
            errors.append(f"host error: {exc!r}")
            terminated_by = "host_error"
            try:
                self.proc.kill()
            except OSError:
                pass
        finally:
            self._close_trace()

        return MatchResult(
            winner=winner,
            scores=scores,
            rounds=rounds,
            replay_path=replay_path,
            trace_path=self.trace_path or "",
            duration=time.time() - t0,
            terminated_by=terminated_by,
            errors=errors,
            frames=len(self._trace),
        )

    # ---- 超时/异常帧 ----
    def _send_timeout(self, player: int, state: int) -> None:
        """按官方约定向 logic 发 AI 超时/异常帧（player=-1）。"""
        content = json.dumps(
            {"error": 1, "state": state, "player": player}
        )
        self._send(
            {"player": -1, "content": content},
            kind="timeout",
        )
        self._last_was_timeout = True
