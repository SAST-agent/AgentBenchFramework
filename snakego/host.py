"""Local SnakeGo match host.

Spawns two AI subprocesses (compiled C++ humans, or a Python agent wrapped to
speak the same adk byte protocol) and relays moves between them while tracking
game state with the bit-exact Engine port of adk.hpp.

Wire protocol (matches adk.hpp on stdio):
  init -> length(1) width(1) max_round(2 BE) player(1)
          0x10(1) item_count(2 BE) [x y type time(2) param(2)] * count
  mover turn: host reads 5 bytes [0 0 0 1 op]; sends ack op(1) to mover,
              sends op(1) to opponent
  game over:  host sends 0x11 then gameover_type(1) winner(1) p0(2 BE) p1(2 BE)
"""
import json
import os
import random
import subprocess
import struct
import sys
import time

from snakego.engine import Engine, Item

LENGTH = 16
WIDTH = 16
MAX_ROUND = 512

# Item spawn schedule copied verbatim from the official judge spawn.py:
# (start_round, end_round, mean_density, area(x1,y1,x2,y2), weight(growth,split,fire))
SPAWN_CONFIGS = [
    (1, 20, 0.25, (6, 6, 9, 9), (5, 0, 1)),
    (21, 40, 0.25, (5, 5, 10, 10), (5, 0, 1)),
    (41, 60, 0.25, (4, 4, 11, 11), (5, 0, 1)),
    (61, 64, 0.25, (3, 3, 12, 12), (5, 0, 1)),
    (65, 80, 0.25, (3, 3, 12, 12), (5, 0, 2)),
    (81, 100, 0.25, (2, 2, 13, 13), (5, 0, 2)),
    (101, 120, 0.25, (1, 1, 14, 14), (5, 0, 2)),
    (121, 384, 0.25, (0, 0, 15, 15), (5, 0, 2)),
    (385, 512, 0.25, (0, 0, 15, 15), (4, 0, 3)),
]
ITEM_EXPIRE_TIME = 16


def generate_items(seed, length=LENGTH, width=WIDTH, max_round=MAX_ROUND):
    """Faithful port of judge_dev_logic/logic/spawn.py generate_items."""
    rng = random.Random(seed)
    item_map = [[-1024 for _ in range(width)] for _ in range(length)]
    items = []
    current_round = 1

    def gen_single(round_no, cfg):
        x1, y1, x2, y2 = cfg[3]
        x = rng.randint(x1, x2)
        y = rng.randint(y1, y2)
        while item_map[x][y] + ITEM_EXPIRE_TIME >= round_no:
            x = rng.randint(x1, x2)
            y = rng.randint(y1, y2)
        item_map[x][y] = round_no
        w = cfg[4]
        t = rng.randint(0, sum(w) - 1)
        if t in range(w[0]):
            return Item(x, y, len(items), round_no, 0, rng.randint(1, 5))
        elif t in range(w[0] + w[1]):
            return Item(x, y, len(items), round_no, 0, rng.randint(1, 5))
        else:
            return Item(x, y, len(items), round_no, 2, max_round)

    for cfg in SPAWN_CONFIGS:
        while cfg[0] <= current_round <= cfg[1]:
            if rng.uniform(0, 1) <= cfg[2]:
                items.append(gen_single(current_round, cfg))
            current_round += 1
    return items


def _b1(v):
    return struct.pack(">B", v & 0xFF)


def _b2(v):
    return struct.pack(">h", v)


def build_init_bytes(items, player_id):
    msg = _b1(LENGTH) + _b1(WIDTH) + _b2(MAX_ROUND) + _b1(player_id)
    msg += _b1(0x10) + _b2(len(items))
    for it in items:
        msg += _b1(it.x) + _b1(it.y) + _b1(it.type) + _b2(it.time) + _b2(it.param)
    return msg


class Player:
    """Wraps a subprocess AI speaking the adk stdio byte protocol."""

    def __init__(self, name, cmdline, cwd=None):
        self.name = name
        self.cmdline = cmdline
        self.cwd = cwd
        self.proc = None
        self.err = b""

    def start(self, init_bytes):
        self.proc = subprocess.Popen(
            self.cmdline,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
        )
        self.proc.stdin.write(init_bytes)
        self.proc.stdin.flush()

    def read_op(self, timeout=10.0):
        # 5 bytes: [0,0,0,1, op_type]
        return _read_exact(self.proc.stdout, 5, self.proc, timeout)

    def send_ack(self, op_type):
        self.proc.stdin.write(_b1(op_type))
        self.proc.stdin.flush()

    def send_op(self, op_type):
        self.proc.stdin.write(_b1(op_type))
        self.proc.stdin.flush()

    def send_gameover(self, gameover_type, winner, scores):
        msg = _b1(0x11) + _b1(gameover_type) + _b1(winner)
        msg += _b2(scores[0]) + _b2(scores[1])
        self.proc.stdin.write(msg)
        self.proc.stdin.flush()

    def drain_stderr(self):
        try:
            self.err = self.proc.stderr.read()
        except Exception:
            self.err = b""
        return self.err

    def close(self):
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass


def _read_exact(stream, n, proc, timeout):
    """Read exactly n bytes, raising on EOF/timeout/crash."""
    buf = b""
    deadline = time.time() + timeout
    while len(buf) < n:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(f"read timed out after {timeout}s")
        if proc.poll() is not None:
            raise RuntimeError(f"subprocess exited (code {proc.returncode})")
        chunk = stream.read(n - len(buf))
        if not chunk:
            time.sleep(0.001)
            continue
        buf += chunk
    return buf


class MatchResult:
    def __init__(self):
        self.winner = -1
        self.scores = [0, 0]
        self.moves = 0
        self.rounds = 0
        self.end_reason = ""
        self.ops = []          # [(round, player, snake_id, op_type)]
        self.snapshots = []    # board snapshots at each step
        self.items = []
        self.duration = 0.0
        self.error = None
        self.names = ["", ""]

    def to_dict(self):
        return {
            "winner": self.winner,
            "scores": self.scores,
            "moves": self.moves,
            "rounds": self.rounds,
            "end_reason": self.end_reason,
            "names": self.names,
            "items": [{"x": it.x, "y": it.y, "time": it.time, "type": it.type, "param": it.param} for it in self.items],
            "ops": self.ops,
            "duration": round(self.duration, 2),
        }


def run_match(player0, player1, seed=0, time_limit_per_move=8.0, record=False, verbose=False):
    """Run a full game between two Player subprocesses.

    player0/player1 must be Player instances (not yet started). Returns MatchResult.
    """
    items = generate_items(seed)
    eng = Engine(LENGTH, WIDTH, MAX_ROUND, items)

    result = MatchResult()
    result.items = items
    result.names = [player0.name, player1.name]
    t0 = time.time()

    player0.start(build_init_bytes(items, 0))
    player1.start(build_init_bytes(items, 1))

    player0.engine = eng
    player1.engine = eng
    player0.player_id = 0
    player1.player_id = 1

    try:
        players = [player0, player1]
        step = 0
        while True:
            cur = eng.current_player
            mover = players[cur]
            other = players[1 - cur]

            if not eng.alive_player():
                # dead player: advance turn without consulting its subprocess
                running = eng.do_operation(0)
                step += 1
                if record:
                    result.ops.append([eng.current_round, cur, -1, 0])
                if not running:
                    scores = eng.score()
                    winner = 0 if scores[0] >= scores[1] else 1
                    player0.send_gameover(0, winner, scores)
                    player1.send_gameover(0, winner, scores)
                    result.scores = scores
                    result.winner = winner
                    result.rounds = eng.current_round
                    result.moves = step
                    result.end_reason = "normal"
                    break
                continue

            data = mover.read_op(timeout=time_limit_per_move + 2.0)
            op_type = data[4]
            if not (1 <= op_type <= 6):
                raise RuntimeError(f"player {cur} ({mover.name}) sent illegal op {op_type}")

            moved_snake_id = eng.current_snake_id
            running = eng.do_operation(op_type)
            step += 1

            if record:
                result.ops.append([eng.current_round, cur, moved_snake_id, op_type])

            # Relay: ack to mover, forward op to opponent.
            mover.send_ack(op_type)
            other.send_op(op_type)

            if not running:
                # Game over: send 0x11 + gameover info to both.
                scores = eng.score()
                winner = 0 if scores[0] >= scores[1] else 1
                if verbose:
                    print(f"[host] game over scores={scores} winner={winner} round={eng.current_round}", file=sys.stderr)
                player0.send_gameover(0, winner, scores)
                player1.send_gameover(0, winner, scores)
                result.scores = scores
                result.winner = winner
                result.rounds = eng.current_round
                result.moves = step
                result.end_reason = "normal"
                break

            if verbose and step % 200 == 0:
                print(f"[host] step {step} round {eng.current_round} player {cur} scores={eng.score()}", file=sys.stderr)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.end_reason = "error"
        result.scores = eng.score()
        result.winner = 1 - eng.current_player if "exited" in str(exc).lower() or "timed out" in str(exc).lower() else -1
        result.rounds = eng.current_round
        result.moves = step if "step" in dir() else 0
        if verbose:
            print(f"[host] ERROR: {result.error}", file=sys.stderr)
            p0_err = player0.drain_stderr()
            p1_err = player1.drain_stderr()
            if p0_err:
                print(f"[host] p0 stderr: {p0_err[:500]!r}", file=sys.stderr)
            if p1_err:
                print(f"[host] p1 stderr: {p1_err[:500]!r}", file=sys.stderr)
    finally:
        result.duration = time.time() - t0
        player0.close()
        player1.close()

    return result


def player_from_name(name, bin_dir=None):
    """Build a Player for a compiled human AI in bin_humans/."""
    if bin_dir is None:
        bin_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin_humans")
    path = name if name.endswith(".exe") else os.path.join(bin_dir, name + ".exe")
    return Player(name, [path])


# ---------------------------------------------------------------------------
# Socket-channel mode (binary-safe; bypasses Windows stdio text-mode EOF bug).
# adk.hpp connects back to host:port as a TCP client when given argc==3.
# ---------------------------------------------------------------------------
import socket
import threading


class SocketPlayer:
    """A human AI run as a TCP client connecting back to a host socket.

    Uses adk.hpp's socket_channel(host, port) which is true binary I/O, unlike
    stdio getchar() that chokes on byte 0x1A under Windows text mode.
    """

    def __init__(self, name, cmdline, host="127.0.0.1", cwd=None):
        self.name = name
        self.cmdline = cmdline
        self.host = host
        self.cwd = cwd
        self.proc = None
        self.conn = None
        self.port = None
        self.err = b""

    def start(self, init_bytes):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, 0))
        self.port = srv.getsockname()[1]
        srv.listen(1)
        srv.settimeout(10.0)
        full_cmd = list(self.cmdline) + [self.host, str(self.port)]
        self.proc = subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.cwd)
        self.conn, _ = srv.accept()
        self.conn.settimeout(10.0)
        srv.close()
        self.conn.sendall(init_bytes)

    def _recv_exact(self, n, timeout):
        buf = b""
        self.conn.settimeout(timeout)
        while len(buf) < n:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} exited (code {self.proc.returncode})")
            chunk = self.conn.recv(n - len(buf))
            if not chunk:
                raise RuntimeError(f"{self.name} connection closed")
            buf += chunk
        return buf

    def read_op(self, timeout=12.0):
        return self._recv_exact(5, timeout)

    def send_ack(self, op_type):
        self.conn.sendall(_b1(op_type))

    def send_op(self, op_type):
        self.conn.sendall(_b1(op_type))

    def send_gameover(self, gameover_type, winner, scores):
        msg = _b1(0x11) + _b1(gameover_type) + _b1(winner) + _b2(scores[0]) + _b2(scores[1])
        self.conn.sendall(msg)

    def drain_stderr(self):
        try:
            self.err = self.proc.stderr.read()
        except Exception:
            self.err = b""
        return self.err

    def close(self):
        try:
            if self.conn:
                self.conn.close()
        except Exception:
            pass
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass


def socket_player(name, bin_dir=None, host="127.0.0.1"):
    if bin_dir is None:
        bin_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin_humans")
    path = name if name.endswith(".exe") else os.path.join(bin_dir, name + ".exe")
    return SocketPlayer(name, [path], host=host)


class PythonPlayer:
    """In-process agent: no subprocess. Decides from the shared Engine state.

    Lets my if-else strategies play the same match loop as compiled human AIs,
    without spawning a process. The host sets .engine and .player_id before the
    first read_op.
    """

    def __init__(self, name, decide_fn):
        self.name = name
        self.decide = decide_fn
        self.engine = None
        self.player_id = None
        self.err = b""

    def start(self, init_bytes):
        pass

    def read_op(self, timeout=12.0):
        op = int(self.decide(self.engine, self.player_id))
        if not (1 <= op <= 6):
            op = 1
        return bytes([0, 0, 0, 1, op])

    def send_ack(self, op_type):
        pass

    def send_op(self, op_type):
        pass

    def send_gameover(self, gameover_type, winner, scores):
        pass

    def drain_stderr(self):
        return b""

    def close(self):
        pass


def python_player(name, decide_fn):
    return PythonPlayer(name, decide_fn)
