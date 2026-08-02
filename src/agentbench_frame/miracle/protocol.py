"""24_miracle 官方逻辑线协议编解码（自己实现，未改动官方代码）。

协议来自官方 `logic/gamecode_logic/main.py`（原样移植于
`AgentBench/backend_sources/corpus/24_miracle/logic/gamecode_logic`）：

- logic → 评测机: ``int32(len) + int32(target) + payload(json)``
- 评测机 → logic: ``int32(len) + payload(json)``
- logic 的选卡/回合消息 ``content[0]`` 为 ``"000000"+长度+json`` 形式
  （6 位十进制长度前缀，前面补零）。
"""

from __future__ import annotations

import json
import os
import select
import time
from typing import Optional, Tuple

__all__ = [
    "encode_to_logic",
    "read_exact",
    "read_logic_frame",
    "decode_content",
    "MAX_FRAME_BYTES",
]

#: 单帧 payload 上限（防御性限制，官方消息远小于此）
MAX_FRAME_BYTES = 1 << 22  # 4 MiB


class ProtocolError(Exception):
    """协议错误：帧头非法、长度越界、JSON 解析失败等。"""


def encode_to_logic(payload: dict) -> bytes:
    """把一条消息编码为发给 logic 的字节流：``int32(len) + json``。

    注意：logic 的 ``read_opt()`` 只读 4 字节长度头 + payload，
    与 logic 发送方向（带 target）不对称，这是官方协议本身的设计。
    """
    data = json.dumps(payload).encode("utf-8")
    if len(data) > MAX_FRAME_BYTES:
        raise ProtocolError(f"payload too large: {len(data)} bytes")
    return len(data).to_bytes(4, "big", signed=True) + data


def read_exact(stream, size: int, timeout: float, label: str = "stream") -> bytes:
    """带超时精确读取 ``size`` 字节；用 select 判定可读，超时抛 TimeoutError。"""
    deadline = time.monotonic() + timeout
    buf = bytearray()
    while len(buf) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timeout reading {label} ({len(buf)}/{size} bytes)")
        ready, _, _ = select.select([stream], [], [], remaining)
        if not ready:
            raise TimeoutError(f"timeout reading {label} ({len(buf)}/{size} bytes)")
        chunk = os.read(stream.fileno(), size - len(buf))
        if not chunk:
            raise EOFError(f"{label}: stream closed while reading {size} bytes")
        buf += chunk
    return bytes(buf)


def read_logic_frame(
    stream, timeout: float, label: str = "logic"
) -> Tuple[int, dict]:
    """从 logic stdout 读一帧，返回 ``(target, payload_dict)``。"""
    header = read_exact(stream, 8, timeout, label)
    length = int.from_bytes(header[:4], "big", signed=True)
    target = int.from_bytes(header[4:], "big", signed=True)
    if length < 0 or length > MAX_FRAME_BYTES:
        raise ProtocolError(f"invalid frame length: {length}")
    raw = read_exact(stream, length, timeout, label)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - 防御分支
        raise ProtocolError(f"bad json from logic: {exc}") from exc
    return target, payload


def decode_content(content: list) -> Optional[dict]:
    """解析 logic ``content[0]``（``"000000"+长度+json``）为内层消息 dict。

    返回 None 表示无法解析（协议异常）。
    """
    if not content or not isinstance(content[0], str):
        return None
    text = content[0]
    if len(text) >= 6 and text[:6].isdigit():
        text = text[6:]
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        return None
    return msg if isinstance(msg, dict) else None
