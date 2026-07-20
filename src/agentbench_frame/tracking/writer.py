"""
JSONL writer with buffered async writes.

Writes record dicts as newline-delimited JSON. Supports optional background
thread for non-blocking writes.
"""

import json
import os
import queue
import threading
from dataclasses import asdict
from typing import Any, Dict, Optional


class JSONLWriter:
    """Buffered JSONL writer with optional background thread.

    Usage:
        writer = JSONLWriter("runs/my_run/events.jsonl")
        writer.write({"event": "step", "reward": 1.0})
        writer.close()
    """

    def __init__(self, path: str, buffer_size: int = 64,
                 async_mode: bool = False):
        self.path = path
        self.buffer_size = buffer_size
        self.async_mode = async_mode

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "w")
        self._buffer: list = []
        self._lock = threading.Lock()

        if async_mode:
            self._queue: queue.Queue = queue.Queue()
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._running = True
            self._thread.start()
        else:
            self._queue = None
            self._thread = None

    def write(self, record: Any):
        """Write a record (dict or dataclass) to the log."""
        if isinstance(record, dict):
            d = record
        elif hasattr(record, "__dataclass_fields__"):
            d = asdict(record)
        else:
            d = {"raw": str(record)}

        if self.async_mode and self._queue is not None:
            self._queue.put(d)
        else:
            self._write_dict(d)

    def _write_dict(self, d: dict):
        with self._lock:
            self._buffer.append(d)
            if len(self._buffer) >= self.buffer_size:
                self._flush_locked()

    def _flush_locked(self):
        if not self._buffer:
            return
        for record in self._buffer:
            line = json.dumps(record, default=str)
            self._fh.write(line + "\n")
        self._fh.flush()
        self._buffer.clear()

    def flush(self):
        with self._lock:
            self._flush_locked()

    def _worker(self):
        while self._running:
            try:
                d = self._queue.get(timeout=0.5)
                self._write_dict(d)
            except queue.Empty:
                continue

    def close(self):
        if self.async_mode:
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=2.0)
        self.flush()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
