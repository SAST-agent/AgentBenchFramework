"""
ResourceSampler background thread.

Samples CPU/RSS via psutil and GPU via pynvml stubs.
Runs in a daemon thread, pushing ResourceRecord dicts to a callback.
"""

import threading
import time
from typing import Any, Callable, Dict, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


class ResourceSampler:
    """Periodically samples system resources in a background thread.

    Usage:
        def on_sample(record):
            writer.write(record)

        sampler = ResourceSampler(on_sample, interval_s=5)
        sampler.start()
        # ... run experiment ...
        sampler.stop()
    """

    def __init__(self,
                 callback: Callable[[Dict[str, Any]], None],
                 interval_s: float = 5.0):
        self.callback = callback
        self.interval_s = interval_s
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._nvidia_available = False

        # Attempt GPU init
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self._nvidia_available = True
            except Exception:
                self._nvidia_available = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        # Shutdown NVML
        if self._nvidia_available and HAS_PYNVML:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _loop(self):
        while self._running:
            record = self._sample()
            try:
                self.callback(record)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def _sample(self) -> Dict[str, Any]:
        ts = time.time()
        r = {
            "event": "resource",
            "timestamp": ts,
            "cpu_percent": 0.0,
            "rss_mb": 0.0,
            "vms_mb": 0.0,
            "gpu_util": None,
            "gpu_mem_mb": None,
        }

        if HAS_PSUTIL:
            try:
                proc = psutil.Process()
                r["cpu_percent"] = proc.cpu_percent(interval=0.1)
                mem = proc.memory_info()
                r["rss_mb"] = mem.rss / (1024 * 1024)
                r["vms_mb"] = mem.vms / (1024 * 1024)
            except Exception:
                pass

        if self._nvidia_available and HAS_PYNVML:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                r["gpu_util"] = util.gpu
                r["gpu_mem_mb"] = mem_info.used / (1024 * 1024)
            except Exception:
                pass

        return r

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
