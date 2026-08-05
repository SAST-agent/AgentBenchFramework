"""Adaptive bounded scheduling for subprocess-supervising match workers."""

from __future__ import annotations

import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Generic, Iterable, TypeVar

import psutil


TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")


def _cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


@dataclass(frozen=True)
class SchedulerConfig:
    target_cpu: float = 70.0
    lower_cpu: float = 65.0
    upper_cpu: float = 75.0
    max_workers: int = field(default_factory=_cpu_count)
    sample_seconds: float = 0.5

    def __post_init__(self) -> None:
        if not 0 <= self.lower_cpu <= self.target_cpu <= self.upper_cpu <= 100:
            raise ValueError("CPU thresholds must satisfy 0 <= lower <= target <= upper <= 100")
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive")
        if self.sample_seconds <= 0:
            raise ValueError("sample_seconds must be positive")


@dataclass(frozen=True)
class TaskOutcome(Generic[ResultT]):
    task_id: str
    status: str
    result: ResultT | None
    error: str | None
    seconds: float


@dataclass(frozen=True)
class SchedulerReport(Generic[ResultT]):
    outcomes: tuple[TaskOutcome[ResultT], ...]
    cpu_history: tuple[float, ...]
    worker_history: tuple[int, ...]
    cancelled_task_ids: tuple[str, ...] = ()


class ProcessTreeCpuSampler:
    """Measure this process and recursive children as percent of the machine."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.known: dict[int, psutil.Process] = {}

    def __call__(self) -> float:
        processes = [self.process]
        try:
            processes.extend(self.process.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        live_pids = {process.pid for process in processes}
        self.known = {pid: process for pid, process in self.known.items() if pid in live_pids}
        total = 0.0
        for process in processes:
            try:
                if process.pid not in self.known:
                    process.cpu_percent(interval=None)
                    self.known[process.pid] = process
                else:
                    total += max(0.0, process.cpu_percent(interval=None))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return min(100.0, total / max(1, psutil.cpu_count() or 1))


class AdaptiveScheduler:
    def __init__(self, config: SchedulerConfig,
                 cpu_sampler: Callable[[], float] | None = None) -> None:
        self.config = config
        self.cpu_sampler = cpu_sampler or ProcessTreeCpuSampler()

    def run(self, tasks: Iterable[TaskT],
            worker: Callable[[TaskT], ResultT]) -> SchedulerReport[ResultT]:
        ordered_tasks = tuple(tasks)
        task_ids = tuple(str(getattr(task, "task_id")) for task in ordered_tasks)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("scheduler task IDs must be unique")
        if not ordered_tasks:
            return SchedulerReport((), (), (), ())

        outcomes: list[TaskOutcome[ResultT] | None] = [None] * len(ordered_tasks)
        cpu_history: list[float] = []
        worker_history: list[int] = []
        next_index = 0
        concurrency = 1
        consecutive_high = 0
        paused = False

        def invoke(index: int, task: TaskT) -> tuple[int, TaskOutcome[ResultT]]:
            started = time.monotonic()
            task_id = task_ids[index]
            try:
                result = worker(task)
                outcome = TaskOutcome(task_id, "completed", result, None,
                                      time.monotonic() - started)
            except Exception as error:  # Worker failures are benchmark evidence.
                outcome = TaskOutcome(
                    task_id, "failed", None,
                    f"{type(error).__name__}: {error}", time.monotonic() - started,
                )
            return index, outcome

        with ThreadPoolExecutor(max_workers=self.config.max_workers,
                                thread_name_prefix="doto-match") as executor:
            running: dict[Future[tuple[int, TaskOutcome[ResultT]]], int] = {}
            while next_index < len(ordered_tasks) or running:
                while (not paused and next_index < len(ordered_tasks)
                       and len(running) < concurrency):
                    future = executor.submit(invoke, next_index, ordered_tasks[next_index])
                    running[future] = next_index
                    next_index += 1

                if running:
                    completed, _ = wait(
                        tuple(running), timeout=self.config.sample_seconds,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in completed:
                        running.pop(future)
                        index, outcome = future.result()
                        outcomes[index] = outcome
                else:
                    time.sleep(self.config.sample_seconds)

                cpu = max(0.0, min(100.0, float(self.cpu_sampler())))
                cpu_history.append(cpu)
                if cpu > self.config.upper_cpu:
                    paused = True
                    consecutive_high += 1
                    if consecutive_high >= 2:
                        concurrency = max(1, concurrency - 1)
                        consecutive_high = 0
                else:
                    paused = False
                    consecutive_high = 0
                    if cpu < self.config.lower_cpu:
                        concurrency = min(self.config.max_workers, concurrency + 1)
                worker_history.append(concurrency)

        return SchedulerReport(
            outcomes=tuple(outcome for outcome in outcomes if outcome is not None),
            cpu_history=tuple(cpu_history),
            worker_history=tuple(worker_history),
            cancelled_task_ids=(),
        )
