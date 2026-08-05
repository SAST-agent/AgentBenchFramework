from dataclasses import dataclass
from time import sleep

from agentbench_frame.doto.scheduler import AdaptiveScheduler, SchedulerConfig


@dataclass(frozen=True)
class ExampleTask:
    task_id: str


class FakeCpuSampler:
    def __init__(self, values):
        self.values = iter(values)
        self.last = 70.0

    def __call__(self):
        self.last = next(self.values, self.last)
        return self.last


def _successful_worker(task):
    sleep(0.005)
    return f"done-{task.task_id}"


def test_scheduler_grows_holds_and_throttles_without_cancelling():
    scheduler = AdaptiveScheduler(
        SchedulerConfig(
            target_cpu=70,
            lower_cpu=65,
            upper_cpu=75,
            max_workers=8,
            sample_seconds=0.001,
        ),
        cpu_sampler=FakeCpuSampler([40, 62, 69, 81, 82, 70]),
    )
    tasks = [ExampleTask(str(index)) for index in range(12)]
    report = scheduler.run(tasks, _successful_worker)
    assert max(report.worker_history) > 1
    assert report.cancelled_task_ids == ()
    assert [outcome.task_id for outcome in report.outcomes] == [str(index) for index in range(12)]
    assert all(outcome.status == "completed" for outcome in report.outcomes)


def test_scheduler_records_worker_exception_and_completes_other_tasks():
    def worker(task):
        if task.task_id == "1":
            raise RuntimeError("expected failure")
        return task.task_id

    report = AdaptiveScheduler(
        SchedulerConfig(max_workers=2, sample_seconds=0.001),
        cpu_sampler=FakeCpuSampler([60, 70, 70]),
    ).run([ExampleTask("0"), ExampleTask("1"), ExampleTask("2")], worker)
    assert [outcome.status for outcome in report.outcomes] == ["completed", "failed", "completed"]
    assert report.outcomes[1].error == "RuntimeError: expected failure"
