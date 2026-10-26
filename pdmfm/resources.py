"""Check CPU and memory before heavy work, and size thread pools to what is free.

Every CPU-heavy script (EDA, feature caching, probes, training on CPU) calls
:func:`claim_resources` first. It samples the machine for about a second, lists other busy
processes (another script of ours, an IDE indexing, a download), and then:

* returns a thread count that leaves one core for the rest of the machine, capped by the
  idle CPUs and by ``PDMFM_MAX_THREADS``; inside a SLURM job the granted cores are used;
* refuses to start (or waits, with ``wait_s``) when fewer than ``min_idle_cpus`` CPUs are
  idle or less than ``min_free_ram_gb`` of RAM is free;
* applies the thread count to PyTorch and to NumPy/SciPy BLAS pools.

    python -m pdmfm.resources            # print the current status
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import psutil

GB = 1024**3


class ResourcesBusy(RuntimeError):
    """Raised when the machine is too busy for a heavy job."""


@dataclass(frozen=True)
class BusyProcess:
    pid: int
    name: str
    cpu_percent: float  # of one core; above 100 means several cores
    memory_gb: float
    command: str = ""


@dataclass(frozen=True)
class ResourceStatus:
    logical_cpus: int
    idle_cpus: float
    free_ram_gb: float
    total_ram_gb: float
    busy: list[BusyProcess] = field(default_factory=list)
    in_slurm: bool = False

    def describe(self) -> str:
        lines = [
            f"CPUs idle {self.idle_cpus:.1f} of {self.logical_cpus}; "
            f"RAM free {self.free_ram_gb:.1f} of {self.total_ram_gb:.1f} GB"
            + ("; inside a SLURM job" if self.in_slurm else "")
        ]
        for proc in self.busy:
            lines.append(
                f"  busy: {proc.name} (pid {proc.pid}) {proc.cpu_percent:.0f}% CPU, "
                f"{proc.memory_gb:.1f} GB  {proc.command[:90]}"
            )
        return "\n".join(lines)


def _slurm_cpus() -> int | None:
    value = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    return int(value) if value and value.isdigit() else None


def sample_status(seconds: float = 1.0, busy_threshold: float = 25.0) -> ResourceStatus:
    """Sample CPU use over ``seconds`` and list processes above ``busy_threshold`` % of a core."""
    own = os.getpid()
    procs = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            proc.cpu_percent(None)  # first call primes the counter
            procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    total_percent = psutil.cpu_percent(interval=seconds)
    busy = []
    for proc in procs:
        try:
            percent = proc.cpu_percent(None)
            if (
                proc.pid == own
                or percent < busy_threshold
                or proc.info["name"] == "System Idle Process"
            ):
                continue
            try:
                command = " ".join(proc.cmdline())
            except (psutil.AccessDenied, psutil.ZombieProcess):
                command = ""
            busy.append(
                BusyProcess(
                    pid=proc.pid,
                    name=proc.info["name"] or "?",
                    cpu_percent=percent,
                    memory_gb=proc.memory_info().rss / GB,
                    command=command,
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    memory = psutil.virtual_memory()
    return ResourceStatus(
        logical_cpus=logical,
        idle_cpus=max(0.0, logical * (1 - total_percent / 100)),
        free_ram_gb=memory.available / GB,
        total_ram_gb=memory.total / GB,
        busy=sorted(busy, key=lambda p: -p.cpu_percent),
        in_slurm="SLURM_JOB_ID" in os.environ,
    )


def recommended_threads(status: ResourceStatus, limit: int | None = None) -> int:
    """Threads for a heavy job: idle CPUs minus one spare, capped by limits."""
    if status.in_slurm and _slurm_cpus():
        threads = _slurm_cpus() or 1
    else:
        threads = math.floor(status.idle_cpus) - 1  # leave a core for the desktop
    cap = os.environ.get("PDMFM_MAX_THREADS", "")
    if cap.isdigit() and int(cap) > 0:
        threads = min(threads, int(cap))
    if limit is not None:
        threads = min(threads, limit)
    return max(1, threads)


def apply_threads(threads: int) -> None:
    """Limit PyTorch intra-op threads and the BLAS pools of NumPy and SciPy."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(threads)
    except ImportError:
        pass


def claim_resources(
    job: str,
    min_idle_cpus: float = 2.0,
    min_free_ram_gb: float = 2.0,
    limit: int | None = None,
    wait_s: float = 0.0,
    poll_s: float = 30.0,
    force: bool = False,
) -> int:
    """Check the machine before ``job``; return (and apply) the thread count to use.

    Raises :class:`ResourcesBusy` when the machine stays below the thresholds for
    ``wait_s`` seconds, unless ``force`` is set (then it only warns).
    """
    deadline = time.monotonic() + wait_s
    while True:
        status = sample_status()
        enough = status.idle_cpus >= min_idle_cpus and status.free_ram_gb >= min_free_ram_gb
        if enough or status.in_slurm:
            break
        if force:
            print(f"[resources] {job}: machine is busy, continuing because force=True")
            break
        if time.monotonic() >= deadline:
            raise ResourcesBusy(
                f"{job} needs {min_idle_cpus} idle CPUs and {min_free_ram_gb} GB free RAM.\n"
                + status.describe()
            )
        print(f"[resources] {job}: waiting for the machine to free up ({status.describe()})")
        time.sleep(poll_s)
    threads = recommended_threads(status, limit)
    apply_threads(threads)
    print(f"[resources] {job}: {status.describe()}\n[resources] {job}: using {threads} threads")
    return threads


def main() -> None:
    status = sample_status()
    print(status.describe())
    print(f"recommended threads for a heavy job: {recommended_threads(status)}")


if __name__ == "__main__":
    main()
