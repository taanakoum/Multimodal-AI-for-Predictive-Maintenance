import pytest

from pdmfm import resources
from pdmfm.resources import ResourcesBusy, ResourceStatus


def _status(idle=6.0, free=8.0, slurm=False):
    return ResourceStatus(
        logical_cpus=8, idle_cpus=idle, free_ram_gb=free, total_ram_gb=16.0, in_slurm=slurm
    )


def test_sample_status_reads_the_machine():
    status = resources.sample_status(seconds=0.1)
    assert status.logical_cpus >= 1
    assert 0 <= status.idle_cpus <= status.logical_cpus
    assert 0 < status.free_ram_gb <= status.total_ram_gb
    assert "CPUs idle" in status.describe()


def test_recommended_threads_leave_a_core_and_respect_caps(monkeypatch):
    monkeypatch.delenv("PDMFM_MAX_THREADS", raising=False)
    assert resources.recommended_threads(_status(idle=6.4)) == 5
    assert resources.recommended_threads(_status(idle=0.3)) == 1  # never zero
    assert resources.recommended_threads(_status(idle=6.4), limit=2) == 2
    monkeypatch.setenv("PDMFM_MAX_THREADS", "3")
    assert resources.recommended_threads(_status(idle=6.4)) == 3


def test_slurm_jobs_use_their_granted_cores(monkeypatch):
    monkeypatch.delenv("PDMFM_MAX_THREADS", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert resources.recommended_threads(_status(idle=0.5, slurm=True)) == 8


def test_claim_refuses_a_busy_machine(monkeypatch):
    monkeypatch.setattr(resources, "sample_status", lambda: _status(idle=0.5, free=1.0))
    with pytest.raises(ResourcesBusy, match="needs 2.0 idle CPUs"):
        resources.claim_resources("test job")


def test_claim_can_be_forced(monkeypatch):
    monkeypatch.setattr(resources, "sample_status", lambda: _status(idle=0.5, free=1.0))
    monkeypatch.setattr(resources, "apply_threads", lambda n: None)
    assert resources.claim_resources("test job", force=True) == 1


def test_claim_waits_until_the_machine_frees_up(monkeypatch):
    states = iter([_status(idle=0.5), _status(idle=6.5)])
    monkeypatch.setattr(resources, "sample_status", lambda: next(states))
    monkeypatch.setattr(resources, "apply_threads", lambda n: None)
    monkeypatch.setattr(resources.time, "sleep", lambda s: None)
    monkeypatch.delenv("PDMFM_MAX_THREADS", raising=False)
    assert resources.claim_resources("test job", wait_s=60, poll_s=1) == 5


def test_apply_threads_limits_torch(monkeypatch):
    torch = pytest.importorskip("torch")
    # apply_threads writes OMP_NUM_THREADS; monkeypatch restores it for the other tests.
    monkeypatch.setenv("OMP_NUM_THREADS", str(torch.get_num_threads()))
    before = torch.get_num_threads()
    try:
        resources.apply_threads(1)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)
