"""The Ruche job scripts in slurm/ (Linux only, like Ruche; CI runs them on Ubuntu)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from pdmfm.paths import REPO_ROOT

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="Linux shell scripts"
)
SLURM = REPO_ROOT / "slurm"


def bash(*args, env=None, cwd=REPO_ROOT):
    return subprocess.run(
        ["bash", *map(str, args)], capture_output=True, text=True, env=env, cwd=cwd
    )


@pytest.mark.parametrize("script", sorted(p.name for p in SLURM.iterdir()))
def test_scripts_parse(script):
    assert bash("-n", SLURM / script).returncode == 0


def test_chain_links_each_part_to_the_previous_one(tmp_path):
    log = tmp_path / "sbatch.log"
    fake = tmp_path / "bin" / "sbatch"
    fake.parent.mkdir()
    fake.write_text(  # records its arguments, prints "<job id>;<cluster>" like --parsable
        "#!/bin/bash\n"
        'echo "$@" >> "$SBATCH_LOG"\n'
        'echo "$(( $(wc -l < "$SBATCH_LOG") + 1000 ));ruche"\n'
    )
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake.parent}:{os.environ['PATH']}", "SBATCH_LOG": str(log)}
    result = bash(
        SLURM / "chain.sh", 3, "--job-name=x", "slurm/job.sbatch", "python", "-c", "pass", env=env
    )
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == 3
    assert "--dependency" not in calls[0]
    assert "--dependency=afterany:1001" in calls[1] and "--dependency=afterany:1002" in calls[2]
    assert all(c.endswith("--job-name=x slurm/job.sbatch python -c pass") for c in calls)
    assert "scancel 1001 1002 1003" in result.stdout


def test_chain_rejects_a_bad_part_count():
    assert bash(SLURM / "chain.sh", "zero", "slurm/job.sbatch").returncode == 2


@pytest.mark.parametrize("script", ["job.sbatch", "cpu.sbatch", "checks.sbatch"])
def test_job_scripts_refuse_to_run_outside_slurm(script):
    env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
    result = bash(SLURM / script, "python", "-c", "pass", env=env)
    assert result.returncode == 2 and "sbatch" in result.stderr


def test_status_runs_without_squeue(tmp_path):
    env = {**os.environ, "PDMFM_DATA_ROOT": str(tmp_path), "PATH": os.environ["PATH"]}
    result = bash(SLURM / "status.sh", env=env)
    assert result.returncode == 0, result.stderr
    for header in (
        "== Your jobs",
        "== Window stores",
        "== Feature caches",
        "== PatchTST",
        "== Gate-1",
        "== Plan B",
        "== Gate-2",
        "== RUL",
    ):
        assert header in result.stdout


def test_status_counts_plan_b_results(tmp_path):
    run = tmp_path / "runs" / "planb" / "planb-v1-nonastreda"
    run.mkdir(parents=True)
    for name in ("fold1-seed0.json", "fold1-seed1.json", "fold1-seed0.npz", "summary.md"):
        (run / name).write_text("{}")
    shutil.copytree(SLURM, tmp_path / "slurm")
    env = {**os.environ, "PDMFM_DATA_ROOT": str(tmp_path / "data")}
    result = bash(tmp_path / "slurm" / "status.sh", env=env, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "planb-v1-nonastreda: 2 of 15" in result.stdout and "summary.md" in result.stdout


def test_status_counts_gate2_and_rul_results(tmp_path):
    gate2 = tmp_path / "runs" / "gate2" / "gate2-v1-nonastreda"
    gate2.mkdir(parents=True)
    rows = ["candidate,outer,split,score", "planb,1,a,0.5", "planb,1,b,0.5", "concat,1,a,0.6"]
    (gate2 / "results.csv").write_text("\n".join(rows) + "\n")
    (gate2 / "decision.json").write_text('{"model": "planb"}')
    rul = tmp_path / "runs" / "rul" / "rul-v1-xjtu"
    rul.mkdir(parents=True)
    (rul / "fold1-seed0.json").write_text("{}")
    shutil.copytree(SLURM, tmp_path / "slurm")
    env = {**os.environ, "PDMFM_DATA_ROOT": str(tmp_path / "data")}
    result = bash(tmp_path / "slurm" / "status.sh", env=env, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "gate2-v1-nonastreda: concat 1, planb 2; model planb" in result.stdout
    assert "rul-v1-xjtu: 1 of 15" in result.stdout
