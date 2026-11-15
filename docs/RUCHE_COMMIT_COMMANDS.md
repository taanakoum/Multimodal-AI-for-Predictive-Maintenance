# Commit commands for the Ruche follow-up audit

Run from the repository root in PowerShell after reviewing `git diff`. No commits have
been created automatically. These four commits include source, regression checks and
current documentation; generated data, weights, manifests and validation reports stay
ignored under `runs/` or the external data directory.

```powershell
git add -- pdmfm/datasets/matwi.py pdmfm/preprocess.py pdmfm/windows.py tests/integration/test_lifecycle.py tests/unit/test_preprocess.py
git commit -m "fix(data): validate MATWI acquisition clocks and signal inputs"

git add -- pdmfm/protocols.py pdmfm/preflight.py pdmfm/model/inputs.py pdmfm/probe.py tests/unit/test_protocols.py tests/integration/test_preflight.py tests/integration/test_probe.py
git commit -m "fix(validation): audit grouped folds and include MATWI in Gate 1"

git add -- pdmfm/jobs.py pdmfm/compute.py pdmfm/encoders.py pdmfm/patchtst.py pdmfm/planb.py pdmfm/pretrain.py pdmfm/rul.py pdmfm/train.py pdmfm/tune.py pdmfm/baselines.py slurm/array.sbatch slurm/monitor.sh slurm/submit.sh slurm/job.sbatch slurm/cpu.sbatch slurm/checks.sbatch slurm/env.sh slurm/prefetch_weights.sh slurm/status.sh tests/unit/test_jobs.py tests/unit/test_audit_regressions.py tests/integration/test_job_workers.py tests/integration/test_pretrain.py tests/integration/test_slurm.py
git commit -m "perf(ruche): pack isolated single-seed jobs with coherent backbones"

git add -- README.md docs/PLAN.md docs/RUCHE_RUNBOOK.md docs/TESTING.md docs/AUDIT.md docs/RUCHE_AUDIT.md docs/RUCHE_COMMIT_COMMANDS.md
git commit -m "docs: present full research architecture and audited Ruche workflow"

git status --short
```

The validation commit depends on the preceding data-policy change. The scheduling commit
depends on the CV/preflight contract. The shared-backbone fixes are kept with the scheduling
code because those generated fine-tuning commands require compatible fusion architectures.
