#!/bin/bash
# Step 0 on a CPU node: tests, the window stores and the feature caches that are present.
# Submit from the repository folder:  sbatch slurm/checks.sbatch
#SBATCH --job-name=step0-checks
#SBATCH --output=runs/logs/%x-%j.out
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Submit this script with sbatch; see docs/RUCHE_RUNBOOK.md." >&2
    exit 2
fi
cd "$SLURM_SUBMIT_DIR" || exit 2
source slurm/env.sh
set -eo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "commit: $(git log -1 --oneline 2>/dev/null || echo unknown)"
echo "=== tests (real-data and checkpoint tests run where their data and weights exist)"
python -m pytest -o addopts= -q -rs tests
echo "=== window stores"
python -m pdmfm.windows info
echo "=== feature caches"
python -m pdmfm.features status
echo "finished $(date)"
