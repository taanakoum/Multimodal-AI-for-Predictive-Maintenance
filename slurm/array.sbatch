#!/bin/bash
# Use: bash slurm/submit.sh runs/jobs/source.json 2
# One array element gets one GPU and runs up to WORKERS isolated cached-model fits.
#SBATCH --job-name=pdmfm-array
#SBATCH --output=runs/logs/%x-%A_%a.out
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

set -eo pipefail
if [ -z "${SLURM_JOB_ID:-}" ] || [ "$#" -ne 2 ]; then
    echo "Use bash slurm/submit.sh MANIFEST WORKERS (submits via sbatch)." >&2
    exit 2
fi
cd "$SLURM_SUBMIT_DIR" || exit 2
source slurm/env.sh
export SLURM_EXPORT_ENV=ALL
source slurm/monitor.sh
echo "commit: $(git log -1 --oneline), job $SLURM_JOB_ID, task ${SLURM_ARRAY_TASK_ID:-0}"
python -m pdmfm.jobs run "$1" --workers "$2"
