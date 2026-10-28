#!/bin/bash
# Runs one pdmfm command on a Ruche GPU node.
#
# Submit it from the repository folder, with the command after the script name:
#   sbatch --job-name=NAME [more sbatch options] slurm/job.sbatch COMMAND [ARGUMENTS...]
# Example:
#   sbatch --job-name=moment-T slurm/job.sbatch \
#     python -m pdmfm.features extract --dataset cwru xjtu_sy nonastreda --encoder moment --arms T --device cuda
#
# The lines below are defaults; options given to sbatch on the command line override them.
#SBATCH --job-name=pdmfm
#SBATCH --output=runs/logs/%x-%j.out
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Submit this script with sbatch; see docs/RUCHE_RUNBOOK.md." >&2
    exit 2
fi
if [ "$#" -eq 0 ]; then
    echo "No command given after slurm/job.sbatch." >&2
    exit 2
fi
cd "$SLURM_SUBMIT_DIR" || exit 2
source slurm/env.sh
set -eo pipefail
# Keep CPU thread pools inside the cores SLURM granted (Ruche gives 4 GB per core).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), partition $SLURM_JOB_PARTITION, started $(date)"
echo "commit: $(git log -1 --oneline 2>/dev/null || echo unknown)"
echo "command: $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
"$@"
echo "finished $(date)"
