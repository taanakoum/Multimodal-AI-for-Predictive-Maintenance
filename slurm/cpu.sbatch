#!/bin/bash
# Runs one pdmfm command on a Ruche CPU node (window building, the Gate-1 probe).
#
#   sbatch --job-name=NAME [more sbatch options] slurm/cpu.sbatch COMMAND [ARGUMENTS...]
# Example:
#   sbatch --job-name=windows-paderborn slurm/cpu.sbatch python -m pdmfm.windows build paderborn --force
#
# cpu_short allows 1 h and one job at a time. For longer work pass --partition and --time
# for a partition with a longer limit (sinfo -s lists them).
#SBATCH --job-name=pdmfm-cpu
#SBATCH --output=runs/logs/%x-%j.out
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Submit this script with sbatch; see docs/RUCHE_RUNBOOK.md." >&2
    exit 2
fi
if [ "$#" -eq 0 ]; then
    echo "No command given after slurm/cpu.sbatch." >&2
    exit 2
fi
cd "$SLURM_SUBMIT_DIR" || exit 2
source slurm/env.sh
set -eo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), partition $SLURM_JOB_PARTITION, started $(date)"
echo "commit: $(git log -1 --oneline 2>/dev/null || echo unknown)"
echo "command: $*"
"$@"
echo "finished $(date)"
