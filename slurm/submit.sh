#!/bin/bash
# Submit a generated immutable manifest; optional further arguments go to sbatch.
# bash slurm/submit.sh runs/jobs/gate2.json 2 --dependency=afterok:12345
set -euo pipefail
if [ "$#" -lt 1 ]; then
    echo "Usage: bash slurm/submit.sh MANIFEST [WORKERS=2] [SBATCH_OPTIONS...]" >&2
    exit 2
fi
manifest=$1
workers=${2:-2}
shift
if [ "$#" -gt 0 ]; then shift; fi
mkdir -p runs/logs
count=$(python -m pdmfm.jobs count "$manifest" --workers "$workers")
sbatch --parsable --array="0-$((count - 1))%4" "$@" slurm/array.sbatch "$manifest" "$workers"
