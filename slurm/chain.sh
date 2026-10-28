#!/bin/bash
# Submits the same job several times in a row, for work longer than the 24 h limit.
# Each part starts when the previous one has ended (finished, failed or out of time) and
# resumes where it stopped: feature extraction skips cached windows, and PatchTST
# pretraining continues from its last finished epoch.
#
# Usage, from the repository folder:
#   bash slurm/chain.sh PARTS [sbatch options] slurm/job.sbatch COMMAND [ARGUMENTS...]
# Example (2 parts of up to 24 h each):
#   bash slurm/chain.sh 2 --job-name=patchtst-fold1 slurm/job.sbatch \
#     python -m pdmfm.patchtst --fold 1 --device cuda
set -euo pipefail

if [ "$#" -lt 2 ] || ! [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
    echo "usage: bash slurm/chain.sh PARTS [sbatch options] slurm/job.sbatch COMMAND ..." >&2
    exit 2
fi
parts=$1
shift

job=$(sbatch --parsable "$@")
job=${job%%;*}
jobs=$job
echo "part 1/$parts: job $job"
for ((part = 2; part <= parts; part++)); do
    job=$(sbatch --parsable --dependency=afterany:"$job" "$@")
    job=${job%%;*}
    jobs="$jobs $job"
    echo "part $part/$parts: job $job (waits for the previous part)"
done
echo "To cancel this chain: scancel $jobs"
