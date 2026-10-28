#!/bin/bash
# Run on the Ruche LOGIN node (it has internet), once, before any GPU job:
#   source slurm/env.sh && bash slurm/prefetch_weights.sh
# Downloads the checkpoints of the encoder bank into $HF_HOME, so that jobs run with
# HF_HUB_OFFLINE=1 (set by slurm/env.sh inside jobs). About 4 GB.
set -euo pipefail
: "${HF_HOME:?source slurm/env.sh first}"
python - <<'PY'
from huggingface_hub import snapshot_download

for repo in [
    "paris-noah/Mantis-8M",
    "AutonLab/MOMENT-1-base",
    "facebook/dinov2-small",
    "Qwen/Qwen3-Embedding-0.6B",
    "MIT/ast-finetuned-audioset-10-10-0.4593",
]:
    path = snapshot_download(repo)
    print("ok", repo, path)
PY
echo "Loading every checkpoint offline is tested by the step-0 job on a compute node"
echo "(sbatch slurm/checks.sbatch), not here: no heavy work on the login node."
