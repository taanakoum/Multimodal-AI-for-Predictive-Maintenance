#!/bin/bash
# Run on the Ruche LOGIN node (it has internet) before any GPU job:
#   source scripts/env/ruche.sh && bash scripts/ruche/prefetch_weights.sh
# Downloads every pretrained checkpoint the plan uses into $HF_HOME so that
# compute jobs can run with HF_HUB_OFFLINE=1.
set -euo pipefail
: "${HF_HOME:?source scripts/env/ruche.sh first}"
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in [
    "paris-noah/Mantis-8M",
    "AutonLab/MOMENT-1-base",
    "facebook/dinov2-small",
    "Qwen/Qwen3-Embedding-0.6B",
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    "google/timesfm-2.5-200m-pytorch",
    "google/timesfm-3.0-pytorch",
]:
    path = snapshot_download(repo)
    print("ok", repo, path)
PY
