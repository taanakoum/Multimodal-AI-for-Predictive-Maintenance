#!/bin/bash
# One-time Python environment on Ruche. Run on the LOGIN node (it has internet), from the
# repository root:
#   bash scripts/ruche/setup_env.sh
#   TORCH_INDEX=https://download.pytorch.org/whl/cu118 bash scripts/ruche/setup_env.sh  # older driver
# Creates $WORKDIR/pdmfm-venv with Python 3.11 managed by uv (no module needed), installs
# torch for CUDA, this package with the extras the GPU jobs use, and OmniFuser when
# ../OmniFuser exists. Afterwards every session and job runs:  source scripts/env/ruche.sh
set -euo pipefail
: "${WORKDIR:?WORKDIR is not set (run this on Ruche)}"
VENV="${PDMFM_VENV:-$WORKDIR/pdmfm-venv}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
EXTRAS="${PDMFM_EXTRAS:-encoders,xai,eda,dev}"
export UV_CACHE_DIR="$WORKDIR/uv-cache"  # keep caches off the small HOME quota

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user --upgrade uv
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.11 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install torch --index-url "$TORCH_INDEX"
uv pip install -e ".[$EXTRAS]"
if [[ -d ../OmniFuser ]]; then uv pip install -e ../OmniFuser; fi

python - <<'PY'
import torch

import pdmfm

print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
print("pdmfm from", pdmfm.__file__)
PY
echo "done. Every session/job: source scripts/env/ruche.sh  (activates $VENV)"
echo "GPU check (inside a gpu job): python -c 'import torch; print(torch.cuda.is_available())'"
