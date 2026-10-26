# Source on Ruche (login or compute node):  source scripts/env/ruche.sh
export PDMFM_DATA_ROOT="$WORKDIR/pdmfm_data"
export HF_HOME="$WORKDIR/hf_cache"
export UV_CACHE_DIR="$WORKDIR/uv-cache"
# Environment created by scripts/ruche/setup_env.sh.
PDMFM_VENV="${PDMFM_VENV:-$WORKDIR/pdmfm-venv}"
# shellcheck disable=SC1091
if [[ -f "$PDMFM_VENV/bin/activate" ]]; then source "$PDMFM_VENV/bin/activate"; fi
# Compute nodes may have no internet: weights are pre-downloaded on the login node.
if [[ -n "$SLURM_JOB_ID" ]]; then export HF_HUB_OFFLINE=1; fi
export PYTHONIOENCODING=utf-8
