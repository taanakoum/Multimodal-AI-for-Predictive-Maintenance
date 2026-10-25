# Source on Ruche (login or compute node):  source scripts/env/ruche.sh
export PDMFM_DATA_ROOT="$WORKDIR/pdmfm_data"
export HF_HOME="$WORKDIR/hf_cache"
# Compute nodes may have no internet: weights are pre-downloaded on the login node.
if [[ -n "$SLURM_JOB_ID" ]]; then export HF_HUB_OFFLINE=1; fi
export PYTHONIOENCODING=utf-8
