# Loads the pdmfm environment on Ruche, in a job or in a terminal:
#   source slurm/env.sh
# Edit defaults for your account; confirm module availability on Ruche first.

CONDA_MODULE="${CONDA_MODULE:-anaconda3/2023.09-0/none-none}"
CONDA_ENV="${CONDA_ENV:-pdmfm}"

# Jobs submitted with --export=NONE start with an empty environment; load the system
# profile so that the module command exists.
if ! type module >/dev/null 2>&1; then
    source /etc/profile
fi
module purge || return 1
module load "$CONDA_MODULE" || return 1
# Ruche asks for "source activate", not "conda activate" (which edits ~/.bashrc).
source activate "$CONDA_ENV" || return 1

: "${WORKDIR:?Ruche WORKDIR is not set; check your account environment}"
export PDMFM_DATA_ROOT="${PDMFM_DATA_ROOT:-$WORKDIR/pdmfm_data}"
export HF_HOME="${HF_HOME:-$WORKDIR/hf_cache}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export HF_HUB_OFFLINE=1  # compute nodes read weights from the cache only
fi
export PYTHONUNBUFFERED=1  # logs show progress while a job runs
export PYTHONIOENCODING=utf-8
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
