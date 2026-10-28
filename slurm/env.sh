# Loads the pdmfm environment on Ruche, in a job or in a terminal:
#   source slurm/env.sh
# Edit the two names below once, after the installation (docs/RUCHE_RUNBOOK.md, section 7).

CONDA_MODULE="anaconda3/2023.09-0/none-none"  # a conda module listed by: module avail anaconda miniconda
CONDA_ENV="pdmfm"                           # the environment created in section 7.2

# Jobs submitted with --export=NONE start with an empty environment; load the system
# profile so that the module command exists.
if ! type module >/dev/null 2>&1; then
    source /etc/profile
fi
module purge
module load "$CONDA_MODULE"
# Ruche asks for "source activate", not "conda activate" (which edits ~/.bashrc).
source activate "$CONDA_ENV"

export PDMFM_DATA_ROOT="$WORKDIR/pdmfm_data"  # raw/, extracted/, interim/ (windows, features)
export HF_HOME="$WORKDIR/hf_cache"            # model weights, downloaded on the login node
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export HF_HUB_OFFLINE=1  # compute nodes read weights from the cache only
fi
export PYTHONUNBUFFERED=1  # logs show progress while a job runs
export PYTHONIOENCODING=utf-8
