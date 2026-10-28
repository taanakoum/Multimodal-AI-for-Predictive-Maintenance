#!/bin/bash
# Paderborn's server refuses connections from the development machine. Fetch it on the
# Ruche LOGIN node (32 RAR archives, resumable, SHA-256 recorded in MANIFEST.json):
#   source slurm/env.sh && bash slurm/fetch_paderborn.sh
# Extraction and the window store are built in a CPU job afterwards
# (docs/RUCHE_RUNBOOK.md, section 13).
set -euo pipefail
: "${PDMFM_DATA_ROOT:?source slurm/env.sh first}"
python -m pdmfm.fetch get paderborn --workers 4
python -m pdmfm.fetch verify paderborn
