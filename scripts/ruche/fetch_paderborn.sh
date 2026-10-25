#!/bin/bash
# Paderborn's server refuses connections from the development machine. Fetch it on the
# Ruche LOGIN node instead (resumable, SHA-256 recorded in MANIFEST.json):
#   source scripts/env/ruche.sh && bash scripts/ruche/fetch_paderborn.sh
# Then copy $PDMFM_DATA_ROOT/raw/paderborn back to D:\Project14\data\raw\paderborn for
# local CPU work (e.g. with rsync or WinSCP).
set -euo pipefail
: "${PDMFM_DATA_ROOT:?source scripts/env/ruche.sh first}"
python -m pdmfm.fetch get paderborn
