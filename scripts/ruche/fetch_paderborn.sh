#!/bin/bash
# Paderborn's server refuses connections from the development machine. Fetch it on the
# Ruche LOGIN node instead (32 RAR archives, resumable, SHA-256 recorded in MANIFEST.json):
#   source scripts/env/ruche.sh && bash scripts/ruche/fetch_paderborn.sh
# Then copy the raw folder back for local CPU work (docs/USER_ACTIONS.md, step 2).
set -euo pipefail
: "${PDMFM_DATA_ROOT:?source scripts/env/ruche.sh first}"
python -m pdmfm.fetch get paderborn --workers 4
python -m pdmfm.fetch verify paderborn
echo
echo "Copy to the development machine (run THERE, in PowerShell):"
echo "  scp -r ${USER}@ruche.mesocentre.universite-paris-saclay.fr:${PDMFM_DATA_ROOT}/raw/paderborn D:\\Project14\\data\\raw\\"
