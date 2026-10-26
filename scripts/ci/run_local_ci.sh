#!/bin/bash
# Reproduce the CI workflow locally in a fresh, throwaway environment: CPU torch, no
# datasets, no OmniFuser. Passing here predicts passing in CI (docs/CI.md):
#   bash scripts/ci/run_local_ci.sh          # Python 3.11
#   bash scripts/ci/run_local_ci.sh 3.12
# Needs uv. The environment is created in a temporary folder and deleted afterwards.
set -euo pipefail
PY="${1:-3.11}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export UV_PROJECT_ENVIRONMENT="$WORK/venv"  # uv run uses this, never the dev environment
export PDMFM_DATA_ROOT="$WORK/ci-data"       # empty: real-data tests skip, as in CI
export PYTHONIOENCODING=utf-8 UV_LINK_MODE=copy
unset PDMFM_NETWORK_TESTS OMP_NUM_THREADS
cd "$ROOT"
start=$(date +%s)

uv venv --python "$PY" "$UV_PROJECT_ENVIRONMENT"
uv pip install --python "$UV_PROJECT_ENVIRONMENT" torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$UV_PROJECT_ENVIRONMENT" -e ".[dev,eda]"

run() { uv run --no-sync "$@"; }
run ruff check pdmfm tests scripts
run ruff format --check pdmfm tests scripts
run actionlint -color
run python -m pdmfm.fetch list >/dev/null
run python -m pdmfm.resources
run pytest -rs -p no:cacheprovider --cov=pdmfm --cov-report=term:skip-covered \
  --junitxml="$WORK/junit.xml"
run pyright pdmfm || echo "pyright reported findings (advisory, as in CI)"
echo "local CI (Python $PY) passed in $(($(date +%s) - start)) s"
