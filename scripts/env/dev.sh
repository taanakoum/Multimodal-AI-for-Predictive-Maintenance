# Optional helper; caller-provided paths take precedence.
export PDMFM_DATA_ROOT="${PDMFM_DATA_ROOT:-$PWD/data}"
export HF_HOME="${HF_HOME:-$PDMFM_DATA_ROOT/hf_cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"
export UV_LINK_MODE=copy
export PYTHONIOENCODING=utf-8
if [ -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ]; then
    source "$UV_PROJECT_ENVIRONMENT/bin/activate"
elif [ -f "$UV_PROJECT_ENVIRONMENT/Scripts/activate" ]; then
    source "$UV_PROJECT_ENVIRONMENT/Scripts/activate"
fi
