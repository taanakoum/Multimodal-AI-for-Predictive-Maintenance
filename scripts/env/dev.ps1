# Optional helper; caller-provided paths take precedence.
if (-not $env:PDMFM_DATA_ROOT) { $env:PDMFM_DATA_ROOT = Join-Path $PWD 'data' }
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $env:PDMFM_DATA_ROOT 'hf_cache' }
if (-not $env:UV_PROJECT_ENVIRONMENT) { $env:UV_PROJECT_ENVIRONMENT = Join-Path $PWD '.venv' }
$env:UV_LINK_MODE = 'copy'
$env:PYTHONIOENCODING = 'utf-8'
$activation = Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts/Activate.ps1'
if (Test-Path -LiteralPath $activation) { . $activation }
Write-Host "pdmfm: data=$env:PDMFM_DATA_ROOT"
