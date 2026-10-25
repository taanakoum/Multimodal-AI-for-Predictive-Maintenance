# Dot-source in PowerShell before working:  . .\scripts\env\dev.ps1
$env:PDMFM_DATA_ROOT = "D:\Project14\data"
$env:HF_HOME = "D:\Project14\hf_cache"
$env:UV_PROJECT_ENVIRONMENT = "D:\Project14\.venv"
$env:UV_LINK_MODE = "copy"
$env:PYTHONIOENCODING = "utf-8"
& "D:\Project14\.venv\Scripts\Activate.ps1"
Write-Host "pdmfm dev env: data=$env:PDMFM_DATA_ROOT hf=$env:HF_HOME"
