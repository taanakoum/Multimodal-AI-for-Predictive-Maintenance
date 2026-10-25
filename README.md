# pdmfm: Project 14 multimodal predictive-maintenance model

Project 14 builds a multimodal predictive-maintenance model from frozen pretrained encoders:
- a sensor foundation model, picked at Gate 1;
- DINOv2 for photos and thermal images;
- Qwen3-Embedding for text.

The encoders' outputs are cached as tokens and fused by a small trained stage, chosen by a fusion search. Evidential heads, whose classes are text descriptions of faults, give each prediction a calibrated uncertainty and a trust value per modality.

- **Plan:** [Project 14 implementation plan](https://claude.ai/artifact/1M2DPuffJP1mRt2oiUujx5) (reference copy in `docs/PLAN.md`)
- **Progress, findings and decisions:** [`PROJECT_LOG.md`](PROJECT_LOG.md)
- **Earlier work reused here:** the OmniFuser reproduction in `../OmniFuser`, installed as an editable dependency

## Setup (Windows development machine)

Large files never live in this OneDrive folder: data, weights and the environment are on `D:\Project14`.

```powershell
uv venv D:\Project14\.venv --python 3.11
. .\scripts\env\dev.ps1                   # sets PDMFM_DATA_ROOT, HF_HOME, activates the venv
uv pip install -e ".[dev]" -e ..\OmniFuser
uv pip install -e ".[encoders,forecast,xai]" "momentfm @ git+https://github.com/moment-timeseries-foundation-model/moment.git"
pytest
```

`momentfm` comes from GitHub because its PyPI release (0.1.4) pins transformers 4.33.3 and numpy 1.25.2.

## Data

```powershell
python -m pdmfm.fetch list                # registered datasets, roles, licences
python -m pdmfm.fetch plan kaist          # files, sizes and publisher hashes; downloads nothing
python -m pdmfm.fetch get kaist           # resumable; verifies size and SHA-256
python -m pdmfm.fetch extract kaist       # zip/rar archives into extracted/kaist
```

Downloads land in `$PDMFM_DATA_ROOT/raw/<dataset>`, each with a `MANIFEST.json` that records the URL, size and SHA-256 of every file. Two datasets need a manual step; see `PROJECT_LOG.md`:
- **Paderborn:** its host refuses connections from this machine; fetch it on Ruche or through a VPN.
- **PHM 2010:** the original download host is gone; the remaining mirrors need an account.

## Layout

```text
pdmfm/          package: paths, dataset registry (sources.py), downloader (fetch.py)
tests/          offline tests (pytest)
scripts/env/    environment scripts for the dev machine and Ruche
docs/           plan copy and notes
configs/        per-stage YAML configs (from week 2)
```

## Licence

Code: MIT (see `LICENSE`). The datasets keep their own licences, listed by `python -m pdmfm.fetch list`. Paderborn is CC BY-NC 4.0, so model weights trained on it are for non-commercial use only.
