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

Downloads land in `$PDMFM_DATA_ROOT/raw/<dataset>`, each with a `MANIFEST.json` that records the URL, size and SHA-256 of every file. `python -m pdmfm.fetch verify <dataset>` re-hashes a folder against its manifest and records files placed by hand.

Two datasets need steps only you can take. [`docs/USER_ACTIONS.md`](docs/USER_ACTIONS.md) has the commands and the acceptance tests for each:
- **Paderborn:** its host refuses connections from this machine; fetch it on Ruche or through a VPN.
- **PHM 2010:** the original download host is gone; get the zips from a mirror and verify them.

## Exploratory data analysis

```powershell
python scripts\eda\run_eda.py             # every dataset on disk; checks CPU and RAM first
```

The findings and their consequences for preprocessing, splits and baselines are in [`docs/EDA_REPORT.md`](docs/EDA_REPORT.md). Every sensor channel goes through `pdmfm.preprocess.model_input`: resampled to 12 kHz, then band-limited to 5 kHz, because recorders' anti-alias roll-offs above 5 kHz identify the source, and on CWRU even the class.

## Tests and CI

```powershell
pytest                                    # real-data tests run when the data is on disk, skip otherwise
bash scripts/ci/run_local_ci.sh           # CI reproduced in a fresh environment (Git Bash, uv)
pre-commit install                        # same lint and format checks before each commit
```

GitHub Actions runs lint, then tests on Ubuntu and Windows with Python 3.11 and 3.12, then an advisory type check. A weekly workflow checks that the data sources still serve the files we used. See [`docs/CI.md`](docs/CI.md).

## Heavy jobs and Ruche

Scripts that load the CPU call `pdmfm.resources.claim_resources` first. `python -m pdmfm.resources` shows idle CPUs, free RAM and busy processes. On Ruche, `scripts/ruche/setup_env.sh` builds the environment once and `source scripts/env/ruche.sh` loads it in every session. `scripts/ruche/net_check.sbatch` tests the compute nodes' internet access, and `prefetch_weights.sh` downloads the model weights on the login node.

## Layout

```text
pdmfm/            package
  paths.py        data root and folders (PDMFM_DATA_ROOT)
  sources.py      dataset registry: URLs, hashes, licences, split lists
  fetch.py        resumable, hash-checked downloader; extract; verify
  schema.py       Sample, ChannelWindow, ImageItem; as-of join of images to windows
  preprocess.py   resampling, 5 kHz band limit, windows, T/F/E arms, descriptors, FPT, RUL
  resources.py    CPU/RAM guard and thread caps
  datasets/       one reader per dataset
tests/            pytest suite (synthetic data; real-data, acceptance and network tests are marked)
scripts/eda/      exploratory data analysis
scripts/ci/       local CI reproduction, data-source check
scripts/ruche/    Ruche environment, network check, Paderborn fetch, weight prefetch
scripts/env/      environment scripts for the dev machine and Ruche
docs/             plan copy, EDA report and figures, CI, your action list
```

## Licence

Code: MIT (see `LICENSE`). The datasets keep their own licences, listed by `python -m pdmfm.fetch list`. Paderborn is CC BY-NC 4.0, so model weights trained on it are for non-commercial use only.
